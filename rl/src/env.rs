use anyhow::{Context, Result, anyhow};
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::{Value, json};
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::process::Stdio;
use std::time::{Duration, Instant};
use tokio::process::{Child, Command};
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;

pub const MAX_STEPS: usize = 20;
pub const MAX_BUTTONS: usize = 5;
const STEP_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug, Clone, Deserialize)]
pub struct Button {
    pub index: usize,
    pub text: String,
    pub clickable: bool,
    pub disabled: bool,
}

#[derive(Debug, Clone)]
pub struct Goal {
    pub item: String,
    pub qty: u32,
}

#[derive(Debug, Clone)]
pub struct Observation {
    pub screen: String,
    pub goal: Goal,
    pub buttons: Vec<Button>,
    pub product: Option<String>,
    pub qty: Option<u32>,
    pub cart: Vec<(String, u32)>,
}

#[derive(Debug, Clone)]
pub enum Action {
    Click(usize),
    Wait,
}

#[derive(Debug, Clone)]
pub struct StepResult {
    pub observation: Observation,
    pub reward: f64,
    pub done: bool,
    pub step: usize,
}

pub struct Env {
    child: Child,
    ws: tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
    session_id: Option<String>,
    next_cdp_id: u64,
    step_count: usize,
    goal: Goal,
    seed: u64,
    episode: u64,
    log: File,
    last_potential: f64,
    popup_p: f64,
    delay_p: f64,
}

impl Env {
    pub async fn launch(seed: u64, log_path: &str) -> Result<Env> {
        if let Some(parent) = Path::new(log_path).parent() {
            std::fs::create_dir_all(parent).with_context(|| format!("create {parent:?}"))?;
        }
        let log = File::create(log_path).with_context(|| format!("create {log_path}"))?;
        let chrome = std::env::var("CHROME_BIN").unwrap_or_else(|_| "chromium".into());

        let child = Command::new(chrome)
            .arg("--headless=new")
            .arg("--disable-gpu")
            .arg("--no-sandbox")
            .arg("--remote-debugging-port=9222")
            .arg("--disable-extensions")
            .arg("--disable-default-apps")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .context("launch chromium")?;

        let ws_url = wait_for_devtools().await?;
        let mut env = Env {
            child,
            ws: connect_async(&ws_url).await?.0,
            session_id: None,
            next_cdp_id: 1,
            step_count: 0,
            goal: Goal {
                item: String::new(),
                qty: 0,
            },
            seed,
            episode: 0,
            log,
            last_potential: 0.0,
            popup_p: 0.15,
            delay_p: 0.10,
        };

        let target = env
            .cdp("Target.createTarget", json!({ "url": "about:blank" }))
            .await?;
        let target_id = target["targetId"].as_str().context("no targetId")?;
        let session = env
            .cdp(
                "Target.attachToTarget",
                json!({ "targetId": target_id, "flatten": true }),
            )
            .await?;
        env.session_id = Some(
            session["sessionId"]
                .as_str()
                .context("no sessionId")?
                .to_string(),
        );
        env.cdp("Page.enable", json!({})).await?;
        env.cdp(
            "Page.addScriptToEvaluateOnNewDocument",
            json!({ "source": "window.alert = () => {}; window.confirm = () => true;" }),
        )
        .await?;

        Ok(env)
    }

    pub async fn reset(&mut self, task: &str, seed: u64) -> Result<Observation> {
        self.goal = parse_task(task)?;
        self.seed = seed;
        self.step_count = 0;
        self.episode += 1;
        self.last_potential = 0.0;

        let url = format!(
            "{task}&seed={seed}&popup_p={}&delay_p={}",
            self.popup_p, self.delay_p
        );
        self.cdp("Page.navigate", json!({ "url": url })).await?;
        self.wait_for_page().await?;
        self.observe().await
    }

    pub fn set_page_params(&mut self, popup_p: f64, delay_p: f64) {
        self.popup_p = popup_p;
        self.delay_p = delay_p;
    }

    pub async fn step(&mut self, action: Action) -> Result<StepResult> {
        let step = self.step_count + 1;
        let started = Instant::now();
        let before = self.observe().await?;
        let popup_before = self.is_popup_visible().await?;
        if popup_before {
            self.dismiss_popup().await?;
        }

        match &action {
            Action::Click(index) => {
                let button = before
                    .buttons
                    .get(*index)
                    .ok_or_else(|| anyhow!("button index {index} out of range"))?;
                if !button.clickable {
                    let result = StepResult {
                        observation: before.clone(),
                        reward: 0.0,
                        done: false,
                        step: self.step_count,
                    };
                    self.write_log(step, &before, &action, &result, started, popup_before)?;
                    return Ok(result);
                }
                let (x, y) = self.button_center(*index).await?;
                self.click_at(x, y).await?;
            }
            Action::Wait => tokio::time::sleep(Duration::from_millis(300)).await,
        }

        let popup_after = self.is_popup_visible().await?;
        if popup_after {
            self.dismiss_popup().await?;
        }

        self.step_count += 1;
        let timed_out = started.elapsed() > STEP_TIMEOUT;
        let (mut reward, eval_done) = if timed_out {
            (-0.1, true)
        } else {
            self.evaluate().await?
        };
        let truncated = self.step_count >= MAX_STEPS;
        if truncated && !eval_done {
            reward = -0.1;
        }
        let done = eval_done || truncated;
        let observation = self.observe().await?;

        if !done {
            let next_potential = self.potential(&observation);
            reward += 0.95 * next_potential - self.last_potential - 0.01;
            self.last_potential = next_potential;
        }

        let result = StepResult {
            observation,
            reward,
            done,
            step: self.step_count,
        };
        self.write_log(
            step,
            &before,
            &action,
            &result,
            started,
            popup_before || popup_after,
        )?;
        Ok(result)
    }

    async fn wait_for_page(&mut self) -> Result<()> {
        for _ in 0..100 {
            let ready = self
                .evaluate_js(
                    "document.readyState === 'complete' && !!document.querySelector('.screen.active')",
                )
                .await?;
            if ready.as_bool() == Some(true) {
                return Ok(());
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        Err(anyhow!("page did not become ready"))
    }

    async fn cdp(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_cdp_id;
        self.next_cdp_id += 1;
        let mut message = json!({ "id": id, "method": method, "params": params });
        if let Some(session_id) = &self.session_id {
            message["sessionId"] = json!(session_id);
        }
        self.ws.send(Message::Text(message.to_string())).await?;

        loop {
            match self.ws.next().await {
                Some(Ok(Message::Text(text))) => {
                    let value: Value = serde_json::from_str(&text)?;
                    if value.get("id").and_then(Value::as_u64) != Some(id) {
                        continue;
                    }
                    if let Some(error) = value.get("error") {
                        return Err(anyhow!("CDP error: {error}"));
                    }
                    return Ok(value["result"].clone());
                }
                Some(Ok(_)) => {}
                Some(Err(error)) => return Err(anyhow!("websocket: {error}")),
                None => return Err(anyhow!("websocket closed")),
            }
        }
    }

    async fn evaluate_js(&mut self, expression: &str) -> Result<Value> {
        let result = self
            .cdp(
                "Runtime.evaluate",
                json!({
                    "expression": expression, "returnByValue": true, "awaitPromise": true
                }),
            )
            .await?;
        Ok(result["result"]["value"].clone())
    }

    async fn observe(&mut self) -> Result<Observation> {
        let value = self
            .evaluate_js(
                r#"
                (function() {
                    const active = document.querySelector('.screen.active');
                    const screen = active ? active.id.replace('screen-', '') : 'unknown';
                    const buttons = [...document.querySelectorAll('.screen.active button')]
                        .filter(button => button.offsetParent !== null)
                        .map((button, index) => ({
                            index,
                            text: (button.textContent || '').trim(),
                            clickable: !button.disabled && !button.classList.contains('btn-hidden'),
                            disabled: button.disabled
                        }));

                    const productName = document.getElementById('product-name');
                    const product = screen === 'product' && productName
                        ? productName.textContent.trim().toLowerCase().replace(/\s+/g, '-')
                        : null;
                    const qtyValue = document.getElementById('qty-val');
                    const qty = screen === 'product' && qtyValue
                        ? parseInt(qtyValue.textContent, 10)
                        : null;
                    const cart = screen === 'cart'
                        ? [...document.querySelectorAll('#cart-list .cart-item')].map(item => {
                            const name = item.querySelector('.name').textContent.trim().toLowerCase().replace(/\s+/g, '-');
                            const match = item.querySelector('.qty-text').textContent.match(/×\s*(\d+)/);
                            return [name, match ? parseInt(match[1], 10) : 0];
                        })
                        : [];

                    return { screen, buttons, product, qty, cart };
                })()
            "#,
            )
            .await?;

        let cart = value["cart"]
            .as_array()
            .map(|rows| {
                rows.iter()
                    .filter_map(|row| {
                        let pair = row.as_array()?;
                        Some((
                            pair.first()?.as_str()?.to_string(),
                            pair.get(1)?.as_u64()? as u32,
                        ))
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(Observation {
            screen: value["screen"].as_str().unwrap_or("unknown").to_string(),
            goal: self.goal.clone(),
            buttons: serde_json::from_value(value["buttons"].clone()).unwrap_or_default(),
            product: value["product"].as_str().map(str::to_string),
            qty: value["qty"].as_u64().map(|qty| qty as u32),
            cart,
        })
    }

    async fn button_center(&mut self, index: usize) -> Result<(f64, f64)> {
        let value = self
            .evaluate_js(&format!(
                r#"
                (function() {{
                    const button = [...document.querySelectorAll('.screen.active button')]
                        .filter(button => button.offsetParent !== null)[{index}];
                    if (!button) return null;
                    const rect = button.getBoundingClientRect();
                    return [rect.x + rect.width / 2, rect.y + rect.height / 2];
                }})()
            "#
            ))
            .await?;
        let rect = value.as_array().context("no button rect")?;
        Ok((
            rect[0].as_f64().unwrap_or(0.0),
            rect[1].as_f64().unwrap_or(0.0),
        ))
    }

    async fn click_at(&mut self, x: f64, y: f64) -> Result<()> {
        for event in ["mousePressed", "mouseReleased"] {
            self.cdp(
                "Input.dispatchMouseEvent",
                json!({
                    "type": event, "x": x, "y": y, "button": "left", "clickCount": 1
                }),
            )
            .await?;
        }
        tokio::time::sleep(Duration::from_millis(30)).await;
        Ok(())
    }

    async fn is_popup_visible(&mut self) -> Result<bool> {
        Ok(self
            .evaluate_js("document.getElementById('overlay').classList.contains('active')")
            .await?
            .as_bool()
            .unwrap_or(false))
    }

    async fn dismiss_popup(&mut self) -> Result<()> {
        let value = self
            .evaluate_js(
                r#"
                (function() {
                    const button = document.getElementById('dismiss-btn');
                    if (!button) return null;
                    const rect = button.getBoundingClientRect();
                    return [rect.x + rect.width / 2, rect.y + rect.height / 2];
                })()
            "#,
            )
            .await?;
        if let Some(rect) = value.as_array() {
            self.click_at(
                rect[0].as_f64().unwrap_or(0.0),
                rect[1].as_f64().unwrap_or(0.0),
            )
            .await?;
        }
        Ok(())
    }

    async fn evaluate(&mut self) -> Result<(f64, bool)> {
        let value = self
            .evaluate_js(
                r#"
                (function() {
                    const done = document.getElementById('screen-done');
                    if (!done || !done.classList.contains('active')) return null;
                    return [...done.querySelectorAll('#order-list .cart-item')].map(item => {
                        const name = item.querySelector('.name').textContent.trim();
                        const slug = name.match(/\(([^)]+)\)$/)?.[1] ?? name;
                        const qty = parseInt(item.querySelector('.qty-text').textContent.replace(/[^0-9]/g, ''), 10);
                        return { slug, qty };
                    });
                })()
            "#,
            )
            .await?;
        if value.is_null() {
            return Ok((0.0, false));
        }

        let items: Vec<Value> = serde_json::from_value(value).unwrap_or_default();
        let correct = items.iter().any(|item| {
            item["slug"].as_str() == Some(&self.goal.item)
                && item["qty"].as_u64() == Some(self.goal.qty as u64)
        });
        Ok(if correct { (1.0, true) } else { (-0.5, true) })
    }

    fn potential(&self, observation: &Observation) -> f64 {
        match observation.screen.as_str() {
            "product" => {
                if observation.product.as_deref() == Some(self.goal.item.as_str()) {
                    0.2
                } else {
                    0.0
                }
            }
            "cart" => {
                if observation.cart.len() == 1
                    && observation.cart[0].0 == self.goal.item
                    && observation.cart[0].1 == self.goal.qty
                {
                    0.5
                } else {
                    0.1
                }
            }
            "done" => 0.5,
            _ => 0.0,
        }
    }

    fn write_log(
        &mut self,
        step: usize,
        before: &Observation,
        action: &Action,
        result: &StepResult,
        started: Instant,
        popup: bool,
    ) -> Result<()> {
        let action = match action {
            Action::Click(index) => format!("click({index})"),
            Action::Wait => "wait".to_string(),
        };
        let line = json!({
            "episode": self.episode,
            "seed": self.seed,
            "goal": format!("{} x{}", self.goal.item, self.goal.qty),
            "step": step,
            "observation": {
                "screen": before.screen,
                "buttons": before.buttons.iter().map(|button| json!({
                    "text": button.text,
                    "clickable": button.clickable,
                    "disabled": button.disabled,
                })).collect::<Vec<_>>(),
                "product": before.product,
                "qty": before.qty,
                "cart": before.cart.iter().map(|(slug, qty)| json!({
                    "slug": slug,
                    "qty": qty,
                })).collect::<Vec<_>>(),
            },
            "action": action,
            "reward": result.reward,
            "done": result.done,
            "elapsed_ms": started.elapsed().as_millis() as u64,
            "popup": popup,
            "popup_p": self.popup_p,
            "delay_p": self.delay_p,
        });
        writeln!(self.log, "{line}")?;
        Ok(())
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

fn parse_task(task: &str) -> Result<Goal> {
    let url = url::Url::parse(task).context("invalid task url")?;
    let mut item = "blue-mug".to_string();
    let mut qty = 2;

    for (key, value) in url.query_pairs() {
        match key.as_ref() {
            "item" => item = value.into_owned(),
            "qty" => qty = value.parse().unwrap_or(qty),
            _ => {}
        }
    }

    Ok(Goal { item, qty })
}

async fn wait_for_devtools() -> Result<String> {
    for _ in 0..50 {
        if let Ok(response) = reqwest::get("http://127.0.0.1:9222/json/version").await
            && let Ok(value) = response.json::<Value>().await
            && let Some(ws_url) = value["webSocketDebuggerUrl"].as_str()
        {
            return Ok(ws_url.to_string());
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    Err(anyhow!("devtools not ready"))
}
