mod agent;
mod env;

use agent::{Agent, WAIT_ACTION};
use anyhow::{Context, Result, anyhow};
use env::{Action, Env, MAX_STEPS, Observation};
use serde_json::json;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

const ITEMS: [&str; 4] = ["blue-mug", "red-shirt", "green-plant", "black-pen"];

struct Options {
    command: String,
    url: String,
    seeds: Vec<u64>,
    episodes: usize,
    eval_episodes: usize,
    alpha: f64,
    gamma: f64,
    epsilon: f64,
    min_epsilon: f64,
    popup_p: f64,
    delay_p: f64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            command: "demo".into(),
            url: "http://localhost:8000/index.html".into(),
            seeds: vec![1, 2, 3],
            episodes: 20,
            eval_episodes: 20,
            alpha: 0.2,
            gamma: 0.95,
            epsilon: 1.0,
            min_epsilon: 0.05,
            popup_p: 0.15,
            delay_p: 0.10,
        }
    }
}

impl Options {
    fn parse() -> Result<Self> {
        let mut args = std::env::args().skip(1);
        let mut options = Options {
            command: args.next().unwrap_or_else(|| "demo".into()),
            ..Options::default()
        };

        while let Some(flag) = args.next() {
            let value = args
                .next()
                .ok_or_else(|| anyhow!("missing value for {flag}"))?;
            match flag.as_str() {
                "--url" => options.url = value,
                "--seeds" => options.seeds = parse_seeds(&value)?,
                "--episodes" => options.episodes = value.parse().context("invalid --episodes")?,
                "--eval-episodes" => {
                    options.eval_episodes = value.parse().context("invalid --eval-episodes")?
                }
                "--alpha" => options.alpha = value.parse().context("invalid --alpha")?,
                "--gamma" => options.gamma = value.parse().context("invalid --gamma")?,
                "--epsilon" => options.epsilon = value.parse().context("invalid --epsilon")?,
                "--min-epsilon" => {
                    options.min_epsilon = value.parse().context("invalid --min-epsilon")?
                }
                "--popup-p" => options.popup_p = value.parse().context("invalid --popup-p")?,
                "--delay-p" => options.delay_p = value.parse().context("invalid --delay-p")?,
                _ => return Err(anyhow!("unknown option: {flag}")),
            }
        }

        if options.seeds.is_empty() {
            return Err(anyhow!("--seeds must contain at least one seed"));
        }
        if options.episodes == 0 || options.eval_episodes == 0 {
            return Err(anyhow!("episode counts must be greater than zero"));
        }
        Ok(options)
    }
}

struct EpisodeResult {
    success: bool,
    steps: usize,
    reward: f64,
    goal: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let options = Options::parse()?;

    match options.command.as_str() {
        "demo" => run_demo(&options).await,
        "random" => run_random(&options).await,
        "train" => run_train(&options).await,
        "eval" => run_eval(&options).await,
        "help" | "--help" => {
            print_usage();
            Ok(())
        }
        command => Err(anyhow!(
            "unknown command {command}; expected demo, random, train, eval, or help"
        )),
    }
}

async fn run_demo(options: &Options) -> Result<()> {
    let seed = options.seeds[0];
    let task = format!("{}?item=blue-mug&qty=2", options.url);
    let log_path = format!("logs/demo_seed_{seed}.jsonl");
    let mut env = Env::launch(seed, &log_path).await?;
    let mut observation = env.reset(&task, seed).await?;

    for _ in 0..MAX_STEPS {
        let result = env.step(pick_action(&observation)).await?;
        println!(
            "step {}  screen={:?}  reward={}  done={}",
            result.step, result.observation.screen, result.reward, result.done
        );
        if result.done {
            break;
        }
        observation = result.observation;
    }

    Ok(())
}

async fn run_random(options: &Options) -> Result<()> {
    let tasks = all_tasks(&options.url);
    let mut total_success = 0usize;
    let mut total_episodes = 0usize;

    for &seed in &options.seeds {
        let log_path = format!("logs/random_seed_{seed}.jsonl");
        let metrics_path = format!("logs/random_metrics_seed_{seed}.jsonl");
        let mut env = Env::launch(seed, &log_path).await?;
        let mut agent = Agent::random(seed);
        let mut metrics = File::create(&metrics_path)?;
        let started = Instant::now();
        let mut success = 0usize;
        let mut episodes = 0usize;

        for episode in 0..options.episodes {
            for task in &tasks {
                let result = run_episode(&mut env, &mut agent, task, seed, false).await?;
                success += result.success as usize;
                episodes += 1;
                writeln!(
                    metrics,
                    "{}",
                    json!({
                        "phase": "random",
                        "seed": seed,
                        "episode": episode,
                        "goal": result.goal,
                        "success": result.success,
                        "steps": result.steps,
                        "reward": result.reward,
                    })
                )?;
            }
        }

        let rate = success as f64 / episodes as f64;
        println!(
            "random seed {seed}: {success}/{episodes} ({:.1}%) in {:.1}s",
            rate * 100.0,
            started.elapsed().as_secs_f64()
        );
        total_success += success;
        total_episodes += episodes;
    }

    let rate = total_success as f64 / total_episodes as f64;
    println!(
        "random overall: {total_success}/{total_episodes} ({:.1}%)",
        rate * 100.0
    );
    Ok(())
}

async fn run_train(options: &Options) -> Result<()> {
    let tasks = all_tasks(&options.url);

    for &seed in &options.seeds {
        let log_path = format!("logs/learning_seed_{seed}.jsonl");
        let metrics_path = format!("logs/learning_metrics_seed_{seed}.jsonl");
        let mut env = Env::launch(seed, &log_path).await?;
        let mut agent = Agent::learning(seed, options.alpha, options.gamma, options.epsilon);
        let mut metrics = File::create(&metrics_path)?;
        let started = Instant::now();
        let total_episodes = options.episodes * tasks.len();
        let mut success = 0usize;
        let mut episodes = 0usize;

        for _ in 0..options.episodes {
            let progress = episodes as f64 / total_episodes as f64;
            let epsilon = options.epsilon + (options.min_epsilon - options.epsilon) * progress;
            agent.set_epsilon(epsilon);

            for task in &tasks {
                let result = run_episode(&mut env, &mut agent, task, seed, true).await?;
                success += result.success as usize;
                episodes += 1;
                writeln!(
                    metrics,
                    "{}",
                    json!({
                        "phase": "train",
                        "seed": seed,
                        "episode": episodes,
                        "goal": result.goal,
                        "success": result.success,
                        "steps": result.steps,
                        "reward": result.reward,
                        "epsilon": epsilon,
                    })
                )?;
            }
        }

        let rate = success as f64 / episodes as f64;
        println!(
            "train seed {seed}: {success}/{episodes} ({:.1}%) in {:.1}s",
            rate * 100.0,
            started.elapsed().as_secs_f64()
        );

        let model_path = format!("models/qtable_seed_{seed}.json");
        agent.save(&model_path)?;

        let eval_seed = seed + 10_000;
        let eval_metrics_path = format!("logs/learning_eval_metrics_seed_{seed}.jsonl");
        let mut eval_metrics = File::create(&eval_metrics_path)?;
        let eval_started = Instant::now();
        let mut eval_success = 0usize;
        let mut eval_episodes = 0usize;
        agent.set_epsilon(0.0);

        for episode in 0..options.eval_episodes {
            for task in &tasks {
                let result = run_episode(&mut env, &mut agent, task, eval_seed, false).await?;
                eval_success += result.success as usize;
                eval_episodes += 1;
                writeln!(
                    eval_metrics,
                    "{}",
                    json!({
                        "phase": "eval",
                        "seed": seed,
                        "episode": episode,
                        "goal": result.goal,
                        "success": result.success,
                        "steps": result.steps,
                        "reward": result.reward,
                    })
                )?;
            }
        }

        let eval_rate = eval_success as f64 / eval_episodes as f64;
        println!(
            "eval seed {seed}: {eval_success}/{eval_episodes} ({:.1}%) in {:.1}s, model {model_path}",
            eval_rate * 100.0,
            eval_started.elapsed().as_secs_f64()
        );
    }

    Ok(())
}

async fn run_eval(options: &Options) -> Result<()> {
    let tasks = all_tasks(&options.url);

    for &seed in &options.seeds {
        let model_path = format!("models/qtable_seed_{seed}.json");
        let steps_path = format!(
            "logs/eval_steps_popup_{}_seed_{seed}.jsonl",
            options.popup_p
        );
        let metrics_path = format!(
            "logs/eval_metrics_popup_{}_seed_{seed}.jsonl",
            options.popup_p
        );
        let mut env = Env::launch(seed, &steps_path).await?;
        env.set_page_params(options.popup_p, options.delay_p);
        let mut agent = Agent::load(&model_path, seed)?;
        let mut metrics = File::create(&metrics_path)?;
        let started = Instant::now();
        let mut success = 0usize;
        let mut episodes = 0usize;

        for episode in 0..options.eval_episodes {
            for task in &tasks {
                let result = run_episode(&mut env, &mut agent, task, seed + 20_000, false).await?;
                success += result.success as usize;
                episodes += 1;
                writeln!(
                    metrics,
                    "{}",
                    json!({
                        "phase": "eval_popup",
                        "popup_p": options.popup_p,
                        "delay_p": options.delay_p,
                        "seed": seed,
                        "episode": episode,
                        "goal": result.goal,
                        "success": result.success,
                        "steps": result.steps,
                        "reward": result.reward,
                    })
                )?;
            }
        }

        let rate = success as f64 / episodes as f64;
        println!(
            "eval popup_p={} seed {seed}: {success}/{episodes} ({:.1}%) in {:.1}s",
            options.popup_p,
            rate * 100.0,
            started.elapsed().as_secs_f64()
        );
    }

    Ok(())
}

async fn run_episode(
    env: &mut Env,
    agent: &mut Agent,
    task: &str,
    seed: u64,
    training: bool,
) -> Result<EpisodeResult> {
    let mut observation = env.reset(task, seed).await?;
    let goal = format!("{} x{}", observation.goal.item, observation.goal.qty);
    let mut total_reward = 0.0;
    let mut steps = 0;

    loop {
        let action = agent.choose(&observation, training);
        let result = env.step(action_from_code(action)).await?;
        agent.update(
            &observation,
            action,
            result.reward,
            &result.observation,
            result.done,
        );
        total_reward += result.reward;
        steps += 1;

        if result.done {
            return Ok(EpisodeResult {
                success: result.reward > 0.0,
                steps,
                reward: total_reward,
                goal,
            });
        }
        if steps >= MAX_STEPS {
            return Ok(EpisodeResult {
                success: false,
                steps,
                reward: total_reward,
                goal,
            });
        }
        observation = result.observation;
    }
}

fn action_from_code(code: usize) -> Action {
    if code == WAIT_ACTION {
        Action::Wait
    } else {
        Action::Click(code)
    }
}

fn all_tasks(url: &str) -> Vec<String> {
    let mut tasks = Vec::with_capacity(ITEMS.len() * 3);
    for item in ITEMS {
        for qty in 1..=3 {
            tasks.push(format!("{url}?item={item}&qty={qty}"));
        }
    }
    tasks
}

fn parse_seeds(value: &str) -> Result<Vec<u64>> {
    value
        .split(',')
        .map(|seed| {
            seed.trim()
                .parse()
                .with_context(|| format!("invalid seed: {seed}"))
        })
        .collect()
}

fn print_usage() {
    println!(
        r#"MiniShop environment and agents

Usage:
  cargo run -- demo  [--url URL] [--seeds 1]
  cargo run -- random [--url URL] [--seeds 1,2,3] [--episodes 20]
  cargo run -- train  [--url URL] [--seeds 1,2,3] [--episodes 20]
                      [--eval-episodes 20] [--alpha 0.2] [--gamma 0.95]
                      [--epsilon 1.0] [--min-epsilon 0.05]
  cargo run -- eval   [--url URL] [--seeds 1,2,3] [--eval-episodes 7]
                      [--popup-p 0.15] [--delay-p 0.10]

--episodes is the number of attempts per goal. There are 12 goals
(4 items x 3 quantities). Training saves models to models/qtable_seed_<seed>.json
and metrics to logs/."#
    );
}

fn pick_action(observation: &Observation) -> Action {
    let target = match observation.screen.as_str() {
        "catalog" => "View",
        "product" => "Add to cart",
        "cart" => "Checkout",
        "newsletter" => "Back",
        _ => return Action::Wait,
    };

    observation
        .buttons
        .iter()
        .find(|button| button.text == target && button.clickable)
        .map(|button| Action::Click(button.index))
        .unwrap_or(Action::Wait)
}
