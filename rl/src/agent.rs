use crate::env::{MAX_BUTTONS, Observation};
use anyhow::{Context, Result};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

pub const WAIT_ACTION: usize = MAX_BUTTONS;

pub enum Agent {
    Random(RandomAgent),
    Learning(LearningAgent),
}

impl Agent {
    pub fn random(seed: u64) -> Self {
        Agent::Random(RandomAgent {
            rng: StdRng::seed_from_u64(seed),
        })
    }

    pub fn learning(seed: u64, alpha: f64, gamma: f64, epsilon: f64) -> Self {
        Agent::Learning(LearningAgent {
            q: HashMap::new(),
            rng: StdRng::seed_from_u64(seed),
            alpha,
            gamma,
            epsilon,
        })
    }

    pub fn load(path: &str, seed: u64) -> Result<Self> {
        let text = fs::read_to_string(path).with_context(|| format!("read {path}"))?;
        let q = serde_json::from_str(&text).with_context(|| format!("parse {path}"))?;
        Ok(Agent::Learning(LearningAgent {
            q,
            rng: StdRng::seed_from_u64(seed),
            alpha: 0.0,
            gamma: 0.0,
            epsilon: 0.0,
        }))
    }

    pub fn choose(&mut self, observation: &Observation, training: bool) -> usize {
        match self {
            Agent::Random(agent) => agent.choose(observation),
            Agent::Learning(agent) => agent.choose(observation, training),
        }
    }

    pub fn update(
        &mut self,
        observation: &Observation,
        action: usize,
        reward: f64,
        next: &Observation,
        done: bool,
    ) {
        if let Agent::Learning(agent) = self {
            agent.update(observation, action, reward, next, done);
        }
    }

    pub fn set_epsilon(&mut self, epsilon: f64) {
        if let Agent::Learning(agent) = self {
            agent.epsilon = epsilon;
        }
    }

    pub fn save(&self, path: &str) -> Result<()> {
        match self {
            Agent::Random(_) => Ok(()),
            Agent::Learning(agent) => agent.save(path),
        }
    }
}

pub struct RandomAgent {
    rng: StdRng,
}

impl RandomAgent {
    fn choose(&mut self, observation: &Observation) -> usize {
        valid_actions(observation)
            .choose(&mut self.rng)
            .copied()
            .unwrap_or(WAIT_ACTION)
    }
}

pub struct LearningAgent {
    q: HashMap<String, f64>,
    rng: StdRng,
    alpha: f64,
    gamma: f64,
    epsilon: f64,
}

impl LearningAgent {
    fn choose(&mut self, observation: &Observation, training: bool) -> usize {
        if observation
            .buttons
            .iter()
            .any(|button| !button.disabled && !button.clickable)
        {
            return WAIT_ACTION;
        }

        let actions = valid_actions(observation);
        if training && self.rng.gen_range(0.0..1.0) < self.epsilon {
            return actions
                .choose(&mut self.rng)
                .copied()
                .unwrap_or(WAIT_ACTION);
        }

        let best_value = actions
            .iter()
            .map(|action| self.value(observation, *action))
            .fold(f64::NEG_INFINITY, f64::max);
        let best_actions: Vec<usize> = actions
            .into_iter()
            .filter(|action| (self.value(observation, *action) - best_value).abs() < 1e-9)
            .collect();

        if training {
            best_actions
                .choose(&mut self.rng)
                .copied()
                .unwrap_or(WAIT_ACTION)
        } else {
            best_actions.first().copied().unwrap_or(WAIT_ACTION)
        }
    }

    fn update(
        &mut self,
        observation: &Observation,
        action: usize,
        reward: f64,
        next: &Observation,
        done: bool,
    ) {
        let key = q_key(observation, action);
        let old = self.q.get(&key).copied().unwrap_or(0.0);
        let target = if done {
            reward
        } else {
            let next_value = valid_actions(next)
                .into_iter()
                .map(|next_action| self.value(next, next_action))
                .fold(f64::NEG_INFINITY, f64::max);
            reward + self.gamma * next_value
        };
        self.q.insert(key, old + self.alpha * (target - old));
    }

    fn value(&self, observation: &Observation, action: usize) -> f64 {
        self.q
            .get(&q_key(observation, action))
            .copied()
            .unwrap_or(0.0)
    }

    fn save(&self, path: &str) -> Result<()> {
        if let Some(parent) = Path::new(path).parent() {
            fs::create_dir_all(parent).with_context(|| format!("create {parent:?}"))?;
        }
        let text = serde_json::to_string_pretty(&self.q)?;
        fs::write(path, text).with_context(|| format!("write {path}"))?;
        Ok(())
    }
}

fn valid_actions(observation: &Observation) -> Vec<usize> {
    let mut actions: Vec<usize> = observation
        .buttons
        .iter()
        .filter(|button| button.clickable && button.index < MAX_BUTTONS)
        .map(|button| button.index)
        .collect();
    actions.sort_unstable();
    actions.dedup();

    if actions.is_empty() {
        vec![WAIT_ACTION]
    } else {
        actions
    }
}

fn q_key(observation: &Observation, action: usize) -> String {
    let disabled: String = observation
        .buttons
        .iter()
        .map(|button| if button.disabled { '1' } else { '0' })
        .collect();
    let context = match observation.screen.as_str() {
        "product" => format!(
            "{}:{}",
            observation.product.as_deref().unwrap_or(""),
            observation.qty.unwrap_or(0)
        ),
        "cart" => observation
            .cart
            .iter()
            .map(|(slug, qty)| format!("{slug}:{qty}"))
            .collect::<Vec<_>>()
            .join(","),
        _ => String::new(),
    };
    format!(
        "{}|{}|{}|{}|{}|{}",
        observation.goal.item, observation.goal.qty, observation.screen, context, disabled, action
    )
}
