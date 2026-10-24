# Decisions

## State

The state is the goal (`item` and `quantity`), the current screen, the visible product and
quantity, the cart contents, and a signature of which buttons are clickable or disabled.
The state deliberately excludes timestamps, popup state, and raw delays so the policy
learns the task rather than a particular timing pattern.

## Actions

The action space is the visible buttons plus `wait`. Button indices are the DOM order used
by the page, and the environment ignores clicks on disabled or hidden buttons. The learning
agent waits whenever a non-disabled button is temporarily hidden by a delay, which prevents
it from clicking an unintended fallback action.

## Reward

Reward is potential-based shaping plus a small step cost:

- `-0.01` per step.
- `+0.2` when the correct product is on the product screen.
- `+0.5` when the cart contains the correct item and quantity.
- `+0.1` for adding the wrong item to the cart.
- `+0.5` for reaching the done state.
- `+1.0` for a successful checkout and `-0.5` for a failed checkout.

The discount factor is `0.95`. Shaping rewards are small relative to the terminal reward,
so the optimal policy remains "complete the task".

## Episode termination

An episode ends successfully when checkout completes with exactly the goal item and
quantity. It ends unsuccessfully on a wrong checkout or after `MAX_STEPS = 20` steps.
The environment also treats an empty-cart checkout as a failure.

## Learning

The learning agent is tabular Q-learning with `alpha = 0.2`, `gamma = 0.95`, and
epsilon-greedy exploration decaying linearly from `1.0` to `0.05` over training. Training
covers all 12 goals (4 items x 3 quantities) for 20 passes per goal, repeated with seeds
1, 2, and 3. Evaluation uses a greedy policy.

## Logging and evaluation

Every step is written as JSONL with the observation, action, reward, done flag, popup flag,
elapsed milliseconds, episode, seed, and goal. Metrics are written separately per episode.
`report.py` reads only these logs and computes success rates, Wilson intervals, the learning
curve, failure categories, and step-time percentiles.

## Skipped

I skipped the optional Part 5 extra (parallel environments or Harbor packaging). I also
skipped neural networks, visual features, and multi-agent training because the tabular
agent already solves the task within the time budget.

## Next

With more time I would run parallel environments in one Chrome process, add recovery
policies for wrong-cart states, test more delay distributions, and increase evaluation
sample sizes to tighten the confidence intervals.
