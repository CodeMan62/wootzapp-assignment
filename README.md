# MiniShop RL

A Rust environment that drives a MiniShop web page through headless Chrome over CDP, with a random baseline and a tabular Q-learning agent, plus a Python script that turns the JSONL logs into `rl/report.md` and charts.

## Run

```bash
./run.sh
```

Requires Linux or WSL2 with Chromium/Chrome, Rust, Python 3, and `curl`. `run.sh` starts the local site, builds the project, trains 3 seeds, runs the random baseline and popup evaluations, and regenerates the report.
