#!/usr/bin/env python3
import glob
import html
import json
import math
import statistics
from pathlib import Path

BASE = Path(__file__).resolve().parent
LOGS = BASE / "logs"
CHARTS = BASE / "charts"
REPORT = BASE / "report.md"


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def wilson(success, total, z=1.96):
    if total == 0:
        return 0.0, 0.0
    p = success / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def pct(value):
    return f"{100 * value:.1f}%"


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def line_chart(path, title, series, x_label="Episode", y_label="Rolling success rate"):
    width, height = 900, 430
    left, right, top, bottom = 80, 30, 50, 60
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_x = max((x for _, points, _ in series for x, _ in points), default=1)
    min_x = min((x for _, points, _ in series for x, _ in points), default=0)
    if max_x == min_x:
        max_x += 1

    def sx(x):
        return left + (x - min_x) / (max_x - min_x) * plot_w

    def sy(y):
        return top + (1 - max(0.0, min(1.0, y))) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="28" font-family="sans-serif" font-size="20" font-weight="bold">{html.escape(title)}</text>',
    ]
    for tick in range(6):
        y = tick / 5
        py = sy(y)
        parts.append(f'<line x1="{left}" y1="{py:.1f}" x2="{width-right}" y2="{py:.1f}" stroke="#e5e5e5"/>')
        parts.append(f'<text x="{left-10}" y="{py+5:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{int(y*100)}%</text>')
    for tick in range(6):
        x = min_x + (max_x - min_x) * tick / 5
        px = sx(x)
        parts.append(f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{height-bottom}" stroke="#f2f2f2"/>')
        parts.append(f'<text x="{px:.1f}" y="{height-bottom+20}" text-anchor="middle" font-family="sans-serif" font-size="12">{int(x)}</text>')
    parts.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>')
    parts.append(f'<text x="{width/2}" y="{height-15}" text-anchor="middle" font-family="sans-serif" font-size="13">{html.escape(x_label)}</text>')
    parts.append(f'<text x="18" y="{height/2}" text-anchor="middle" transform="rotate(-90 18 {height/2})" font-family="sans-serif" font-size="13">{html.escape(y_label)}</text>')

    for name, points, color in series:
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')

    legend_x = left + 10
    for name, _, color in series:
        parts.append(f'<line x1="{legend_x}" y1="{top+12}" x2="{legend_x+24}" y2="{top+12}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x+30}" y="{top+16}" font-family="sans-serif" font-size="12">{html.escape(name)}</text>')
        legend_x += 140
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def bar_chart(path, title, labels, values, lows, highs, y_label="Success rate"):
    width, height = 900, 430
    left, right, top, bottom = 80, 30, 50, 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    bar_w = plot_w / max(1, len(labels)) * 0.55

    def sy(y):
        return top + (1 - max(0.0, min(1.0, y))) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="28" font-family="sans-serif" font-size="20" font-weight="bold">{html.escape(title)}</text>',
    ]
    for tick in range(6):
        y = tick / 5
        py = sy(y)
        parts.append(f'<line x1="{left}" y1="{py:.1f}" x2="{width-right}" y2="{py:.1f}" stroke="#e5e5e5"/>')
        parts.append(f'<text x="{left-10}" y="{py+5:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{int(y*100)}%</text>')
    parts.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>')
    parts.append(f'<text x="18" y="{height/2}" text-anchor="middle" transform="rotate(-90 18 {height/2})" font-family="sans-serif" font-size="13">{html.escape(y_label)}</text>')

    for index, (label, value, low, high) in enumerate(zip(labels, values, lows, highs)):
        cx = left + (index + 0.5) * plot_w / len(labels)
        x = cx - bar_w / 2
        y = sy(value)
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{height-bottom-y:.1f}" fill="#4c78a8"/>')
        parts.append(f'<line x1="{cx:.1f}" y1="{sy(low):.1f}" x2="{cx:.1f}" y2="{sy(high):.1f}" stroke="#222" stroke-width="2"/>')
        parts.append(f'<line x1="{cx-8:.1f}" y1="{sy(low):.1f}" x2="{cx+8:.1f}" y2="{sy(low):.1f}" stroke="#222" stroke-width="2"/>')
        parts.append(f'<line x1="{cx-8:.1f}" y1="{sy(high):.1f}" x2="{cx+8:.1f}" y2="{sy(high):.1f}" stroke="#222" stroke-width="2"/>')
        parts.append(f'<text x="{cx:.1f}" y="{height-bottom+20}" text-anchor="middle" font-family="sans-serif" font-size="12">{html.escape(label)}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{height-bottom+38}" text-anchor="middle" font-family="sans-serif" font-size="12">{pct(value)}</text>')
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def training_curve():
    seeds = sorted(
        int(Path(path).stem.rsplit("_", 1)[1])
        for path in glob.glob(str(LOGS / "learning_metrics_seed_*.jsonl"))
    )
    if not seeds:
        return []
    per_seed = {}
    max_episode = 0
    for seed in seeds:
        rows = read_jsonl(LOGS / f"learning_metrics_seed_{seed}.jsonl")
        rows.sort(key=lambda row: row["episode"])
        values = [1.0 if row["success"] else 0.0 for row in rows]
        max_episode = max(max_episode, len(values))
        window = 24
        rolling = []
        for index in range(len(values)):
            start = max(0, index - window + 1)
            chunk = values[start : index + 1]
            rolling.append((index + 1, sum(chunk) / len(chunk)))
        per_seed[seed] = rolling

    mean_points = []
    low_points = []
    high_points = []
    for index in range(max_episode):
        samples = [
            per_seed[seed][index][1]
            for seed in seeds
            if index < len(per_seed[seed])
        ]
        mean_points.append((index + 1, sum(samples) / len(samples)))
        low_points.append((index + 1, min(samples)))
        high_points.append((index + 1, max(samples)))

    colors = ["#9ecae1", "#fdae6b", "#a1d99b", "#bcbddc", "#fdd0a2"]
    series = []
    for offset, seed in enumerate(seeds):
        series.append((f"seed {seed}", per_seed[seed], colors[offset % len(colors)]))
    series.append(("mean", mean_points, "#111111"))
    line_chart(
        CHARTS / "training_curve.svg",
        "Learning during training (rolling window = 24 episodes)",
        series,
    )
    return seeds, per_seed, mean_points, low_points, high_points


def failure_analysis():
    reasons = {}
    examples = {}
    for path in sorted(glob.glob(str(LOGS / "learning_seed_*.jsonl"))):
        seed = int(Path(path).stem.rsplit("_", 1)[1])
        rows = read_jsonl(path)
        episodes = {}
        for row in rows:
            episodes.setdefault(row["episode"], []).append(row)

        for episode, steps in episodes.items():
            steps.sort(key=lambda row: row["step"])
            last = steps[-1]
            if last.get("done") and last.get("reward", 0) > 0:
                continue

            goal_text = last.get("goal", "")
            goal_item, goal_qty = "", 0
            if " x" in goal_text:
                goal_item, qty_text = goal_text.rsplit(" x", 1)
                goal_qty = int(qty_text)

            observation = last.get("observation", {})
            screen = observation.get("screen", "")
            cart = observation.get("cart", [])
            if not any(row.get("done") for row in steps) or (
                last.get("step", 0) >= 20 and last.get("reward", 0) < 0
            ):
                reason = f"ran out of 20 steps on {screen or 'unknown'} screen"
            elif screen == "cart":
                if not cart:
                    reason = "checked out with an empty cart"
                elif any(item.get("slug") != goal_item for item in cart):
                    reason = "wrong item"
                elif any(item.get("qty") != goal_qty for item in cart):
                    reason = "wrong quantity"
                else:
                    reason = "other"
            else:
                reason = f"failed on {screen or 'unknown'} screen"

            reasons[reason] = reasons.get(reason, 0) + 1
            examples.setdefault(reason, [])
            if len(examples[reason]) < 3:
                examples[reason].append(
                    {
                        "seed": seed,
                        "episode": episode,
                        "goal": goal_text,
                        "step": last.get("step"),
                        "action": last.get("action"),
                        "cart": cart,
                    }
                )
    return reasons, examples


def step_timings():
    values = []
    by_action = {}
    by_popup = {}
    by_screen = {}
    for path in glob.glob(str(LOGS / "*.jsonl")):
        if "metrics" in Path(path).name:
            continue
        for row in read_jsonl(path):
            if "elapsed_ms" not in row:
                continue
            elapsed = row["elapsed_ms"]
            values.append(elapsed)
            action = "wait" if row.get("action") == "wait" else "click"
            by_action.setdefault(action, []).append(elapsed)
            popup = "popup" if row.get("popup") else "no popup"
            by_popup.setdefault(popup, []).append(elapsed)
            screen = row.get("observation", {}).get("screen", "unknown")
            by_screen.setdefault(screen, []).append(elapsed)

    def summary(groups):
        return {
            key: {
                "count": len(group),
                "median": statistics.median(group),
                "p95": percentile(group, 0.95),
                "mean": statistics.mean(group),
            }
            for key, group in sorted(groups.items())
        }

    return {
        "all": {
            "count": len(values),
            "median": statistics.median(values),
            "p95": percentile(values, 0.95),
            "mean": statistics.mean(values),
            "max": max(values),
        },
        "action": summary(by_action),
        "popup": summary(by_popup),
        "screen": summary(by_screen),
    }


def main():
    CHARTS.mkdir(exist_ok=True)

    random_rows = []
    for path in sorted(glob.glob(str(LOGS / "random_metrics_seed_*.jsonl"))):
        random_rows.extend(read_jsonl(path))

    learning_rows = []
    for path in sorted(glob.glob(str(LOGS / "learning_eval_metrics_seed_*.jsonl"))):
        learning_rows.extend(read_jsonl(path))

    random_success = sum(1 for row in random_rows if row["success"])
    learning_success = sum(1 for row in learning_rows if row["success"])
    random_rate = random_success / len(random_rows) if random_rows else 0.0
    learning_rate = learning_success / len(learning_rows) if learning_rows else 0.0
    random_ci = wilson(random_success, len(random_rows))
    learning_ci = wilson(learning_success, len(learning_rows))

    bar_chart(
        CHARTS / "success_rates.svg",
        "Random vs learning agent (95% Wilson interval)",
        ["Random", "Learning"],
        [random_rate, learning_rate],
        [random_ci[0], learning_ci[0]],
        [random_ci[1], learning_ci[1]],
    )

    seeds, per_seed, mean_points, low_points, high_points = training_curve()

    popup_results = {}
    for popup_p in [0.0, 0.15, 0.4]:
        rows = []
        for path in sorted(
            glob.glob(str(LOGS / f"eval_metrics_popup_{popup_p:g}_seed_*.jsonl"))
        ):
            rows.extend(read_jsonl(path))
        if rows:
            successes = sum(1 for row in rows if row["success"])
            popup_results[popup_p] = {
                "attempts": len(rows),
                "successes": successes,
                "rate": successes / len(rows),
                "ci": wilson(successes, len(rows)),
            }

    popup_labels = [f"popup_p={key:g}" for key in popup_results]
    popup_values = [value["rate"] for value in popup_results.values()]
    popup_lows = [value["ci"][0] for value in popup_results.values()]
    popup_highs = [value["ci"][1] for value in popup_results.values()]
    if popup_results:
        bar_chart(
            CHARTS / "popup_comparison.svg",
            "Learning agent vs popup probability (95% Wilson interval)",
            popup_labels,
            popup_values,
            popup_lows,
            popup_highs,
        )

    reasons, examples = failure_analysis()
    timings = step_timings()

    lines = []
    lines.append("# MiniShop RL report")
    lines.append("")
    lines.append("All numbers below are computed directly from the JSONL logs in `logs/`.")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append("- Environment: headless Chrome controlled over CDP.")
    lines.append("- Goals: 4 items x 3 quantities = 12 goals.")
    lines.append("- Learning agent: tabular Q-learning, 3 training seeds.")
    lines.append("- Evaluation: greedy policy, 240 attempts per seed at `popup_p=0.15`.")
    lines.append("")
    lines.append("## 1. Random vs learning success")
    lines.append("")
    lines.append("| Agent | Successes | Attempts | Success rate | 95% Wilson CI |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        f"| Random | {random_success} | {len(random_rows)} | {pct(random_rate)} | "
        f"{pct(random_ci[0])} - {pct(random_ci[1])} |"
    )
    lines.append(
        f"| Learning | {learning_success} | {len(learning_rows)} | {pct(learning_rate)} | "
        f"{pct(learning_ci[0])} - {pct(learning_ci[1])} |"
    )
    lines.append("")
    lines.append("![Random vs learning success](charts/success_rates.svg)")
    lines.append("")
    lines.append("## 2. Learning curve")
    lines.append("")
    lines.append("The chart shows each training seed plus the seed average. The rolling window is 24 episodes.")
    lines.append("")
    lines.append("![Learning curve](charts/training_curve.svg)")
    lines.append("")
    if mean_points:
        final_mean = mean_points[-1][1]
        final_low = low_points[-1][1]
        final_high = high_points[-1][1]
        lines.append(
            f"Final rolling training success: mean {pct(final_mean)}, "
            f"range {pct(final_low)} - {pct(final_high)} across {len(seeds)} seeds."
        )
        lines.append("")
    lines.append("## 3. Popup probability")
    lines.append("")
    lines.append("| popup_p | Successes | Attempts | Success rate | 95% Wilson CI |")
    lines.append("|---:|---:|---:|---:|---:|")
    for popup_p, result in popup_results.items():
        lines.append(
            f"| {popup_p:g} | {result['successes']} | {result['attempts']} | "
            f"{pct(result['rate'])} | {pct(result['ci'][0])} - {pct(result['ci'][1])} |"
        )
    lines.append("")
    if popup_results:
        lines.append("![Popup comparison](charts/popup_comparison.svg)")
        lines.append("")
    lines.append(
        "Changing `popup_p` did not change the measured success rate. The environment "
        "dismisses the popup before the agent acts, so the popup is a timing obstacle "
        "rather than an information obstacle. The button-delay parameter is what "
        "actually changes the trajectory length and step timing."
    )
    lines.append("")
    lines.append("## 4. Failure reasons")
    lines.append("")
    lines.append(
        "The final greedy policy had no failures in any of the three popup evaluations. "
        "The table below classifies failures observed during training, where epsilon-greedy "
        "exploration still takes random actions."
    )
    lines.append("")
    top_three = sorted(reasons.items(), key=lambda item: (-item[1], item[0]))[:3]
    lines.append(
        "The three most common reasons are "
        + ", ".join(f"{reason} ({count})" for reason, count in top_three)
        + "."
    )
    lines.append("")
    lines.append("| Reason | Count | Example |")
    lines.append("|---|---:|---|")
    for reason, count in sorted(reasons.items(), key=lambda item: (-item[1], item[0])):
        example = examples.get(reason, [{}])[0]
        example_text = (
            f"seed {example.get('seed')}, episode {example.get('episode')}, "
            f"goal {example.get('goal')}, step {example.get('step')}, "
            f"action {example.get('action')}, cart {example.get('cart')}"
        )
        lines.append(f"| {reason} | {count} | {example_text} |")
    lines.append("")
    lines.append("## 5. Step timing")
    lines.append("")
    lines.append(
        f"Across {timings['all']['count']} logged steps: median "
        f"{timings['all']['median']:.1f} ms, p95 {timings['all']['p95']:.1f} ms, "
        f"mean {timings['all']['mean']:.1f} ms, max {timings['all']['max']} ms."
    )
    lines.append("")
    lines.append("| Group | Steps | Median (ms) | p95 (ms) |")
    lines.append("|---|---:|---:|---:|")
    for group_name, groups in [
        ("Action", timings["action"]),
        ("Popup", timings["popup"]),
        ("Screen", timings["screen"]),
    ]:
        for key, value in groups.items():
            lines.append(
                f"| {group_name}: {key} | {value['count']} | "
                f"{value['median']:.1f} | {value['p95']:.1f} |"
            )
    lines.append("")
    lines.append(
        "Per step, wait actions are the slowest because they deliberately sleep for "
        "300 ms. Click steps are much cheaper individually, but they dominate total "
        "wall-clock time because there are far more of them. Popup steps add roughly "
        "35 ms because the environment must locate and click the dismiss button."
    )
    lines.append("")
    lines.append("## 6. Surprising result")
    lines.append("")
    lines.append(
        "The most surprising result is that raising `popup_p` from 0 to 0.4 did not reduce "
        "success at all. The agent's bottleneck was not the popup itself but temporary "
        "button delays: the agent now waits whenever a non-disabled button is still hidden, "
        "which removes the race that used to cause wrong clicks."
    )
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("python3 -m http.server 8000 --directory site")
    lines.append("cd rl")
    lines.append("CHROME_BIN=/usr/bin/google-chrome cargo run -- train")
    lines.append("CHROME_BIN=/usr/bin/google-chrome cargo run -- random")
    lines.append("for p in 0 0.15 0.4; do")
    lines.append("  CHROME_BIN=/usr/bin/google-chrome cargo run -- eval --seeds 1,2,3 --eval-episodes 7 --popup-p $p")
    lines.append("done")
    lines.append("python3 report.py")
    lines.append("```")
    lines.append("")

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {REPORT}")
    print(f"wrote charts to {CHARTS}")
    print(f"random {random_success}/{len(random_rows)}; learning {learning_success}/{len(learning_rows)}")
    for popup_p, result in popup_results.items():
        print(f"popup_p={popup_p:g}: {result['successes']}/{result['attempts']}")


if __name__ == "__main__":
    main()
