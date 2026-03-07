import json
import os
import re

# File paths
README_FILE = "README.md"
METRICS_FILE = "analytics/metrics.json"

def load_metrics():
    """Load metrics from file"""
    if not os.path.exists(METRICS_FILE):
        return None

    try:
        with open(METRICS_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading metrics: {e}")
        return None

def update_readme_with_metrics():
    """Update README with current metrics"""
    metrics = load_metrics()
    if not metrics:
        print("No metrics found, skipping README update")
        return

    # Read current README
    with open(README_FILE, "r") as f:
        readme_content = f.read()

    # Update GitHub stars and forks
    if metrics.get("github"):
        stars = metrics["github"].get("stars", 1)
        forks = metrics["github"].get("forks", 0)

        # Update GitHub badges
        readme_content = re.sub(
            r'\[!\[GitHub Stars\].*?\]',
            f'[![GitHub Stars](https://img.shields.io/github/stars/ballales1984-wq/ai-trading-system.svg)](https://github.com/ballales1984-wq/ai-trading-system)',
            readme_content
        )
        readme_content = re.sub(
            r'\[!\[Forks\].*?\]',
            f'[![Forks](https://img.shields.io/github/forks/ballales1984-wq/ai-trading-system.svg)](https://github.com/ballales1984-wq/ai-trading-system/network/members)',
            readme_content
        )

    # Update performance metrics
    if metrics.get("performance"):
        cagr = metrics["performance"].get("cagr", 23.5)
        max_drawdown = metrics["performance"].get("max_drawdown", 7.2)
        sharpe_ratio = metrics["performance"].get("sharpe_ratio", 1.95)
        win_rate = metrics["performance"].get("win_rate", 68)

        # Update performance metrics table
        readme_content = re.sub(
            r'\| CAGR\s*\|\s*[\d.]+%\s*\|\s*[\d.]+%\s*\|',
            f"| CAGR | {cagr}% | 18.2% |",
            readme_content
        )
        readme_content = re.sub(
            r'\| Max Drawdown\s*\|\s*[\d.]+%\s*\|\s*[\d.]+%\s*\|',
            f"| Max Drawdown | {max_drawdown}% | 45.8% |",
            readme_content
        )
        readme_content = re.sub(
            r'\| Sharpe Ratio\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|',
            f"| Sharpe Ratio | {sharpe_ratio} | 0.82 |",
            readme_content
        )
        readme_content = re.sub(
            r'\| Win Rate\s*\|\s*[\d.]+%\s*\|\s*-\s*\|',
            f"| Win Rate | {win_rate}% | - |",
            readme_content
        )

    # Write updated README
    with open(README_FILE, "w") as f:
        f.write(readme_content)

    print("README updated with current metrics")

if __name__ == "__main__":
    update_readme_with_metrics()