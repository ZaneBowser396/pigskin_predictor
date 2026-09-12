# Pigskin Predictor

> Data-driven NFL confidence-pool picks, probability modelling, calibration and weekly confidence optimisation.

[![GitHub Pages](https://img.shields.io/badge/dashboard-GitHub%20Pages-58a6ff?style=for-the-badge&logo=github)](./docs/index.html)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-building-f4b942?style=for-the-badge)](#roadmap)

The Pigskin Predictor is a statistics-first NFL tipping system built for a weekly confidence pool. It predicts every game, ranks each pick from least to most confident, and tracks whether the model actually improves on the betting market.

## The scoreboard so far

| Strategy | Winner accuracy | Confidence points |
| --- | ---: | ---: |
| Market favourite | 66.49% | 72.15% |
| Walk-forward model | 66.42% | 71.98% |
| Market + ELO blend | ~66.1–66.5% | ~72.5–72.7% |

The early result is useful: a more complicated model has not beaten the market consistently yet. The next phase is identifying which feature groups add real out-of-sample signal.

## Explore the dashboard

The project includes a responsive, scrollable dashboard in [`docs/`](./docs/). It presents:

- headline backtest results;
- market-versus-model charts;
- the experiment history;
- the weekly confidence board; and
- a plain-English explanation of the modelling process.

To preview it locally, open `docs/index.html` in a browser. To publish it for free, follow [Publishing with GitHub Pages](#publishing-with-github-pages).

## Repository structure

```text
fourth-and-data/
├── docs/                       # Public GitHub Pages dashboard
│   ├── data/dashboard-data.js  # Values displayed by the site
│   ├── app.js
│   ├── index.html
│   └── styles.css
├── scripts/
│   └── update_dashboard.py     # Updates the dashboard from CSV files
├── outputs/                    # Model-generated CSV files go here
│   ├── model_summary.csv
│   └── weekly_picks.csv
└── README.md
```

## Updating the dashboard

The included example CSV files show the expected shape. Replace their rows with model output, then run:

```bash
python scripts/update_dashboard.py
```

That rewrites `docs/data/dashboard-data.js`. Commit and push the changed file and GitHub Pages will display the new results.

## Publishing with GitHub Pages

1. Push these files to the `main` branch.
2. Open the repository on GitHub.
3. Go to **Settings → Pages**.
4. Under **Build and deployment**, choose **Deploy from a branch**.
5. Select branch **main**, folder **/docs**, then click **Save**.

GitHub will display the public URL after the first deployment finishes.

## Roadmap

- [x] Market-favourite baseline
- [x] ELO ratings and market/ELO blending
- [x] Walk-forward validation
- [ ] Feature-group ablation testing
- [ ] Automated weekly game ingestion
- [ ] Probability calibration
- [ ] Confidence-rank optimisation
- [ ] Weekly picks published automatically

## Important note

This is a sports analytics and learning project, not betting advice.

