# Restaurant Ratings & Satisfaction Dashboard

A Streamlit dashboard built with **Altair** to analyze what influences restaurant ratings and customer satisfaction.

## Included files

- `app.py` — Streamlit application
- `restaurant_kpi_dashboard_dataset.csv` — dataset used by the app
- `requirements.txt` — Python dependencies
- `.streamlit/config.toml` — Streamlit theme and server settings
- `.gitignore` — common Python and Streamlit ignores

## Dashboard views

- Overview
- Descriptive analytics
- Diagnostic analytics
- Predictive later
- Prescriptive later

## KPIs covered

- Highest-rated cuisines
- Price range influence on ratings
- Review volume vs overall rating
- Rating differences across cities
- Impact of food, service, ambience, and value on overall rating

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## GitHub deployment

1. Create a new GitHub repository.
2. Upload all files in this folder to the repository root.
3. Make sure `app.py` and `restaurant_kpi_dashboard_dataset.csv` are both in the root.
4. Commit and push to GitHub.

## Streamlit Community Cloud deployment

1. Sign in to Streamlit Community Cloud.
2. Connect your GitHub account.
3. Click **Create app**.
4. Select your repository, branch, and set the main file path to `app.py`.
5. Deploy.

## Recommended repo structure

```text
.
├── .streamlit/
│   └── config.toml
├── app.py
├── requirements.txt
├── restaurant_kpi_dashboard_dataset.csv
└── README.md
```
