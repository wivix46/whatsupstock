# Whatsupstock

A simple Streamlit app for relative stock comparison by sector.

## Main features
- Home page with Top 10 across all covered sectors
- Sector overview
- Up to 10 liquid stocks per sector
- Price, Market Cap, P/E, Forward P/E, EPS, Dividend Yield
- Analyst Rating and Analyst Upside
- Internal Rating (0–100)
- Analyst price-target detail for selected stocks

## Internal Rating
- P/E: 15%
- Forward P/E: 30%
- Analyst Upside: 30%
- Analyst Rating: 15%
- Dividend Yield: 10%

P/E and Forward P/E values less than or equal to zero are treated as unavailable for scoring.

## Run locally
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository, for example `whatsupstock`.
2. Upload `app.py` and `requirements.txt` to the repository root.
3. Sign in to Streamlit Community Cloud with GitHub.
4. Create a new app.
5. Select your repository and branch.
6. Set the main file path to `app.py`.
7. Optionally choose the subdomain `whatsupstock` if available.
8. Deploy.

Data source: Yahoo Finance via yfinance. Intended for personal/research use.
