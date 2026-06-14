# 📈 Nifty 100 Swing-Trading Opportunity System

A **signal-only**, personal-use research system for swing trading the **Nifty 100**
(NSE, India). It scans all 100 stocks after market close, scores each one with a
transparent multi-factor model, and sends **BUY** and **EXIT** alerts to your
**Telegram** with a full plain-English rationale. A password-protected
**Streamlit** dashboard lets you review everything.

> **It never places orders.** You receive alerts and execute manually on Upstox
> (or any broker). See the [Legal note](#-legal-note).

Holding horizon: **1 day to ~1 month**. Every rule is **percentage-based and
capital-agnostic** — there are no hardcoded rupee amounts; everything lives in
[`config/settings.yaml`](config/settings.yaml).

---

## How it works (two-stage decision)

```
For each Nifty-100 stock, end-of-day:

  STAGE 1 — HARD GATES  (ALL must pass, else no signal)
    • Liquidity      avg daily turnover ≥ ₹25 cr
    • Trend          price > 200-DMA  (or sector outperforming)
    • Event          not within N days of results, not in F&O ban
    • Market regime  India VIX < ceiling AND Nifty 100 > its 200-DMA

  STAGE 2 — WEIGHTED COMPOSITE SCORE (0–100)
    Technical 35% · Sector 15% · FII/DII 10% · News sentiment 10%
    · Fundamentals 10% · Event 8% · Macro 7% · VIX/regime 5%
    (conflicting signals — e.g. strong chart + bad news — get a penalty)

  → BUY if all gates pass AND composite ≥ 65 (default), with:
       entry (breakout/close), ATR stop-loss, target (2R or pattern move),
       and a % position-size suggestion.

  → EXIT for open positions on: target / stop / trailing-stop / time-stop
       / signal-decay / trend-reversal / sector-rollover.
```

Every signal, sub-score, gate result and reason is stored in SQLite for the
dashboard and for audit.

---

## Project layout

```
nifty100_swing/
  config/        settings.yaml (all knobs) · nifty100.csv (universe) · schema/loader
  common/        shared types, NSE calendar, paths, logging
  data_ingestion/ base interfaces + adapters (prices, fundamentals, fii_dii, vix,
                  macro, news, events, sectors) + bhavcopy validators
  analyzers/     indicators, technical, patterns(scipy), sentiment(FinBERT),
                  fundamental, sector/fii_dii/vix/macro/event factors
  scoring/       gates → composite → signal → exits
  backtest/      costs (Indian round-trip) · engine · walk-forward · metrics
  alerting/      Telegram formatter + sender
  storage/       SQLAlchemy models + DB helpers (SQLite)
  dashboard/     Streamlit app (auth, components)
  scheduler/     run_eod.py  ← GitHub Actions entry point
  scripts/       seed_data · build_universe · run_backtest
  tests/         unit + integration tests
  .github/workflows/eod.yml
```

---

## Quick start (local)

```bash
cd nifty100_swing
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or the lean core set, see below

# 1) Configure
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then edit it
#   (optional) refresh the universe with live ISINs:
python scripts/build_universe.py

# 2) Warm the data caches (optional but speeds the first run)
python scripts/seed_data.py --limit 25

# 3) Run the end-of-day pipeline once (writes SQLite, sends Telegram)
python -m scheduler.run_eod --limit 25            # test on 25 stocks
python -m scheduler.run_eod                        # full universe
python -m scheduler.run_eod --no-send              # don't push Telegram

# 4) Open the dashboard
streamlit run dashboard/app.py
```

**Lean install** (just the scoring engine + tests, no ML/dashboard libs):

```bash
pip install pandas numpy scipy PyYAML pydantic python-dateutil requests SQLAlchemy pytest
pytest -q
```

Heavy/optional libraries (FinBERT `transformers`+`torch`, `vectorbt`,
`quantstats`, `streamlit`) **degrade gracefully** — e.g. sentiment falls back to
a finance lexicon if FinBERT isn't installed, and the backtest reports metrics
even without quantstats.

---

## Configuration

Open [`config/settings.yaml`](config/settings.yaml) — it is heavily commented.
Highlights you'll likely tune:

| Setting | Meaning | Default |
|---|---|---|
| `scoring.entry_threshold` | composite needed to BUY | 65 |
| `scoring.exit_threshold` | composite below which to EXIT | 45 |
| `scoring.weights.*` | factor weights (auto-normalised) | tech 35% … |
| `gates.market_regime.vix_ceiling` | block new entries above this VIX | 22 |
| `risk.atr_stop_multiple` | stop = entry − k×ATR | 2.0 |
| `risk.rr_target_multiple` | target = entry + R×risk | 2.0 |
| `risk.max_holding_days` | time-based exit (~1 month) | 22 |
| `risk.risk_per_trade_pct` / `max_position_pct` | sizing | 1% / 10% |
| `costs.*` | Indian round-trip costs for the backtest | delivery |

The universe is [`config/nifty100.csv`](config/nifty100.csv)
(`symbol, ISIN, upstox_key, sector, name`). The yfinance path needs only the
symbol; `upstox_key` is auto-derived as `NSE_EQ|<ISIN>`.

---

## Data sources (and how to swap them)

Every source sits behind an interface in
[`data_ingestion/base.py`](data_ingestion/base.py):

| Data | Default | Alternatives |
|---|---|---|
| Daily OHLCV | yfinance `<SYM>.NS` | Upstox V3 / Dhan (set `data.primary_price_source`) |
| Validation | NSE bhavcopy (jugaad-data/nsepython) | — |
| Fundamentals | screener.in (cached daily) | Tickertape |
| FII/DII | nsepython | nsefin |
| India VIX | `^INDIAVIX` (yfinance) | nsepython |
| Macro | `INR=X`, `CL=F`, `^GSPC`, `^IXIC` | — (RBI repo in config) |
| News | RSS (ET, Moneycontrol, BS, Mint, Pulse) | Marketaux/NewsData.io |
| Sentiment | FinBERT (`ProsusAI/finbert`) | lexicon fallback / LLM API |
| Sectors | NSE sectoral indices | — |

**Bhavcopy validation:** prices are cross-checked against the official NSE
bhavcopy; bars that are flat (O=H=L=C) or deviate materially are flagged and
repaired (`data.validate_with_bhavcopy: true`).

**Marketaux news (optional, better than RSS):** Marketaux returns headlines
**already tagged to stock tickers** plus a built-in sentiment score, so it skips
the fuzzy headline→company matching and can feed sentiment directly. To enable:
1. Get a free key at <https://www.marketaux.com> (free tier ≈ 100 requests/day).
2. Add `MARKETAUX_KEY` to your secrets / env (and GitHub repo Secrets for the job).
3. Set `news.provider: marketaux` in `settings.yaml`.
The system falls back to free RSS automatically if the key is missing. Tune the
free-tier batching under `news.marketaux` (e.g. `max_requests`).

---

## Telegram setup

1. In Telegram, message **@BotFather** → `/newbot` → copy the **bot token**.
2. Send any message to your new bot.
3. Get your **chat id**: open
   `https://api.telegram.org/bot<TOKEN>/getUpdates` and read `chat.id`
   (or message **@userinfobot**).
4. Put `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in your secrets (local
   `.streamlit/secrets.toml`, GitHub Actions secrets, and/or Streamlit Cloud).

Without these, the pipeline runs in **dry-run** mode (alerts are logged, not sent).

---

## Deploy the dashboard (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. On <https://share.streamlit.io>, create an app from the repo with main file
   `dashboard/app.py`.
3. In the app's **Advanced settings → Secrets**, paste your `app_password` and
   Telegram keys (same format as `.streamlit/secrets.toml.example`).
4. The dashboard **reads** the SQLite DB that the scheduled job **writes** and
   commits back, so it always reflects the latest run.

> Streamlit Community Cloud can't reliably run cron — that's why scheduling lives
> in GitHub Actions.

---

## Schedule the daily run (GitHub Actions)

[`.github/workflows/eod.yml`](.github/workflows/eod.yml) runs
**weekdays at 10:30 UTC** (after the 15:30 IST close; weekday NSE holidays are
skipped automatically). It checks out, installs a lean dependency set, runs
`python -m scheduler.run_eod`, then commits the updated `data/swing.db` back so
state persists.

Add these **repository Secrets** (Settings → Secrets and variables → Actions):
`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` (and `UPSTOX_*` / `DHAN_*` only if you
switch the price source). You can also trigger it manually via **Run workflow**.

### Upstox token caveat
Upstox access tokens **expire daily at 03:30 IST** regardless of when generated.
For the unattended GitHub Actions run, prefer **yfinance/NSE** (the default).
Only use the Upstox adapter if you have a daily token-refresh routine.

---

## Backtesting

```bash
# Self-contained demo (synthetic data, no network) + QuantStats tearsheet:
python scripts/run_backtest.py --synthetic --tearsheet

# On real data (install yfinance; seed first for speed):
python scripts/run_backtest.py --limit 40 --start 2019-01-01 --tearsheet
python scripts/run_backtest.py --limit 40 --walkforward      # out-of-sample
```

Outputs land in `backtest/output/` (`tearsheet.html`, `trades.csv`, `equity.csv`)
plus a plain-English metrics summary (CAGR, Sharpe, Sortino, max drawdown, win
rate, profit factor, avg win/loss, exposure, trade count).

**Honesty note:** the backtest replays the **technical engine** (the part that is
computable point-in-time from price/volume: 200-DMA trend, N-day breakout with
volume confirmation, RSI, ATR stops/targets, trailing/time exits). Fundamentals,
news sentiment and FII/DII are **not** replayed — reliable point-in-time history
for them isn't freely available, and faking it would inflate results. Execution
decides on the close and fills the next open (no look-ahead), accounts for NSE
holidays, and charges the full Indian round-trip cost stack.

---

## Testing

```bash
pytest -q
```

Covers indicators, costs, the NSE calendar, config validation, news matching,
gates, the composite + conflict penalty, trade-plan sizing, exits, and a full
end-to-end pipeline run with a fake data provider.

---

## ⚖️ Legal note

This is a **personal-use, signal-only** system. It **places no orders** and is
not sold or redistributed, so it falls **outside SEBI's retail algo-trading
framework** (SEBI circular dated 4 Feb 2025, mandatory from 1 Apr 2026 — which is
triggered by automated **API order placement** above the 10-orders-per-second
threshold). It also does **not** require **Research Analyst** registration (that
applies to *selling* signals or black-box algos to others). **Keep it
personal-use only.**

---

## ⚠️ Disclaimer

For personal research and education only. **Not investment advice.** Markets are
risky; backtests and past performance do not guarantee future results. You are
solely responsible for any trades you place. Verify the NSE holiday list in
[`common/calendar_nse.py`](common/calendar_nse.py) and the RBI repo rate in
settings each year/MPC.
```
