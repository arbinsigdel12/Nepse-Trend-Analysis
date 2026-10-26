# NEPSE Trend Analysis

A personal research toolkit for the Nepal Stock Exchange (NEPSE). It collects daily prices,
company announcements and the floorsheet (every trade), ranks stocks with an honestly tested
model, screens them for long-term risk, and shows everything in a local dashboard.

> **Not investment advice.** This is a research tool built by an individual. Read the
> "What the tests found" section before relying on any output.

## What it does

- **Collects data every trading day:** prices, official NEPSE announcements, and the floorsheet.
- **Cleans it:** adjusts prices for bonus and right shares so history is continuous.
- **Ranks tradable stocks** with a gradient-boosted model, tested only on data it had never seen
  (walk-forward testing), and compares it with simple rules.
- **Reads announcements** and turns right-share, ex-date, lock-in, insider-sale and project-shutdown
  news into warning flags.
- **Screens for long-term risk:** history, liquidity, past falls, volatility, dilution, red-flag news,
  plus dividend history.
- **Plans portfolios:** concentration limits per stock and sector, and a replay of how a mix behaved in past falls.
- **Values stocks you follow** from EPS and book value that you enter by hand.
- **Watches itself:** a `status` command shows how fresh every dataset is.

## Dashboard tabs

| Tab | Purpose |
|---|---|
| Today's ranking | The model's ordering of tradable stocks, with fresh negative-news warnings |
| Stock | Chart, recent announcements, and broker activity (floorsheet) for one stock |
| Model check | How reliable the ranking has been in past tests |
| Data health | Completeness and freshness of the data |
| How to read this | Plain explanation and limits |
| Long-term | Six-check risk and history screen with dividend history |
| Portfolio | Holdings, concentration warnings, and a replay of past falls |
| Watchlist | PE, PB, ROE and yield from numbers you enter |

## How the pieces fit
prices (NEPSE + open dataset) --> clean + bonus adjustment --> features --> model --> ranking --+
announcements --> tag event types --> warning flags ---------------------------------------+--> dashboard
company data + dividend text --> long-term screen ------------------------------------------+
floorsheet (daily) --> broker concentration measures --------------------------------------+

## Quick start (Windows, PowerShell)

The helper scripts expect the project folder to be called `nepse-ranker` in your user folder,
so clone it under that name.

```powershell
git clone https://github.com/arbinsigdel12/Nepse-Trend-Analysis.git nepse-ranker
cd nepse-ranker
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Load history and build everything once:

```powershell
git clone --depth 1 https://github.com/Aabishkar2/nepse-data.git external\nepse-data
python -m nepse_ranker import-folder external\nepse-data\data\company-wise
python -m nepse_ranker fetch --start 2026-04-06
python -m nepse_ranker build
python -m nepse_ranker features
python -m nepse_ranker train --max-folds 20
python -m nepse_ranker news-fetch
python -m nepse_ranker news-build
python -m nepse_ranker company-fetch
python -m nepse_ranker dividends-build
python -m nepse_ranker longterm-build
python -m nepse_ranker predict
python -m nepse_ranker dashboard
```

`train --max-folds 20` tests the latest 20 blocks and takes a few minutes. Leave out
`--max-folds 20` for the full history test (about 15 to 25 minutes).

## Daily, weekly, monthly

**Automatic daily job.** `run_daily.ps1` collects the floorsheet, updates prices and news, rebuilds
features, ranks, runs `status`, and backs up. Register it once (adjust the time if your PC clock
is not on Nepal time). NEPSE trades Monday to Friday.

```powershell
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$HOME\nepse-ranker\run_daily.ps1`""
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At 4:30PM
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName "NEPSE Daily" -Action $action -Trigger $trigger -Settings $settings -Force
```

The log is `logs\daily.log`. Every run ends with `RUN OK` or lists the steps that failed.

| How often | What to run |
|---|---|
| Any day | `python -m nepse_ranker dashboard` |
| Weekly | `python -m nepse_ranker status` (CHECK or MISSING lines say what to run) |
| Monthly | `train`, `company-fetch`, `dividends-build`, `longterm-build` |
| Each quarter | Update EPS and book value in the Watchlist tab |

## Rebuild on a new computer

1. Follow the quick start above (clone, virtual environment, install).
2. Restore the irreplaceable data from the backup folder (`backup.ps1` writes it to
   `OneDrive\nepse-backup` or `Documents\nepse-backup`): `data\floorsheet`, `data\watchlist`,
   `data\news\raw`, `data\company`, `data\imported`, `data\raw`.
3. Re-run the "load history" commands. Everything else is rebuilt from them.

The floorsheet cannot be downloaded again, because NEPSE only serves the latest trading day.
Every trading day that is not collected is lost for good.

## Commands

| Command | What it does |
|---|---|
| `fetch --start YYYY-MM-DD` / `update` | Download NEPSE daily prices (about the last year is available) |
| `import-folder PATH` / `import-csv FILE` | Load price history from CSV files |
| `build` / `features` | Clean and adjust prices / create features and labels |
| `train` / `predict` / `run-all` | Train and test the model / rank the latest day / do all of it |
| `news-fetch` / `news-build` / `news-study` | Download announcements / tag them / check what follows each type |
| `company-fetch` / `dividends-build` / `longterm-build` | Sectors and bonus history / cash dividends from text / long-term screen |
| `floorsheet-collect` / `floorsheet-build` | Save today's trades / broker concentration measures |
| `status` | Freshness of every dataset |
| `dashboard` | Open the dashboard in the browser |
| `inspect-raw`, `news-inspect`, `company-inspect`, `floorsheet-inspect` | Show what a downloaded dataset looks like |

## Project layout
nepse_ranker/ the Python package (one module per step)
dashboard/ Streamlit app (app.py)
tests/ tests
run_daily.ps1 the automatic daily job
backup.ps1 copies the irreplaceable data to a backup folder
data/ all downloaded and built data (not in git)
logs/ daily job log (not in git)
external/ the open price dataset clone (not in git)

## What the tests found

These are the results as of October 2026, from data back to 2019.

- **The ranker has a small, fairly steady skill at ordering stocks.** Over 2020 to 2026 its rank correlation
  with what happened next was about 0.095 (0.077 among stocks of similar volatility). It was positive
  in 6 of 7 years. Its top 10 picks beat the average stock by about 0.45% over 5 days before costs.
- **That edge is smaller than trading costs for weekly trading.** Holding winners longer to cut trading did
  not produce a result that was statistically visible at realistic costs. Treat the ranking as a
  screening aid and an avoid-list, not a trading signal.
- **Announcements carry some information.** Stocks with fresh right-share, ex-date, lock-in, merger,
  insider-sale or project-shutdown news trailed the others by about 0.6% over the next 5 days, in every
  year with data. The dividend and bonus effects were larger before 2024 and are near zero now.
  Adding the news to the model did not improve its ranking, so it is used as a warning flag instead.
- **Floorsheet patterns need time.** Collection started in October 2026. Broker features can be tested
  once a few months of days have accumulated.

## Data sources and credits

- NEPSE public website data through the unofficial [`nepsepy`](https://pypi.org/project/nepsepy/) client.
- Price history before the NEPSE daily window comes from
  [Aabishkar2/nepse-data](https://github.com/Aabishkar2/nepse-data). See that repository for its license.
- Downloaded data stays on your computer and is not part of this repository. Please respect NEPSE's
  terms of use and do not republish scraped data.

## Limits

- Not affiliated with NEPSE. Not investment advice.
- NEPSE changed its trading week on 2026-04-06 (Monday to Friday) and its daily price limit to 15% on 2026-04-20.
- Cash dividend history is read from announcement text and is incomplete for older years.
- Earnings, book value and bad-loan figures are not in the data. You enter them by hand.
- The data comes through an unofficial library. If NEPSE changes its website, downloads can break.
- No license file is included yet, so by default all rights are reserved. Add one if you want others to reuse the code.
