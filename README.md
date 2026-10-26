# NEPSE Ranker

Personal research tool for the Nepal Stock Exchange. It collects prices, announcements and
floorsheet data, ranks stocks, screens them for long-term risk, and shows everything in a
dashboard. It is not investment advice.

## Daily (automatic)
A Windows scheduled task named "NEPSE Daily" runs run_daily.ps1 at 4:30 PM on trading days:
floorsheet, prices, news, features, ranking, status, backup. The log is logs\daily.log.

## Every day or week
    python -m nepse_ranker dashboard      open the dashboard
    python -m nepse_ranker status         is every dataset fresh? anything marked CHECK says what to run

## Monthly
    python -m nepse_ranker train          retrain the model (about 15 minutes)
    python -m nepse_ranker company-fetch  refresh sectors and bonus/right history
    python -m nepse_ranker dividends-build
    python -m nepse_ranker longterm-build

## Each quarter
Update EPS and book value in the dashboard Watchlist tab from the new quarterly reports.

## Rebuild from scratch on a new computer
1. Install Python and Git. Copy the project folder, then: python -m venv .venv ; .\.venv\Scripts\Activate.ps1 ; pip install -r requirements.txt
2. Restore data\floorsheet, data\watchlist and data\news\raw from the backup folder.
3. git clone --depth 1 https://github.com/Aabishkar2/nepse-data.git external\nepse-data
   python -m nepse_ranker import-folder external\nepse-data\data\company-wise
4. python -m nepse_ranker fetch --start 2026-04-06      (NEPSE daily prices, about the last year only)
5. python -m nepse_ranker build ; features ; train
6. python -m nepse_ranker news-fetch ; news-build ; company-fetch ; dividends-build ; longterm-build
7. python -m nepse_ranker predict ; dashboard

## What the tests found (so you do not have to rediscover it)
- The ranker has a small, fairly steady skill at ordering stocks (best at flagging what to avoid),
  but its edge is smaller than trading costs for weekly trading.
- News about right shares, ex-date price adjustments, lock-in expiries, insider sales and project
  shutdowns is followed by weaker returns. It is used as a warning flag, not as a ranking input.
- The long-term screen checks risk and history only. Earnings and book value are typed in by hand.
- Floorsheet history cannot be downloaded later. Every missed trading day is lost.

## Known limits
- NEPSE's trading week changed on 2026-04-06 (Monday to Friday) and its daily price limit became 15% on 2026-04-20.
- Cash dividend history is read from announcement text and is incomplete for older years.
- NEPSE data comes through an unofficial library (nepsepy). If NEPSE changes its site, downloads can break.
