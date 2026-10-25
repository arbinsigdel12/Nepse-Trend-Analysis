Set-Location "$HOME\nepse-ranker"
.\.venv\Scripts\Activate.ps1
python -m nepse_ranker floorsheet-collect
python -m nepse_ranker update
python -m nepse_ranker build
python -m nepse_ranker features
python -m nepse_ranker news-fetch
python -m nepse_ranker news-build
python -m nepse_ranker predict
