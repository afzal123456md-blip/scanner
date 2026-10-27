Files to upload to the repo root: bot.py, strategy.py, events.json, requirements.txt
Then create the file .github/workflows/scanner.yml and paste the content of scanner.yml (included here).
Add two secrets: TELEGRAM_TOKEN and TELEGRAM_CHAT_ID.
Run the workflow once with mode = check, then it runs by itself every ~10 minutes.
