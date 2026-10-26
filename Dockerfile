FROM python:3.12-slim

WORKDIR /app

# ffmpeg converts Telegram voice messages for transcription
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gate/ gate/
COPY bot/ bot/
COPY prompts/ prompts/
COPY run.py .

CMD ["python", "run.py"]
