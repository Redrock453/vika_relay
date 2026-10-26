"""Telegram front end: text and voice messages from allowlisted users."""
import asyncio
import logging
import subprocess
import tempfile
from pathlib import Path

from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import Command

from gate.agent import Agent
from gate.config import config

logger = logging.getLogger("gate.telegram")

TELEGRAM_LIMIT = 4096


def _chunks(text: str, size: int = TELEGRAM_LIMIT):
    """Split a long answer into Telegram-sized messages."""
    text = text or "…"
    return [text[i:i + size] for i in range(0, len(text), size)]


def create_bot() -> tuple[Bot, Dispatcher]:
    bot = Bot(token=config.telegram_token)
    dp = Dispatcher()
    agent = Agent()
    allowed = {str(i) for i in config.allowed_ids}

    async def reply(message: types.Message, text: str):
        for part in _chunks(text):
            await message.answer(part)

    @dp.message(Command("start"))
    async def cmd_start(message: types.Message):
        await message.answer("Привіт! Я Віка 💙 Готова допомогти.")

    @dp.message(Command("new"))
    async def cmd_new(message: types.Message):
        uid = str(message.from_user.id)
        if uid not in allowed:
            return
        agent.new_chat(uid)
        await message.answer("🧠 Контекст скинуто. Починаємо з чистого аркуша.")

    @dp.message(F.voice | F.audio)
    async def handle_audio(message: types.Message):
        uid = str(message.from_user.id)
        if uid not in allowed:
            return
        audio = message.voice or message.audio
        file = await bot.get_file(audio.file_id)
        with tempfile.TemporaryDirectory() as tmp:
            src, mp3 = Path(tmp) / "in.ogg", Path(tmp) / "in.mp3"
            await bot.download_file(file.file_path, str(src))
            await asyncio.to_thread(
                subprocess.run, ["ffmpeg", "-i", str(src), "-acodec", "libmp3lame", "-y", str(mp3)],
                capture_output=True, timeout=60)
            text = await asyncio.to_thread(agent.transcribe, str(mp3)) if mp3.exists() else None
        if not text:
            await message.answer("❌ Не вдалося розпізнати аудіо.")
            return
        await message.answer(f"🎤 {text}")
        await reply(message, await asyncio.to_thread(agent.ask, text, uid))

    @dp.message(F.text)
    async def handle_text(message: types.Message):
        uid = str(message.from_user.id)
        if uid not in allowed:
            return
        # The router blocks while providers answer; run it in a worker thread so the bot
        # keeps serving other chats (the router keeps per-turn state in thread-locals).
        await reply(message, await asyncio.to_thread(agent.ask, message.text, uid))

    return bot, dp
