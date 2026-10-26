"""Entrypoint: start the Telegram bot."""
import asyncio
import logging

from bot.telegram import create_bot
from gate.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("gate")


async def main():
    if not config.telegram_token:
        logger.error("TELEGRAM_BOT_TOKEN is not set")
        return
    if not config.allowed_ids:
        logger.error("ALLOWED_IDS is not set")
        return
    bot, dp = create_bot()
    logger.info("vika_relay starting")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
