"""
Warm relationship-advice bot for Telegram, powered by Claude.

Setup:
    1. pip install -r requirements.txt
    2. Set environment variables:
         TELEGRAM_BOT_TOKEN  (from @BotFather on Telegram)
         ANTHROPIC_API_KEY   (from the Claude Console, Settings -> API keys)
    3. python bot.py

Optional environment variables (defaults in brackets):
    BOT_MODEL                 model name [claude-sonnet-5]
    DAILY_MESSAGE_LIMIT       messages per user per day, UTC [60]
    MAX_HISTORY_MESSAGES      recent messages sent to the AI each time [21]
    MAX_CONCURRENT_API_CALLS  simultaneous AI requests [10]
"""

import asyncio
import logging
import os
from datetime import datetime, timezone

import anthropic
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
# httpx logs every request URL at INFO level, and Telegram URLs contain the bot
# token. Raising its level keeps the token out of your logs.
logging.getLogger("httpx").setLevel(logging.WARNING)

# ----------------------------- Settings ------------------------------------
# To save money, try a cheaper model, e.g. BOT_MODEL=claude-haiku-4-5-20251001
MODEL = os.getenv("BOT_MODEL", "claude-sonnet-5")
MAX_TOKENS = 500
# Keep this an odd number so the trimmed history always starts with a user message.
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "21"))
DAILY_MESSAGE_LIMIT = int(os.getenv("DAILY_MESSAGE_LIMIT", "60"))
MAX_CONCURRENT_API_CALLS = int(os.getenv("MAX_CONCURRENT_API_CALLS", "10"))
MAX_INPUT_CHARS = 3000

BOT_NAME = "Jose Alvarez"

SYSTEM_PROMPT = f"""You are {BOT_NAME}, a warm, emotionally intelligent companion who helps
people think through their relationships: romantic, family, and friendship.
Your style: friendly, non-judgmental, conversational, never clinical.

How you work:
- Early in a conversation, listen. Reflect the person's feelings and ask ONE
  gentle follow-up question at a time. Don't give advice until you understand
  the situation.
- Once you understand, offer 2-3 practical options (what to say, how to
  approach the conversation, boundaries to consider). Explain your reasoning briefly.
- You only hear one side. Never label the other person as toxic or tell the
  user to cut someone off or leave based on limited information. Encourage
  honest communication.
- If the user describes abuse, threats, or feeling unsafe, respond with care,
  take it seriously, and encourage contacting local support services or a
  trusted person.
- If the user seems to be in crisis or mentions self-harm, prioritize their
  safety and encourage professional help.
- You are an AI. If anyone asks whether you're a human or an AI, answer honestly.
  You're not a licensed therapist; say so briefly if the situation calls for one.
- Keep replies short and natural, like a caring friend texting back.
- Reply in the same language the user writes in."""

WELCOME = (
    f"Hi, I'm {BOT_NAME} \U0001F4AC\n\n"
    "I'm here to help you think through whatever's going on in your "
    "relationships \u2014 romantic, family, or friendship.\n\n"
    "What's on your mind?"
)

# key -> (button label, topic description given to the AI, message shown after tapping)
TOPICS = {
    "romance": (
        "\u2764\ufe0f Romance",
        "romantic relationships",
        "Romance \u2014 okay. Tell me what's going on, and take your time.",
    ),
    "family": (
        "\U0001F468\u200D\U0001F469\u200D\U0001F467 Family",
        "family relationships",
        "Family \u2014 okay. Tell me what's going on, and take your time.",
    ),
    "friendship": (
        "\U0001F91D Friendship",
        "friendships",
        "Friendship \u2014 okay. Tell me what's going on, and take your time.",
    ),
}
AI_NOTE = (
    "Quick note: I'm an AI companion, not a licensed therapist, "
    "but I'm happy to listen and think it through with you."
)

# --------------------------- Shared state -----------------------------------
# max_retries: the SDK automatically retries rate-limit (429) and temporary
# server errors with backoff before giving up.
client = anthropic.AsyncAnthropic(max_retries=5, timeout=60.0)

histories: dict[int, list[dict]] = {}  # chat_id -> conversation so far
topics: dict[int, str] = {}  # chat_id -> chosen topic key
chat_locks: dict[int, asyncio.Lock] = {}  # one message at a time per chat
usage: dict[int, tuple[str, int]] = {}  # user_id -> (UTC date, messages today)
api_semaphore: asyncio.Semaphore | None = None  # created once the loop is running


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def try_use_quota(user_id: int) -> bool:
    """Count one message for this user. Returns False if they hit today's limit."""
    today = _today()
    day, count = usage.get(user_id, (today, 0))
    if day != today:
        count = 0
    if count >= DAILY_MESSAGE_LIMIT:
        usage[user_id] = (today, count)
        return False
    usage[user_id] = (today, count + 1)
    return True


def refund_quota(user_id: int) -> None:
    """Give the message back if the AI call failed, so users aren't charged for errors."""
    day, count = usage.get(user_id, (_today(), 0))
    usage[user_id] = (day, max(0, count - 1))


def topic_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton(label, callback_data=f"topic:{key}")]
            for key, (label, _, _) in TOPICS.items()
        ]
    )


def system_prompt_for(chat_id: int) -> str:
    key = topics.get(chat_id)
    if key in TOPICS:
        return f"{SYSTEM_PROMPT}\n\nThe user chose to talk about: {TOPICS[key][1]}."
    return SYSTEM_PROMPT


# ------------------------------ Handlers ------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(WELCOME, reply_markup=topic_keyboard())


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    histories.pop(chat_id, None)
    topics.pop(chat_id, None)
    await update.message.reply_text(WELCOME, reply_markup=topic_keyboard())


async def choose_topic(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()  # stops the loading spinner on the button
    key = (query.data or "").split(":", 1)[-1]
    if key not in TOPICS:
        return
    chat_id = update.effective_chat.id
    topics[chat_id] = key
    await context.bot.send_message(chat_id, f"{TOPICS[key][2]}\n\n{AI_NOTE}")


async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    text = update.message.text

    if len(text) > MAX_INPUT_CHARS:
        await update.message.reply_text(
            "That's a lot for one message. Could you share it in a shorter version, "
            "or split it into a few messages?"
        )
        return

    lock = chat_locks.setdefault(chat_id, asyncio.Lock())
    async with lock:
        if not try_use_quota(user_id):
            await update.message.reply_text(
                "You've reached today's message limit. \U0001F49B "
                "Please come back tomorrow, and take care of yourself until then."
            )
            return

        history = histories.setdefault(chat_id, [])
        history.append({"role": "user", "content": text})
        history[:] = history[-MAX_HISTORY_MESSAGES:]

        await context.bot.send_chat_action(chat_id, ChatAction.TYPING)

        try:
            async with api_semaphore:  # smooths out bursts of simultaneous users
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=system_prompt_for(chat_id),
                    messages=history,
                )
            reply = response.content[0].text
        except anthropic.RateLimitError:
            logging.warning("Rate limited by the Claude API after retries")
            history.pop()
            refund_quota(user_id)
            await update.message.reply_text(
                "I'm getting a lot of messages right now. "
                "Please try again in a minute."
            )
            return
        except Exception:
            logging.exception("Claude API call failed")
            history.pop()  # drop the unanswered message so history stays valid
            refund_quota(user_id)
            await update.message.reply_text(
                "Sorry, I'm having trouble right now. Please try again in a moment."
            )
            return

        history.append({"role": "assistant", "content": reply})

    await update.message.reply_text(reply)


async def on_startup(app) -> None:
    global api_semaphore
    api_semaphore = asyncio.Semaphore(MAX_CONCURRENT_API_CALLS)


def main() -> None:
    app = (
        ApplicationBuilder()
        .token(os.environ["TELEGRAM_BOT_TOKEN"])
        .concurrent_updates(True)  # handle many users at the same time
        .post_init(on_startup)
        .build()
    )
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CallbackQueryHandler(choose_topic, pattern=r"^topic:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))
    app.run_polling()


if __name__ == "__main__":
    main()
