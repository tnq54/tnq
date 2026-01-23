import streamlit as st
import os
import logging
import time
import threading
import asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from huggingface_hub import InferenceClient

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Streamlit Interface (Minimal)
st.title("VBot1 System")
st.write("System is running. Telegram Bot is active (HF Only).")

# Load Environment Variables
HF_TOKEN = os.environ.get("HF_TOKEN")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")

# Initialize AI Clients
try:
    if HF_TOKEN:
        hf_client = InferenceClient(token=HF_TOKEN)
    else:
        logging.warning("HF_TOKEN is missing.")
        hf_client = None
except Exception as e:
    logging.error(f"Error initializing clients: {e}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hello! I am VBot1 (Rebuilt).\n"
        "- Chat with me to use Llama 3."
    )

async def chat_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not hf_client:
        await update.message.reply_text("Llama 3 is not configured (missing HF_TOKEN).")
        return

    user_text = update.message.text
    status_msg = await update.message.reply_text("Thinking (Llama 3)...")

    try:
        messages = [{"role": "user", "content": user_text}]
        response = hf_client.chat_completion(
            messages=messages,
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            max_tokens=800
        )
        reply = response.choices[0].message.content
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status_msg.message_id, text=reply)
    except Exception as e:
        logging.error(f"Llama Error: {e}")
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status_msg.message_id, text="Error processing with Llama 3.")

# Bot Thread Logic
def run_bot_loop():
    if not TELEGRAM_TOKEN:
        print("TELEGRAM_TOKEN not found. Bot cannot start.")
        return

    # Sleep to prevent network errors on startup (inside the thread to not block UI)
    print("Waiting 20s before starting bot polling...")
    time.sleep(20)

    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), chat_handler))

    print("Starting Bot Polling (Background Thread)...")
    # stop_signals=None is critical for running in non-main thread
    application.run_polling(stop_signals=None)

# Initialize Bot in Background (Singleton)
@st.cache_resource
def start_bot():
    if not TELEGRAM_TOKEN:
        return None

    thread = threading.Thread(target=run_bot_loop, daemon=True)
    thread.start()
    return thread

# Start the bot
start_bot()
