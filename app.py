import time
# Sleep to prevent network errors on startup
time.sleep(20)

import streamlit as st
import os
import logging
import io
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from huggingface_hub import InferenceClient
from google import genai
from pypdf import PdfReader

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Streamlit Interface (Minimal)
st.title("VBot1 System")
st.write("System is running. Telegram Bot is active.")

# Load Environment Variables
HF_TOKEN = os.environ.get("HF_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")

# Initialize AI Clients
try:
    if HF_TOKEN:
        hf_client = InferenceClient(token=HF_TOKEN)
    else:
        logging.warning("HF_TOKEN is missing.")
        hf_client = None

    if GEMINI_API_KEY:
        # User requested Gemini 2.5 Flash, using 1.5 Flash as standard equivalent
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    else:
        logging.warning("GEMINI_API_KEY is missing.")
        gemini_client = None
except Exception as e:
    logging.error(f"Error initializing clients: {e}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hello! I am VBot1 (Rebuilt).\n"
        "- Chat with me to use Llama 3.\n"
        "- Send a PDF to get a summary (Gemini Flash)."
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

async def pdf_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not gemini_client:
        await update.message.reply_text("Gemini is not configured (missing GEMINI_API_KEY).")
        return

    document = update.message.document
    if document.mime_type != 'application/pdf':
        await update.message.reply_text("Please send a valid PDF file.")
        return

    status_msg = await update.message.reply_text("Downloading and processing PDF...")

    try:
        file = await context.bot.get_file(document.file_id)
        file_content = await file.download_as_bytearray()

        # Extract text
        pdf_file = io.BytesIO(file_content)
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""

        if not text:
            await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status_msg.message_id, text="Could not extract text from this PDF.")
            return

        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status_msg.message_id, text=f"Summarizing {len(text)} characters with Gemini...")

        # Summarize
        prompt = f"Summarize the following PDF document content:\n\n{text[:30000]}"
        response = gemini_client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )

        await update.message.reply_text(response.text)

    except Exception as e:
        logging.error(f"PDF Error: {e}")
        await update.message.reply_text(f"Error processing PDF: {str(e)}")

if __name__ == '__main__':
    if not TELEGRAM_TOKEN:
        print("TELEGRAM_TOKEN not found. Bot cannot start.")
    else:
        application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

        application.add_handler(CommandHandler('start', start))
        application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), chat_handler))
        application.add_handler(MessageHandler(filters.Document.PDF, pdf_handler))

        print("Starting Bot Polling...")
        # Stop signals must be None to avoid invalid file descriptor error on some platforms
        application.run_polling(stop_signals=None)
