import streamlit as st
import os
import logging
import time
import threading
import asyncio
import io
import socket
from pypdf import PdfReader
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from telegram.error import NetworkError
from huggingface_hub import InferenceClient

# Import Google GenAI (New SDK)
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Streamlit Interface
st.title("VBot1 System (Rebuilt)")
st.write("System Status: Running")
st.write("Features: Llama 3 (Chat), Gemini 2.5 Flash (PDF Summary)")

# Load Environment Variables
HF_TOKEN = os.environ.get("HF_TOKEN")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Initialize Clients
hf_client = None
if HF_TOKEN:
    try:
        hf_client = InferenceClient(token=HF_TOKEN)
        logger.info("HF Client Initialized")
    except Exception as e:
        logger.error(f"HF Client Init Error: {e}")

# PDF Helper Functions
def extract_pdf_text(file_bytes):
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        logger.error(f"PDF Extraction Error: {e}")
        return None

def summarize_with_gemini(text):
    if not GOOGLE_API_KEY:
        return "Gemini API Key missing."
    if not genai:
        return "google-genai library missing."

    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        # Using gemini-2.5-flash as requested
        prompt = f"Please summarize the following document:\n\n{text[:30000]}"

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        logger.error(f"Gemini Error: {e}")
        return f"Error summarizing: {e}"

# Bot Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hello! I am VBot1.\n"
        "- Send text to chat (Llama 3).\n"
        "- Send a PDF file to summarize (Gemini 2.5 Flash)."
    )

async def chat_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not hf_client:
        await update.message.reply_text("Llama 3 not configured.")
        return

    user_text = update.message.text
    status_msg = await update.message.reply_text("Thinking (Llama 3)...")

    try:
        # Use simple chat completion
        messages = [{"role": "user", "content": user_text}]
        response = hf_client.chat_completion(
            messages=messages,
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            max_tokens=800
        )
        reply = response.choices[0].message.content
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status_msg.message_id, text=reply)
    except Exception as e:
        logger.error(f"Llama Chat Error: {e}")
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status_msg.message_id, text="Error processing chat.")

async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    document = update.message.document
    if document.mime_type != 'application/pdf':
        await update.message.reply_text("Please send a PDF file.")
        return

    status_msg = await update.message.reply_text("Downloading PDF...")

    try:
        file = await context.bot.get_file(document.file_id)
        file_bytes = await file.download_as_bytearray()

        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status_msg.message_id, text="Reading PDF...")
        text = extract_pdf_text(file_bytes)

        if not text:
            await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status_msg.message_id, text="Could not extract text from PDF.")
            return

        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status_msg.message_id, text="Summarizing with Gemini...")
        summary = summarize_with_gemini(text)

        # Send summary (split if too long)
        if len(summary) > 4000:
            for i in range(0, len(summary), 4000):
                await update.message.reply_text(summary[i:i+4000])
        else:
            await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status_msg.message_id, text=summary)

    except Exception as e:
        logger.error(f"Document Handler Error: {e}")
        await update.message.reply_text(f"Error handling document: {e}")

# Bot Thread Logic with Retry
def run_bot_loop():
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN missing.")
        return

    # Add sleep to prevent Errno -5 (No address associated with hostname)
    logger.info("Bot thread started. Waiting for network...")
    for i in range(20):
        try:
            with socket.create_connection(("api.telegram.org", 443), timeout=3):
                pass
            logger.info("Network detected.")
            break
        except Exception:
            time.sleep(1)

    while True:
        try:
            logger.info("Initializing Bot Application...")
            # Create new loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

            application.add_handler(CommandHandler('start', start))
            application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), chat_handler))
            application.add_handler(MessageHandler(filters.Document.PDF, document_handler))

            logger.info("Starting Polling...")
            # stop_signals=None prevents conflict with main thread signals and fixes Invalid file descriptor error
            application.run_polling(stop_signals=None)

        except NetworkError as e:
            logger.error(f"NetworkError in bot loop: {e}. Retrying in 10s...")
            time.sleep(10)
        except Exception as e:
            logger.error(f"Critical Error in bot loop: {e}. Retrying in 20s...")
            time.sleep(20)
        finally:
            try:
                loop.close()
            except:
                pass

# Start Bot in Background
@st.cache_resource
def start_bot_thread():
    if not TELEGRAM_TOKEN:
        st.error("TELEGRAM_TOKEN not found!")
        return None

    thread = threading.Thread(target=run_bot_loop, daemon=True)
    thread.start()
    return thread

if __name__ == "__main__":
    start_bot_thread()
