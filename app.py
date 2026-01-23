import streamlit as st
import os
import logging
import time
import threading
import asyncio
import io
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
st.write("Features: Llama 3 (Chat), Gemini 1.5 Flash (PDF Summary)")

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
    """
    Extracts and concatenated text from a PDF provided as raw bytes.
    
    Parameters:
        file_bytes (bytes): Raw bytes of a PDF file.
    
    Returns:
        str or None: The concatenated text of all pages if extraction succeeds, `None` if an error occurred during extraction.
    """
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
    """
    Summarizes a document text using Google Gemini 1.5 Flash.
    
    Parameters:
        text (str): Document text to summarize; only the first 30,000 characters are used.
    
    Returns:
        str: The generated summary. If the Gemini API key is missing, returns "Gemini API Key missing."; if the google-genai library is unavailable, returns "google-genai library missing."; on failure returns a string beginning with "Error summarizing: " followed by the error message.
    """
    if not GOOGLE_API_KEY:
        return "Gemini API Key missing."
    if not genai:
        return "google-genai library missing."

    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        # Using gemini-1.5-flash as standard.
        # If user meant 2.0, it can be updated to 'gemini-2.0-flash-exp'
        prompt = f"Please summarize the following document:\n\n{text[:30000]}"

        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        logger.error(f"Gemini Error: {e}")
        return f"Error summarizing: {e}"

# Bot Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Sends a greeting and usage instructions to the user.
    
    Replies to the incoming message with a brief welcome and two-line instructions:
    - send text to chat with Llama 3
    - send a PDF to summarize with Gemini 1.5 Flash
    """
    await update.message.reply_text(
        "Hello! I am VBot1.\n"
        "- Send text to chat (Llama 3).\n"
        "- Send a PDF file to summarize (Gemini 1.5 Flash)."
    )

async def chat_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle incoming text messages by sending the user's message to the configured Llama 3 HF client and returning the model's reply to the chat.
    
    If the HuggingFace client is not configured, notifies the user. Sends an interim "Thinking" message, invokes the HF chat completion to obtain a reply, and replaces the interim message with the model's response; on error, replaces the interim message with a generic error message and logs the failure.
    """
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
    """
    Handle incoming Telegram document messages: accept only PDFs, download and extract their text, summarize the extracted text with Gemini, and reply with the summary.
    
    The handler validates that the uploaded file is a PDF, sends status updates by editing a temporary message (e.g., "Downloading PDF...", "Reading PDF...", "Summarizing with Gemini..."), downloads the file bytes, extracts text using extract_pdf_text, obtains a summary from summarize_with_gemini, and delivers the summary to the chat. If the summary exceeds 4000 characters it is split into multiple messages. On extraction or summarization failure the handler updates the status message or replies with an appropriate error message and logs the exception.
    """
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
    """
    Start and maintain the Telegram bot polling loop in the current thread, restarting on failures.
    
    This function initializes a new asyncio event loop, builds the Telegram Application with configured handlers, and runs long‑running polling to receive updates. If TELEGRAM_TOKEN is not configured, it logs an error and returns without starting. On transient network errors the loop is retried after a short backoff; on other exceptions it retries with a longer backoff. The function blocks and runs indefinitely until the process exits.
    """
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN missing.")
        return

    logger.info("Bot thread started. Waiting 20s for network warmup...")
    time.sleep(20)

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
            # stop_signals=None prevents conflict with main thread signals
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
    """
    Start the Telegram bot loop in a background daemon thread.
    
    If TELEGRAM_TOKEN is missing, displays an error message in Streamlit and returns None.
    
    Returns:
        threading.Thread | None: The started daemon Thread running the bot loop, or `None` if the bot was not started due to a missing TELEGRAM_TOKEN.
    """
    if not TELEGRAM_TOKEN:
        st.error("TELEGRAM_TOKEN not found!")
        return None

    thread = threading.Thread(target=run_bot_loop, daemon=True)
    thread.start()
    return thread

if __name__ == "__main__":
    start_bot_thread()