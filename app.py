import streamlit as st
import time
import os
import threading
import asyncio
import io
import pypdf
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from huggingface_hub import InferenceClient
from google import genai

# Load environment variables
HF_TOKEN = os.environ.get("HF_TOKEN")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Streamlit Interface
st.set_page_config(page_title="vbot1", page_icon="🤖")
st.title("vbot1 is running 🚀")
st.write("System Status: **Active**")
st.write("The bot is running in the background.")

# --- Bot Logic ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hello! I am vbot1.\n"
        "- Send me text to chat (Llama 3).\n"
        "- Send me a PDF to analyze (Gemini 2.0 Flash)."
    )

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    if not user_text:
        return

    try:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

        # Llama 3 via InferenceClient
        client = InferenceClient(token=HF_TOKEN)
        # Using Meta-Llama-3-8B-Instruct
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

        messages = [{"role": "user", "content": user_text}]

        # Using chat_completion API
        response = client.chat_completion(
            messages=messages,
            model=model_id,
            max_tokens=1024
        )

        bot_reply = response.choices[0].message.content
        await update.message.reply_text(bot_reply)

    except Exception as e:
        error_msg = f"Error processing text: {str(e)}"
        print(error_msg)
        await update.message.reply_text("Sorry, I encountered an error processing your message.")

async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    document = update.message.document

    if not document or document.mime_type != 'application/pdf':
        await update.message.reply_text("Please send a valid PDF file.")
        return

    try:
        await update.message.reply_text("Downloading and analyzing PDF...")
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

        # Download file from Telegram
        file = await context.bot.get_file(document.file_id)
        file_byte_array = await file.download_as_bytearray()

        # Extract text using pypdf
        try:
            pdf_stream = io.BytesIO(file_byte_array)
            reader = pypdf.PdfReader(pdf_stream)
            text_content = ""
            for page in reader.pages:
                text_content += page.extract_text() + "\n"
        except Exception as e:
            await update.message.reply_text(f"Failed to read PDF: {str(e)}")
            return

        if not text_content.strip():
            await update.message.reply_text("The PDF appears to be empty or contains no extractable text.")
            return

        # Gemini 2.0 Flash via google-genai
        # Note: Using "gemini-2.0-flash-exp" or "gemini-2.0-flash" depending on availability.
        # User requested "Gemini 2.0 Flash".
        client = genai.Client(api_key=GOOGLE_API_KEY)

        prompt = "Analyze the following PDF content. Summarize it and answer any potential questions a user might have:\n\n"

        # Truncate if necessary (Gemini has large context, but let's be safe with very large PDFs)
        # 1 token ~= 4 chars. 1M tokens context is huge, so we usually don't need to truncate for typical PDFs.

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt + text_content
        )

        await update.message.reply_text(response.text)

    except Exception as e:
        error_msg = f"Error processing PDF: {str(e)}"
        print(error_msg)
        await update.message.reply_text("Sorry, I encountered an error analyzing the PDF.")

def run_bot_process():
    # Immortality Fix: Sleep 20s for DNS
    print("Initializing Bot: Sleeping for 20s to fix DNS issues...")
    time.sleep(20)

    # Check tokens
    if not TELEGRAM_TOKEN:
        print("Error: TELEGRAM_TOKEN is missing.")
        return

    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.Document.PDF, handle_pdf))

    print("Bot is starting polling...")
    # Immortality Fix: stop_signals=None
    app.run_polling(stop_signals=None)

# Start Bot in Background Thread
if "bot_started" not in st.session_state:
    st.session_state["bot_started"] = True
    thread = threading.Thread(target=run_bot_process, daemon=True)
    thread.start()
