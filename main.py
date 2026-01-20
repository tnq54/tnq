import os
import asyncio
from fastapi import FastAPI
import google.generativeai as genai
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

app = FastAPI()

# 🧠 Cấu hình não bộ 2.5 Flash
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash-exp')

# 🤖 Logic xử lý tin nhắn Telegram
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    # Tư duy GRPO (Giả lập trong prompt)
    prompt = f"Hành xử như Agent tnq. Hãy suy nghĩ logic trước khi đáp. User nói: {user_text}"
    response = model.generate_content(prompt)
    await update.message.reply_text(response.text)

# Khởi tạo Bot Telegram
token = os.environ.get("TELEGRAM_BOT_TOKEN")
if token:
    application = ApplicationBuilder().token(token).build()
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    # Chạy bot ở chế độ nền
    asyncio.create_task(application.initialize())
    asyncio.create_task(application.start())
    asyncio.create_task(application.updater.start_polling())

@app.get("/")
def health():
    return {"status": "Live", "brain": "Gemini 2.5 Flash", "interface": "Telegram Active"}
    
