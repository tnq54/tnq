import os
import asyncio
from fastapi import FastAPI
from google import genai # Import chuẩn SDK v2
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

app = FastAPI()

# 🔑 KHẮC PHỤC NameError: Thêm 'genai.' vào trước Client
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) 

# 🤖 Nơ-ron xử lý tin nhắn
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    # Gọi Gemini 3 Flash tư duy
    response = client.models.generate_content(
        model="gemini-3-flash", 
        contents=f"Bạn là Agent tnq. Trả lời sếp: {user_text}"
    )
    await update.message.reply_text(response.text)

@app.on_event("startup")
async def startup_event():
    # 📡 Kích hoạt Telegram Bot chạy ngầm khi App khởi động
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if token:
        application = ApplicationBuilder().token(token).build()
        application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
        await application.initialize()
        await application.start()
        await application.updater.start_polling()

@app.get("/")
def health():
    return {"status": "Live", "brain": "Gemini 3 Flash Online"}
    
