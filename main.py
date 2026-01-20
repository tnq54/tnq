import os
import asyncio
from fastapi import FastAPI
from google import genai # Đúng chuẩn SDK v2 sếp tìm thấy
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

app = FastAPI()

# 🧠 Khởi tạo bộ não Gemini 3 Flash
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    # Gọi model Gemini 3 Flash xử lý
    response = client.models.generate_content(
        model="gemini-3-flash", 
        contents=f"Bạn là Agent tnq. Trả lời sếp: {user_text}"
    )
    await update.message.reply_text(response.text)

# 📡 Biến toàn cục để quản lý Bot
application = None

@app.on_event("startup")
async def startup_event():
    global application
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if token:
        # Khởi tạo Bot theo cách an toàn nhất cho FastAPI
        application = ApplicationBuilder().token(token).build()
        application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
        
        await application.initialize()
        await application.start()
        # Dùng cơ chế loop của FastAPI để tránh lỗi Updater
        asyncio.create_task(application.updater.start_polling())

@app.get("/")
def health():
    return {"status": "Live", "brain": "Gemini 3 Flash Online", "bot": "Polling Started"}
    
