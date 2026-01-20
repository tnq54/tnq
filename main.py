import os
import asyncio
from fastapi import FastAPI
from genai import Client # Sử dụng SDK v2 mới nhất sếp vừa gửi
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

app = FastAPI()

# 🧪 Khởi tạo Client với API Key sếp đã dán ở mục Environment
api_key = os.environ.get("GEMINI_API_KEY")
client = Client(api_key=api_key)

# 🤖 Logic xử lý với Gemini 3 Flash
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    
    # Ép não tư duy theo phong cách GRPO
    prompt = f"Bạn là Agent tnq. Hãy suy nghĩ cực kỳ logic trong thẻ <thinking>. Sau đó trả lời sếp: {user_text}"
    
    response = client.models.generate_content(
        model="gemini-3-flash", # Nâng cấp lên model mạnh nhất sếp vừa thấy
        contents=prompt,
    )
    await update.message.reply_text(response.text)

# 📡 Kích hoạt cổng Telegram
token = os.environ.get("TELEGRAM_BOT_TOKEN")
if token:
    application = ApplicationBuilder().token(token).build()
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    
    # Chạy Bot song song với Web Service trên Render
    @app.on_event("startup")
    async def startup_event():
        asyncio.create_task(application.initialize())
        asyncio.create_task(application.start())
        asyncio.create_task(application.updater.start_polling())

@app.get("/")
def health():
    return {"status": "Live", "brain": "Gemini 3 Flash Online", "version": "2026.1.0"}
    
