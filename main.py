import os
import threading
import telebot
from google import genai
from flask import Flask

# --- CẤU HÌNH ---
# Hãy đảm bảo tên biến môi trường trên Render khớp với 2 dòng dưới
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- KHỞI TẠO ---
bot = telebot.TeleBot(BOT_TOKEN)
app = Flask(__name__)

# Khởi tạo Client Gemini (SDK google-genai mới)
client = None
if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"Lỗi khởi tạo Gemini Client: {e}")

# --- WEB SERVER (Để Render không tắt Bot) ---
@app.route('/')
def home():
    return "Bot đang chạy model Gemini 3 Flash Preview!"

def run_web_server():
    # Render sẽ cấp port qua biến môi trường, mặc định 10000
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

# --- XỬ LÝ TIN NHẮN ---
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    chat_id = message.chat.id
    bot.send_chat_action(chat_id, 'typing') # Báo đang gõ...

    if not client:
        bot.reply_to(message, "Lỗi: Chưa có GEMINI_API_KEY!")
        return

    try:
        # GỌI MODEL GEMINI 3 FLASH PREVIEW
        # Lưu ý: Nếu model này chưa public cho tài khoản của bạn, nó sẽ báo lỗi 404/400
        response = client.models.generate_content(
            model="gemini-3-flash-preview", 
            contents=message.text
        )
        
        # Gửi phản hồi
        if response.text:
            bot.reply_to(message, response.text, parse_mode='Markdown')
        else:
            bot.reply_to(message, "Gemini không trả lời gì cả (Phản hồi rỗng).")

    except Exception as e:
        # In lỗi ra log để debug trên Render
        print(f"Lỗi API: {e}")
        
        # Báo về cho user biết
        error_message = str(e)
        if "404" in error_message or "not found" in error_message.lower():
            bot.reply_to(message, "Lỗi: Model 'gemini-3-flash-preview' chưa khả dụng với API Key này hoặc tên sai. Bạn thử đổi về 'gemini-2.0-flash-exp' xem sao nhé.")
        else:
            bot.reply_to(message, f"Lỗi kỹ thuật: {error_message}")

# --- CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    # 1. Chạy Web Server luồng phụ
    t = threading.Thread(target=run_web_server)
    t.start()

    # 2. Chạy Bot luồng chính
    print("Bot đang khởi động với Gemini 3 Flash Preview...")
    try:
        bot.infinity_polling()
    except Exception as e:
        print(f"Bot bị sập: {e}")
        
