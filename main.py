import os
import threading
import telebot
from flask import Flask

# --- CẤU HÌNH ---
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") # Lấy token từ Environment Variable trên Render
bot = telebot.TeleBot(BOT_TOKEN)
app = Flask(__name__)

# --- PHẦN 1: WEB SERVER ĐỂ RENDER KHÔNG TẮT APP ---
@app.route('/')
def home():
    return "Bot is running!"

def run_web_server():
    # Render cung cấp cổng qua biến môi trường PORT, mặc định là 10000
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

# --- PHẦN 2: BOT TELEGRAM (ĐÃ BỎ CODE GEMINI) ---
@bot.message_handler(func=lambda message: True)
def echo_all(message):
    # Bot chỉ đơn giản là nhại lại tin nhắn (Echo) để test
    # Không còn code gọi Gemini AI ở đây nữa
    bot.reply_to(message, "Bot đang chạy ổn định! Bạn vừa nói: " + message.text)

# --- PHẦN 3: CHẠY SONG SONG CẢ HAI ---
if __name__ == "__main__":
    print("Đang khởi động...")
    
    # Chạy Web Server ở một luồng riêng (Thread)
    t = threading.Thread(target=run_web_server)
    t.start()

    # Chạy Bot Telegram ở luồng chính
    try:
        print("Bot bắt đầu polling...")
        bot.infinity_polling() # Dùng infinity_polling ổn định hơn polling thường
    except Exception as e:
        print(f"Lỗi bot: {e}")
        
