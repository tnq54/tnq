import os
import threading
import telebot
import numpy as np
from google import genai
from flask import Flask

# --- CẤU HÌNH ---
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- KHỞI TẠO ---
bot = telebot.TeleBot(BOT_TOKEN)
app = Flask(__name__)
client = genai.Client(api_key=GEMINI_API_KEY)

# --- 📚 KHO TRI THỨC (RAG) ---
# Đây là "não phụ" của Bot. Sếp sửa nội dung trong này nhé.
KNOWLEDGE_BASE = [
    "Tôi là Trợ lý AI TNQ, chạy trên nền tảng Render với framework Flask.",
    "Tôi sử dụng model 'gemini-3-flash-preview' mới nhất của Google.",
    "Sếp đang dạy tôi kỹ thuật RAG để tôi trả lời thông minh hơn.",
    "Mã nguồn của tôi kết hợp giữa Flask (để giữ server sống) và Telebot (để chat).",
    "Sở thích của sếp là lập trình AI và tối ưu hóa hệ thống tự động."
]

# Biến lưu trữ Vector (Bộ nhớ tạm)
VECTOR_DB = []

def build_vector_db():
    """Mã hóa văn bản thành số (Vector) để tìm kiếm"""
    global VECTOR_DB
    print("--- ĐANG NẠP DỮ LIỆU RAG... ---")
    try:
        for text in KNOWLEDGE_BASE:
            # Dùng model embedding để mã hóa
            result = client.models.embed_content(
                model="text-embedding-004",
                contents=text
            )
            VECTOR_DB.append({"text": text, "embedding": result.embeddings[0].values})
        print(f"--- ĐÃ NẠP XONG {len(VECTOR_DB)} MẢNH KIẾN THỨC ---")
    except Exception as e:
        print(f"LỖI EMBEDDING: {e}")

def find_best_context(query_text):
    """Tìm thông tin liên quan nhất trong kho não"""
    if not VECTOR_DB: return None
    try:
        # Mã hóa câu hỏi của sếp
        query_embed = client.models.embed_content(
            model="text-embedding-004",
            contents=query_text
        ).embeddings[0].values

        # So sánh với kho dữ liệu
        best_score = -1
        best_text = ""
        for item in VECTOR_DB:
            score = np.dot(query_embed, item["embedding"])
            if score > best_score:
                best_score = score
                best_text = item["text"]

        # Chỉ lấy nếu độ giống > 0.5
        return best_text if best_score > 0.5 else None
    except Exception as e:
        print(f"Lỗi tìm kiếm: {e}")
        return None

# --- WEB SERVER (Để Render không tắt Bot) ---
@app.route('/')
def home():
    return "Bot TNQ đang chạy RAG với Gemini 3 Flash Preview!"

def run_web_server():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

# --- XỬ LÝ TIN NHẮN ---
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    chat_id = message.chat.id
    user_text = message.text
    bot.send_chat_action(chat_id, 'typing')

    try:
        # BƯỚC 1: Tìm kiếm thông tin (RAG)
        context_info = find_best_context(user_text)

        # BƯỚC 2: Tạo Prompt (Kết hợp tìm được + ngôn ngữ sếp)
        if context_info:
            sys_instruct = f"""
            Bạn là trợ lý AI. Hãy trả lời dựa trên thông tin sau:
            "{context_info}"
            Yêu cầu: Trả lời bằng cùng ngôn ngữ với người dùng (Việt/Anh).
            """
        else:
            sys_instruct = "Bạn là trợ lý AI. Hãy trả lời ngắn gọn bằng ngôn ngữ của người dùng."

        # BƯỚC 3: Gọi Gemini 3 Flash Preview
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=user_text,
            config={"system_instruction": sys_instruct}
        )

        bot.reply_to(message, response.text, parse_mode='Markdown')

    except Exception as e:
        print(f"Lỗi: {e}")
        bot.reply_to(message, f"Lỗi não bộ: {str(e)}")

# --- CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    # 1. Nạp kiến thức vào não trước
    build_vector_db()

    # 2. Chạy Web Server luồng phụ
    t = threading.Thread(target=run_web_server)
    t.start()

    # 3. Chạy Bot luồng chính
    if os.environ.get("SPACE_ID"):
        print("Bot đang chạy trên Hugging Face Spaces. Tắt Polling để tránh xung đột với Render.")
    else:
        print("Bot đang khởi động với RAG...")
        try:
            bot.infinity_polling()
        except Exception as e:
            print(f"Bot bị sập: {e}")
