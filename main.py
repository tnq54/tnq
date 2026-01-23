import os
import threading
import telebot
import numpy as np
from huggingface_hub import InferenceClient
from flask import Flask

# --- CẤU HÌNH ---
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")

# --- KHỞI TẠO ---
bot = telebot.TeleBot(BOT_TOKEN)
app = Flask(__name__)
client = InferenceClient(api_key=HF_TOKEN)

# --- MODELS ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL = "HuggingFaceH4/zephyr-7b-beta"

# --- 📚 KHO TRI THỨC (RAG) ---
# Đây là "não phụ" của Bot. Sếp sửa nội dung trong này nhé.
KNOWLEDGE_BASE = [
    "Tôi là Trợ lý AI TNQ, chạy trên nền tảng Render với framework Flask.",
    "Tôi sử dụng model 'Zephyr 7B Beta' từ Hugging Face.",
    "Sếp đang dạy tôi kỹ thuật RAG để tôi trả lời thông minh hơn.",
    "Mã nguồn của tôi kết hợp giữa Flask (để giữ server sống) và Telebot (để chat).",
    "Sở thích của sếp là lập trình AI và tối ưu hóa hệ thống tự động."
]

# Biến lưu trữ Vector (Bộ nhớ tạm)
VECTOR_DB = []

def get_embedding(text):
    """Gọi Hugging Face API để lấy embedding"""
    try:
        # feature_extraction trả về ndarray
        output = client.feature_extraction(text, model=EMBEDDING_MODEL)

        # Xử lý output shape (thường là (1, 384))
        if isinstance(output, np.ndarray) and output.ndim == 2:
            return output[0]
        if isinstance(output, list) and len(output) > 0 and isinstance(output[0], list):
            return np.array(output[0])

        return np.array(output)
    except Exception as e:
        print(f"Lỗi embedding: {e}")
        return None

def build_vector_db():
    """Mã hóa văn bản thành số (Vector) để tìm kiếm"""
    global VECTOR_DB
    print("--- ĐANG NẠP DỮ LIỆU RAG... ---")
    try:
        for text in KNOWLEDGE_BASE:
            embed = get_embedding(text)
            if embed is not None:
                VECTOR_DB.append({"text": text, "embedding": embed})
        print(f"--- ĐÃ NẠP XONG {len(VECTOR_DB)} MẢNH KIẾN THỨC ---")
    except Exception as e:
        print(f"LỖI BUILD DB: {e}")

def find_best_context(query_text):
    """Tìm thông tin liên quan nhất trong kho não"""
    if not VECTOR_DB: return None
    try:
        query_embed = get_embedding(query_text)
        if query_embed is None: return None
        
        # So sánh với kho dữ liệu
        best_score = -1
        best_text = ""
        for item in VECTOR_DB:
            # Tính cosine similarity (giả sử vector đã chuẩn hóa hoặc dùng dot product đơn giản)
            # Embedding từ HF thường là 1D array cho 1 câu input
            score = np.dot(query_embed, item["embedding"])
            if score > best_score:
                best_score = score
                best_text = item["text"]
        
        # Chỉ lấy nếu độ giống > 0.4 (threshold tùy chỉnh)
        return best_text if best_score > 0.4 else None
    except Exception as e:
        print(f"Lỗi tìm kiếm: {e}")
        return None

# --- WEB SERVER (Để Render không tắt Bot) ---
@app.route('/')
def home():
    return "Bot TNQ đang chạy RAG với Hugging Face!"

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
        
        # BƯỚC 2: Tạo Prompt
        messages = []
        if context_info:
            sys_instruct = f"""
            Bạn là trợ lý AI hữu ích. Dưới đây là thông tin bối cảnh liên quan:
            "{context_info}"
            Hãy trả lời câu hỏi của người dùng dựa trên thông tin này.
            Nếu thông tin không đủ, hãy dùng kiến thức của bạn nhưng ưu tiên bối cảnh trên.
            Trả lời bằng ngôn ngữ của người dùng (Tiếng Việt/Anh).
            """
        else:
            sys_instruct = "Bạn là trợ lý AI hữu ích. Hãy trả lời thân thiện và ngắn gọn."

        messages.append({"role": "system", "content": sys_instruct})
        messages.append({"role": "user", "content": user_text})

        # BƯỚC 3: Gọi Hugging Face Inference API
        response = client.chat_completion(
            model=CHAT_MODEL,
            messages=messages,
            max_tokens=500
        )
        
        bot_reply = response.choices[0].message.content
        bot.reply_to(message, bot_reply, parse_mode='Markdown')

    except Exception as e:
        print(f"Lỗi: {e}")
        bot.reply_to(message, f"Lỗi xử lý: {str(e)}")

# --- CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    # 1. Nạp kiến thức vào não trước
    build_vector_db()
    
    # 2. Chạy Web Server luồng phụ
    t = threading.Thread(target=run_web_server)
    t.start()
    
    # 3. Chạy Bot luồng chính
    print("Bot đang khởi động với Hugging Face...")
    try:
        bot.infinity_polling()
    except Exception as e:
        print(f"Bot bị sập: {e}")
