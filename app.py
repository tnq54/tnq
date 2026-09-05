import streamlit as st
import time
import os
import sys
import json
import subprocess
import threading
import asyncio
import io
import logging
import zipfile
from PIL import Image
from pypdf import PdfReader
from telegram import Update
from telegram.error import NetworkError
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from huggingface_hub import InferenceClient, HfApi

# Page configuration
st.set_page_config(
    page_title="Hugging Face Image LoRA Training Studio",
    page_icon="🎨",
    layout="wide"
)

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Try importing Google GenAI
try:
    from google import genai
except ImportError:
    genai = None

# Load Environment Variables
HF_TOKEN = os.environ.get("HF_TOKEN", "")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# Directories
DATASET_DIR = "./dataset"
OUTPUT_DIR = "./output"
CONFIG_DIR = "./config"
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

# Initialize HF Client
if HF_TOKEN:
    try:
        hf_client = InferenceClient(token=HF_TOKEN)
    except Exception as e:
        logger.error(f"Failed to init HF Client: {e}")
        hf_client = None
else:
    hf_client = None

# PDF Text Extraction
def extract_pdf_text(file_bytes):
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        logger.error(f"PDF Extraction Error: {e}")
        return None

# Gemini Summarization & Image Captioning
def summarize_with_gemini(text):
    if not GOOGLE_API_KEY:
        return "Error: GOOGLE_API_KEY not found."
    if not genai:
        return "Error: google-genai library not installed."

    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=f"Summarize this document:\n\n{text[:30000]}"
        )
        return response.text
    except Exception as e:
        logger.error(f"Gemini Error: {e}")
        return f"Error summarizing: {e}"

def generate_image_caption_with_gemini(image_bytes):
    if not GOOGLE_API_KEY or not genai:
        return "a high quality detailed photo"
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        img = Image.open(io.BytesIO(image_bytes))
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=["Provide a detailed 1-sentence prompt caption describing the subject, style, lighting, and background of this image for LoRA diffusion model training.", img]
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini Image Captioning Error: {e}")
        return "a high quality detailed photo"

# Telegram Bot Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Welcome to VBot1 Image LoRA Studio!\n"
        "- Chat with me (Llama 3).\n"
        "- Send a PDF to summarize (Gemini 1.5 Flash)."
    )

async def chat_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    if not hf_client:
        await update.message.reply_text("Llama 3 is not available (HF_TOKEN missing).")
        return

    status_msg = await update.message.reply_text("Thinking...")
    try:
        messages = [{"role": "user", "content": user_text}]
        completion = hf_client.chat_completion(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=messages,
            max_tokens=500
        )
        reply = completion.choices[0].message.content
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status_msg.message_id, text=reply)
    except Exception as e:
        logger.error(f"Llama 3 Error: {e}")
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status_msg.message_id, text=f"Error: {e}")

async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if doc.mime_type != 'application/pdf':
        await update.message.reply_text("Please send a PDF file.")
        return

    status_msg = await update.message.reply_text("Downloading PDF...")
    try:
        file = await context.bot.get_file(doc.file_id)
        file_bytes = await file.download_as_bytearray()

        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status_msg.message_id, text="Extracting text...")
        text = extract_pdf_text(file_bytes)

        if not text:
            await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status_msg.message_id, text="No text found in PDF.")
            return

        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status_msg.message_id, text="Summarizing (Gemini 1.5 Flash)...")
        summary = summarize_with_gemini(text)

        if len(summary) > 4000:
            for i in range(0, len(summary), 4000):
                await update.message.reply_text(summary[i:i+4000])
        else:
            await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status_msg.message_id, text=summary)

    except Exception as e:
        logger.error(f"Document Error: {e}")
        await update.message.reply_text(f"Error processing document: {e}")

# Bot Runner
def run_bot():
    logger.info("Waiting 5s for network initialization...")
    time.sleep(5)

    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN is missing")
        return

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    while True:
        try:
            application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
            application.add_handler(CommandHandler("start", start))
            application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat_handler))
            application.add_handler(MessageHandler(filters.Document.PDF, document_handler))

            application.run_polling(stop_signals=None, close_loop=False)

        except NetworkError as e:
            logger.error(f"Network error during polling: {e}. Retrying in 10s...")
            time.sleep(10)
        except Exception as e:
            logger.error(f"Error during polling: {e}")
            time.sleep(10)

# Background Thread Launcher
if __name__ == "__main__" or "bot_thread" not in st.session_state:
    if "bot_thread" not in st.session_state:
        st.session_state.bot_thread = True
        thread = threading.Thread(target=run_bot, daemon=True)
        thread.start()

# --- Streamlit Navigation & UI ---
st.title("🎨 Hugging Face Image LoRA Training Studio")
st.caption("Huấn luyện LoRA Model Cho Ảnh Trên Hugging Face Spaces (FLUX.1 / SDXL / SD 1.5)")

main_tabs = st.tabs([
    "📸 1. Dataset & Auto-Caption",
    "⚙️ 2. Cấu Hình LoRA",
    "🚀 3. Huấn Luyện & Logs",
    "📦 4. Export & Push HF Hub",
    "🤖 5. Telegram Bot & System"
])

# TAB 1: DATASET MANAGEMENT
with main_tabs[0]:
    st.header("📸 Quản Lý Dataset Ảnh & Tự Động Tạo Caption")
    st.write("Tải lên danh sách ảnh huấn luyện (`.png`, `.jpg`, `.jpeg`, `.webp`). Gemini 1.5 Flash sẽ hỗ trợ tạo caption tự động.")

    uploaded_files = st.file_uploader(
        "Chọn các tệp ảnh huấn luyện:",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("💾 Lưu Dataset & Tạo Captions", type="primary"):
            progress_bar = st.progress(0)
            for idx, file in enumerate(uploaded_files):
                img_bytes = file.read()
                filename_base = os.path.splitext(file.name)[0]
                img_path = os.path.join(DATASET_DIR, f"{filename_base}.png")

                # Save Image
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img.save(img_path)

                # Generate Caption
                caption = generate_image_caption_with_gemini(img_bytes)
                cap_path = os.path.join(DATASET_DIR, f"{filename_base}.txt")
                with open(cap_path, "w", encoding="utf-8") as f:
                    f.write(caption)

                progress_bar.progress((idx + 1) / len(uploaded_files))

            st.success(f"Đã lưu thành công {len(uploaded_files)} ảnh và tệp caption vào `{DATASET_DIR}`!")

    # Existing Dataset Files
    st.subheader("📂 Danh Sách Tệp Trong Dataset Hiện Tại")
    if os.path.exists(DATASET_DIR):
        files = os.listdir(DATASET_DIR)
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        st.info(f"Tổng số ảnh trong dataset: {len(image_files)}")

        if image_files:
            cols = st.columns(3)
            for idx, img_f in enumerate(image_files[:6]):
                with cols[idx % 3]:
                    img_path = os.path.join(DATASET_DIR, img_f)
                    txt_path = os.path.join(DATASET_DIR, os.path.splitext(img_f)[0] + ".txt")

                    st.image(img_path, use_container_width=True)
                    caption_val = ""
                    if os.path.exists(txt_path):
                        with open(txt_path, "r", encoding="utf-8") as f:
                            caption_val = f.read()

                    new_caption = st.text_area(f"Caption for {img_f}", value=caption_val, key=f"cap_{img_f}")
                    if new_caption != caption_val:
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(new_caption)
                        st.caption("Saved!")

            # Zip Export Download
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for f in os.listdir(DATASET_DIR):
                    f_path = os.path.join(DATASET_DIR, f)
                    if os.path.isfile(f_path):
                        zip_file.write(f_path, arcname=f)
            zip_buffer.seek(0)

            st.download_button(
                label="📦 Tải Về Dataset ZIP",
                data=zip_buffer,
                file_name="lora_dataset.zip",
                mime="application/zip"
            )

# TAB 2: LORA HYPERPARAMETERS CONFIGURATION
with main_tabs[1]:
    st.header("⚙️ Thiết Lập Cấu Hình LoRA & Siêu Tham Số")

    col1, col2 = st.columns(2)

    with col1:
        base_model_preset = st.selectbox(
            "Chọn Mô Hình Gốc (Base Model):",
            [
                "black-forest-labs/FLUX.1-dev",
                "stabilityai/stable-diffusion-xl-base-1.0",
                "runwayml/stable-diffusion-v1-5",
                "meta-llama/Llama-3.2-11B-Vision-Instruct",
                "Tùy Chỉnh Repo ID"
            ]
        )
        if base_model_preset == "Tùy Chỉnh Repo ID":
            base_model = st.text_input("Nhập HF Repo ID:", value="black-forest-labs/FLUX.1-dev")
        else:
            base_model = base_model_preset

        resolution = st.select_slider("Độ phân giải ảnh (Resolution):", options=[512, 768, 1024, 1280], value=1024)
        lora_r = st.selectbox("LoRA Rank (r):", [4, 8, 16, 32, 64, 128], index=2)
        lora_alpha = st.slider("LoRA Alpha:", min_value=1, max_value=128, value=16)

        learning_rate = st.number_input("Base Learning Rate:", value=1e-4, format="%.6f")
        unet_lr = st.number_input("UNet / Transformer Backbone LR:", value=1e-4, format="%.6f")
        text_encoder_lr = st.number_input("Text Encoder LR:", value=5e-5, format="%.6f")

    with col2:
        repeat = st.number_input("Số lần lặp Dataset (Repeat):", value=10, min_value=1)
        save_every_n_epochs = st.number_input("Lưu Checkpoint sau mỗi N Epochs:", value=1, min_value=1)
        mixed_precision = st.selectbox("Chế độ Mixed Precision:", ["fp16", "bf16", "no"], index=0)

        gradient_checkpointing = st.checkbox("Gradient Checkpointing (Tiết kiệm VRAM)", value=True)
        use_4bit = st.checkbox("4-bit QLoRA Quantization", value=False)
        use_safetensors = st.checkbox("Xuất Định Dạng SafeTensors (.safetensors)", value=True)

        enable_bucket = st.checkbox("Aspect Ratio Bucketing (ARB)", value=True)
        bucket_reso_steps = st.number_input("Bucket Reso Steps:", value=64)
        min_bucket_reso = st.number_input("Min Bucket Reso:", value=256)
        max_bucket_reso = st.number_input("Max Bucket Reso:", value=1024)

    st.subheader("🛠️ Cấu Hình Nâng Cao (Conv LoRA & Min SNR)")
    c1, c2, c3 = st.columns(3)
    with c1:
        conv_dim = st.number_input("Conv LoRA Dim:", value=4)
    with c2:
        conv_alpha = st.number_input("Conv LoRA Alpha:", value=4)
    with c3:
        min_snr_gamma = st.number_input("Min SNR Gamma:", value=5.0)

    # Save Config JSON
    training_config = {
        "base_model": base_model,
        "dataset_dir": DATASET_DIR,
        "output_dir": OUTPUT_DIR,
        "config_save_dir": CONFIG_DIR,
        "resolution": resolution,
        "learning_rate": learning_rate,
        "unet_lr": unet_lr,
        "text_encoder_lr": text_encoder_lr,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "repeat": repeat,
        "save_every_n_epochs": save_every_n_epochs,
        "caption_extension": ".txt",
        "warmup_ratio": 0.05,
        "max_seq_length": 512,
        "mixed_precision": mixed_precision,
        "gradient_checkpointing": gradient_checkpointing,
        "use_4bit": use_4bit,
        "use_safetensors": use_safetensors,
        "enable_bucket": enable_bucket,
        "bucket_reso_steps": bucket_reso_steps,
        "min_bucket_reso": min_bucket_reso,
        "max_bucket_reso": max_bucket_reso,
        "conv_dim": conv_dim,
        "conv_alpha": conv_alpha,
        "min_snr_gamma": min_snr_gamma
    }

    config_json_str = json.dumps(training_config, indent=2, ensure_ascii=False)
    st.download_button(
        "💾 Tải Về File Cấu Hình (training_config.json)",
        data=config_json_str,
        file_name="training_config.json",
        mime="application/json"
    )

# TAB 3: TRAINING EXECUTION
with main_tabs[2]:
    st.header("🚀 Bắt Đầu Huấn Luyện & Theo Dõi Tiến Độ")

    if st.button("🔥 Chạy Train Image LoRA Ngay", type="primary"):
        st.info("Đang khởi chạy tiến trình `train_lora.py`...")

        # Save config file first
        conf_file = os.path.join(CONFIG_DIR, "training_config.json")
        with open(conf_file, "w", encoding="utf-8") as f:
            f.write(config_json_str)

        cmd = [
            sys.executable, "train_lora.py",
            "--base_model", base_model,
            "--dataset_dir", DATASET_DIR,
            "--output_dir", OUTPUT_DIR,
            "--config_save_dir", CONFIG_DIR,
            "--resolution", str(resolution),
            "--learning_rate", str(learning_rate),
            "--unet_lr", str(unet_lr),
            "--text_encoder_lr", str(text_encoder_lr),
            "--lora_r", str(lora_r),
            "--lora_alpha", str(lora_alpha),
            "--repeat", str(repeat),
            "--save_every_n_epochs", str(save_every_n_epochs),
            "--mixed_precision", mixed_precision,
            "--conv_dim", str(conv_dim),
            "--conv_alpha", str(conv_alpha),
            "--min_snr_gamma", str(min_snr_gamma)
        ]

        if gradient_checkpointing:
            cmd.append("--gradient_checkpointing")
        if use_4bit:
            cmd.append("--use_4bit")
        if use_safetensors:
            cmd.append("--use_safetensors")
        if enable_bucket:
            cmd.append("--enable_bucket")

        log_area = st.empty()

        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            logs = ""
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    logs += line
                    log_area.code(logs[-3000:])

            process.wait()
            if process.returncode == 0:
                st.success("🎉 Quá trình huấn luyện LoRA thành công!")
            else:
                st.error("❌ Quá trình huấn luyện có lỗi xảy ra. Vui lòng kiểm tra logs bên trên.")
        except Exception as e:
            st.error(f"Lỗi khi thực thi train_lora.py: {e}")

# TAB 4: EXPORT & PUSH TO HF HUB
with main_tabs[3]:
    st.header("📦 Export & Push Weights Lên Hugging Face Hub")

    if os.path.exists(OUTPUT_DIR):
        out_files = os.listdir(OUTPUT_DIR)
        st.write("📂 Các Checkpoints LoRA Đã Tạo:")
        for of in out_files:
            file_p = os.path.join(OUTPUT_DIR, of)
            if os.path.isfile(file_p):
                with open(file_p, "rb") as f:
                    st.download_button(f"📥 Tải Về {of}", data=f, file_name=of)

    st.subheader("🤗 Push Adapter Lên Hugging Face Hub Repo")
    target_repo = st.text_input("Nhập Target Hugging Face Repo ID (Ví dụ: `username/flux-my-style-lora`):")
    target_token = st.text_input("Hugging Face API Token (Mặc định dùng HF_TOKEN):", value=HF_TOKEN, type="password")

    if st.button("🚀 Push All Weights To HF Hub", type="primary"):
        if not target_repo:
            st.error("Vui lòng nhập Repo ID!")
        else:
            try:
                api = HfApi(token=target_token or HF_TOKEN)
                api.create_repo(repo_id=target_repo, exist_ok=True, repo_type="model")
                api.upload_folder(
                    folder_path=OUTPUT_DIR,
                    repo_id=target_repo,
                    repo_type="model"
                )
                st.success(f"✅ Đã tải thành công Adapter LoRA lên https://huggingface.co/{target_repo}")
            except Exception as e:
                st.error(f"Lỗi khi push lên HF Hub: {e}")

# TAB 5: TELEGRAM BOT & SYSTEM STATUS
with main_tabs[4]:
    st.header("🤖 Telegram Bot & System Diagnostic Status")
    st.write(f"- Llama 3 Client: {'✅ Hoạt Động' if hf_client else '❌ Chưa Cấu Hình HF_TOKEN'}")
    st.write(f"- Gemini 1.5 Flash: {'✅ Hoạt Động' if GOOGLE_API_KEY else '❌ Chưa Cấu Hình GOOGLE_API_KEY'}")
    st.write(f"- Telegram Token: {'✅ Hoạt Động' if TELEGRAM_TOKEN else '❌ Chưa Cấu Hình TELEGRAM_TOKEN'}")

    st.subheader("🧪 Thử Nghiệm Tóm Tắt PDF Trực Tiếp")
    test_pdf = st.file_uploader("Upload PDF:", type=["pdf"], key="test_pdf_upload")
    if test_pdf:
        if st.button("Tóm Tắt Ngay"):
            t = extract_pdf_text(test_pdf.read())
            if t:
                s = summarize_with_gemini(t)
                st.write(s)
            else:
                st.error("Không thể đọc file PDF.")
