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

def generate_image_caption_with_gemini(image_bytes, trigger_word="", custom_prompt=""):
    if not custom_prompt:
        custom_prompt = "Provide a detailed 1-sentence prompt caption describing the subject, style, lighting, and background of this image for LoRA diffusion model training."

    if not GOOGLE_API_KEY or not genai:
        caption = "a high quality detailed photo"
    else:
        try:
            client = genai.Client(api_key=GOOGLE_API_KEY)
            img = Image.open(io.BytesIO(image_bytes))
            response = client.models.generate_content(
                model='gemini-1.5-flash',
                contents=[custom_prompt, img]
            )
            caption = response.text.strip()
        except Exception as e:
            logger.error(f"Gemini Image Captioning Error: {e}")
            caption = "a high quality detailed photo"

    if trigger_word and not caption.startswith(trigger_word):
        caption = f"{trigger_word}, {caption}"
    return caption

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
    "🖼️ 4. Xem Ảnh Sample Đã Train",
    "📦 5. Export & Push HF Hub",
    "🤖 6. Telegram Bot & System"
])

# TAB 1: DATASET MANAGEMENT
with main_tabs[0]:
    st.header("📸 Quản Lý Dataset Ảnh & Tự Động Tạo Caption Prompt")
    st.write("Tải lên ảnh huấn luyện (`.png`, `.jpg`, `.jpeg`, `.webp`). Tùy chỉnh Trigger Word (Từ Khóa Trigger) và Prompt để Gemini 1.5 tạo caption tự động.")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        trigger_word = st.text_input("Từ khóa Trigger (Trigger Word / Prompt Prefix):", value="sks photo", help="Mẫu từ khóa sẽ tự động được chèn vào đầu mọi tệp caption ảnh (ví dụ: sks style, ohwx man)")
    with col_p2:
        custom_gemini_prompt = st.text_area(
            "Prompt Hướng Dẫn Cho Gemini Auto-Caption:",
            value="Provide a detailed 1-sentence prompt caption describing the subject, style, lighting, and background of this image for LoRA diffusion model training.",
            height=68
        )

    uploaded_files = st.file_uploader(
        "Chọn các tệp ảnh huấn luyện:",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("💾 Lưu Dataset & Tạo Captions Với Prompt", type="primary"):
            progress_bar = st.progress(0)
            for idx, file in enumerate(uploaded_files):
                img_bytes = file.read()
                filename_base = os.path.splitext(file.name)[0]
                img_path = os.path.join(DATASET_DIR, f"{filename_base}.png")

                # Save Image
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img.save(img_path)

                # Generate Caption
                caption = generate_image_caption_with_gemini(img_bytes, trigger_word=trigger_word, custom_prompt=custom_gemini_prompt)
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
            if st.button("⚡ Tự Động Thêm Trigger Word Vào Tất Cả Caption Hiện Có"):
                count_updated = 0
                for img_f in image_files:
                    txt_path = os.path.join(DATASET_DIR, os.path.splitext(img_f)[0] + ".txt")
                    if os.path.exists(txt_path):
                        with open(txt_path, "r", encoding="utf-8") as f:
                            content = f.read().strip()
                        if trigger_word and not content.startswith(trigger_word):
                            content = f"{trigger_word}, {content}"
                            with open(txt_path, "w", encoding="utf-8") as f:
                                f.write(content)
                            count_updated += 1
                st.success(f"Đã cập nhật Trigger Word '{trigger_word}' cho {count_updated} tệp caption!")

            cols = st.columns(3)
            for idx, img_f in enumerate(image_files[:6]):
                with cols[idx % 3]:
                    img_path = os.path.join(DATASET_DIR, img_f)
                    txt_path = os.path.join(DATASET_DIR, os.path.splitext(img_f)[0] + ".txt")

                    st.image(img_path, use_column_width=True)
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
    st.header("⚙️ Thiết Lập Cấu Hình LoRA & Siêu Tham Số (Advanced Training Settings)")

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

    st.checkbox("resize_control", value=True)

    c_ep1, c_ep2, c_ep3 = st.columns(3)
    with c_ep1:
        num_repeats = st.number_input("num_repeats", value=20, min_value=1)
    with c_ep2:
        max_train_epochs = st.number_input("max_train_epochs", value=4, min_value=1)
    with c_ep3:
        max_train_steps = st.number_input("max_train_steps", value=0, min_value=0)

    c_sav1, c_sav2, c_sav3 = st.columns(3)
    with c_sav1:
        save_every_n_epochs = st.number_input("save_every_n_epochs", value=1, min_value=1)
    with c_sav2:
        save_last_n_epochs = st.number_input("save_last_n_epochs", value=0, min_value=0)
    with c_sav3:
        save_every_n_steps = st.number_input("save_every_n_steps", value=0, min_value=0)

    st.markdown("---")
    st.markdown("### 🔹 Điều chỉnh Learning Rate")
    st.info("💡 **Nên đặt lr = 1e-4 với lora, 1e-6 với finetuning**")

    c_lr1, c_lr2 = st.columns(2)
    with c_lr1:
        learning_rate = st.number_input("learning_rate", value=1e-4, format="%.6f")
    with c_lr2:
        optimizer_type = st.selectbox("optimizer_type", ["adamw8bit", "adamw", "adafactor", "lion", "prodigy"], index=0)

    st.caption("💡 **Mặc định finetuning sẽ dùng adafactor**")

    c_sch1, c_sch2, c_sch3 = st.columns(3)
    with c_sch1:
        lr_scheduler = st.selectbox("lr_scheduler", ["constant", "cosine", "linear", "cosine_with_restarts", "polynomial"], index=0)
    with c_sch2:
        lr_poly_power = st.number_input("lr_poly_power", value=0)
    with c_sch3:
        lr_warmup_steps = st.number_input("lr_warmup_steps", value=10)

    c_sch4, c_dim, c_alpha = st.columns(3)
    with c_sch4:
        lr_restarts_num_cycles = st.number_input("lr_restarts_num_cycles", value=4)
    with c_dim:
        network_dim = st.number_input("network_dim (lora_r)", value=32, min_value=1)
    with c_alpha:
        network_alpha = st.number_input("network_alpha (lora_alpha)", value=16, min_value=1)

    st.markdown("---")
    st.info("💡 **Phương pháp timestep_sampling: shift - Cân bằng, logsnr - Tổng thể, sigma - Chi tiết**")
    timestep_sampling = st.selectbox("timestep_sampling", ["None", "shift", "logsnr", "sigma"], index=0)

    st.info("💡 **Điều chỉnh khoảng timestep: Thấp (vd 0-200) - Train chi tiết (face, detail), Cao (400-1000) - Train tổng thể (màu, ánh sáng, phong cách)**")

    c_ts1, c_ts2 = st.columns(2)
    with c_ts1:
        min_timestep = st.number_input("min_timestep", value=0, min_value=0)
    with c_ts2:
        max_timestep = st.number_input("max_timestep", value=1000, min_value=0)

    preserve_distribution_shape = st.checkbox("preserve_distribution_shape", value=False)

    st.markdown("---")
    st.subheader("🛠️ Cấu Hình Bổ Sung (Resolution, Precision & Bucketing)")
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        resolution = st.select_slider("Resolution (px):", options=[512, 768, 1024, 1280], value=1024)
        unet_lr = st.number_input("UNet LR:", value=1e-4, format="%.6f")
        text_encoder_lr = st.number_input("Text Encoder LR:", value=5e-5, format="%.6f")
        mixed_precision = st.selectbox("Mixed Precision:", ["fp16", "bf16", "no"], index=0)
    with col_opt2:
        gradient_checkpointing = st.checkbox("Gradient Checkpointing", value=True)
        use_4bit = st.checkbox("4-bit QLoRA Quantization", value=False)
        use_safetensors = st.checkbox("SafeTensors (.safetensors)", value=True)
        enable_bucket = st.checkbox("Aspect Ratio Bucketing (ARB)", value=True)

    # Save Config JSON
    training_config = {
        "base_model": base_model,
        "dataset_dir": DATASET_DIR,
        "output_dir": OUTPUT_DIR,
        "config_save_dir": CONFIG_DIR,
        "num_repeats": num_repeats,
        "max_train_epochs": max_train_epochs,
        "max_train_steps": max_train_steps,
        "save_every_n_epochs": save_every_n_epochs,
        "save_last_n_epochs": save_last_n_epochs,
        "save_every_n_steps": save_every_n_steps,
        "learning_rate": learning_rate,
        "unet_lr": unet_lr,
        "text_encoder_lr": text_encoder_lr,
        "optimizer_type": optimizer_type,
        "lr_scheduler": lr_scheduler,
        "lr_poly_power": lr_poly_power,
        "lr_warmup_steps": lr_warmup_steps,
        "lr_restarts_num_cycles": lr_restarts_num_cycles,
        "network_dim": network_dim,
        "network_alpha": network_alpha,
        "resolution": resolution,
        "timestep_sampling": timestep_sampling,
        "min_timestep": min_timestep,
        "max_timestep": max_timestep,
        "preserve_distribution_shape": preserve_distribution_shape,
        "mixed_precision": mixed_precision,
        "gradient_checkpointing": gradient_checkpointing,
        "use_4bit": use_4bit,
        "use_safetensors": use_safetensors,
        "enable_bucket": enable_bucket
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
            "--num_repeats", str(num_repeats),
            "--max_train_epochs", str(max_train_epochs),
            "--max_train_steps", str(max_train_steps),
            "--save_every_n_epochs", str(save_every_n_epochs),
            "--save_last_n_epochs", str(save_last_n_epochs),
            "--save_every_n_steps", str(save_every_n_steps),
            "--learning_rate", str(learning_rate),
            "--unet_lr", str(unet_lr),
            "--text_encoder_lr", str(text_encoder_lr),
            "--optimizer_type", optimizer_type,
            "--lr_scheduler", lr_scheduler,
            "--lr_poly_power", str(lr_poly_power),
            "--lr_warmup_steps", str(lr_warmup_steps),
            "--lr_restarts_num_cycles", str(lr_restarts_num_cycles),
            "--network_dim", str(network_dim),
            "--network_alpha", str(network_alpha),
            "--resolution", str(resolution),
            "--timestep_sampling", timestep_sampling,
            "--min_timestep", str(min_timestep),
            "--max_timestep", str(max_timestep),
            "--mixed_precision", mixed_precision
        ]

        if preserve_distribution_shape:
            cmd.append("--preserve_distribution_shape")
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

# TAB 4: VIEW TRAINED SAMPLE IMAGES
with main_tabs[3]:
    st.header("🖼️ Thư Viện Ảnh Sample Sau Khi Train (Trained Preview Gallery)")
    st.write("Xem trực tiếp các bức ảnh sample được sinh ra trong quá trình huấn luyện LoRA từ thư mục `./output`.")

    if os.path.exists(OUTPUT_DIR):
        out_files = os.listdir(OUTPUT_DIR)
        sample_images = [f for f in out_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

        if sample_images:
            st.success(f"Tìm thấy {len(sample_images)} ảnh sample huấn luyện!")
            gallery_cols = st.columns(3)
            for idx, sample_f in enumerate(sorted(sample_images)):
                sample_p = os.path.join(OUTPUT_DIR, sample_f)
                with gallery_cols[idx % 3]:
                    st.image(sample_p, caption=f"Sample: {sample_f}", use_column_width=True)
                    with open(sample_p, "rb") as sf:
                        st.download_button(
                            label=f"📥 Tải Ảnh {sample_f}",
                            data=sf.read(),
                            file_name=sample_f,
                            mime="image/png",
                            key=f"dl_sample_{sample_f}"
                        )
        else:
            st.warning("Chưa có ảnh sample nào trong `./output`. Hãy chạy Huấn Luyện ở Tab 3 để tạo ảnh sample!")

# TAB 5: EXPORT & PUSH TO HF HUB
with main_tabs[4]:
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

# TAB 6: TELEGRAM BOT & SYSTEM STATUS
with main_tabs[5]:
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
