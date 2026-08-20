import os
import sys

# Direct execution relaunch guard: if executed directly with python app.py, relaunch via streamlit run
try:
    import streamlit as st
    if not st.runtime.exists():
        os.execv(sys.executable, [sys.executable, "-m", "streamlit", "run", __file__] + sys.argv[1:])
except Exception:
    pass

import time
import threading
import asyncio
import io
import logging
import subprocess
from pypdf import PdfReader
from telegram import Update
from telegram.error import NetworkError
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from huggingface_hub import InferenceClient

# Call set_page_config as the first Streamlit command
st.set_page_config(page_title="VBot1 System & LoRA Studio", layout="wide")

# Setup logging filter to suppress missing ScriptRunContext warnings
class ScriptRunContextFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return "missing ScriptRunContext" not in msg and "Session state does not function" not in msg

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

context_filter = ScriptRunContextFilter()
logging.getLogger("streamlit").addFilter(context_filter)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").addFilter(context_filter)

# Try importing Google GenAI
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

# Load Environment Variables
HF_TOKEN = os.environ.get("HF_TOKEN")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

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

# Gemini Summarization
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

# Gemini Photo/Image Analysis
def analyze_photo_with_gemini(image_bytes, mime_type="image/jpeg", prompt="Describe this image in detail and highlight key features."):
    if not GOOGLE_API_KEY:
        return "Error: GOOGLE_API_KEY not found."
    if not genai:
        return "Error: google-genai library not installed."

    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        if types and hasattr(types, "Part"):
            part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            contents = [prompt, part]
        else:
            contents = [prompt, {"mime_type": mime_type, "data": image_bytes}]

        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=contents
        )
        return response.text
    except Exception as e:
        logger.error(f"Gemini Photo Error: {e}")
        return f"Error analyzing photo: {e}"

# Telegram Bot Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Welcome to VBot1!\n"
        "- Chat with me (Llama 3).\n"
        "- Send a PDF to summarize (Gemini 1.5 Flash).\n"
        "- Send a Photo to analyze (Gemini 1.5 Flash)."
    )

async def chat_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    if not hf_client:
        await update.message.reply_text("Llama 3 is not available (HF_TOKEN missing).")
        return

    status_msg = await update.message.reply_text("Thinking...")
    try:
        messages = [{"role": "user", "content": user_text}]
        # Llama 3 via HF Inference API
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

        # Split long messages
        if len(summary) > 4000:
            for i in range(0, len(summary), 4000):
                await update.message.reply_text(summary[i:i+4000])
        else:
            await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status_msg.message_id, text=summary)

    except Exception as e:
        logger.error(f"Document Error: {e}")
        await update.message.reply_text(f"Error processing document: {e}")

async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.photo:
        return

    status_msg = await update.message.reply_text("Downloading photo...")
    try:
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        image_bytes = await file.download_as_bytearray()

        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status_msg.message_id, text="Analyzing photo with Gemini 1.5 Flash...")
        caption = update.message.caption or "Describe this image in detail and highlight key features."
        analysis = analyze_photo_with_gemini(image_bytes, mime_type="image/jpeg", prompt=caption)

        if len(analysis) > 4000:
            for i in range(0, len(analysis), 4000):
                await update.message.reply_text(analysis[i:i+4000])
        else:
            await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status_msg.message_id, text=analysis)

    except Exception as e:
        logger.error(f"Photo Error: {e}")
        await update.message.reply_text(f"Error processing photo: {e}")

# Bot Runner
def run_bot():
    # FIX: Network Error - Wait for network to be ready
    logger.info("Waiting 20s for network initialization...")
    time.sleep(20)

    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN is missing")
        return

    # Setup Loop for Thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    logger.info("Starting polling loop...")

    while True:
        try:
            # Rebuild application on every iteration to ensure fresh httpx client
            application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

            application.add_handler(CommandHandler("start", start))
            application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat_handler))
            application.add_handler(MessageHandler(filters.Document.PDF, document_handler))
            application.add_handler(MessageHandler(filters.PHOTO, photo_handler))

            # FIX: System Error - Invalid file descriptor
            application.run_polling(stop_signals=None, close_loop=False)

        except NetworkError as e:
            logger.error(f"Network error during polling: {e}. Retrying in 10s...")
            time.sleep(10)
        except Exception as e:
            logger.error(f"Critical error during polling: {e}")
            # Wait a bit before retrying to avoid rapid crash loops
            time.sleep(10)
        finally:
            # Clean up if necessary, though ApplicationBuilder makes a new one
            pass

# Background Thread
if "bot_thread" not in st.session_state:
    st.session_state.bot_thread = True
    thread = threading.Thread(target=run_bot, daemon=True)
    thread.start()

# Streamlit UI
st.title("VBot1 System & LoRA Fine-Tuning Studio")

tab1, tab2, tab3 = st.tabs(["🤖 Bot Dashboard", "🖼️ Photo Analysis", "🎛️ LoRA Training Studio"])

with tab1:
    st.header("Bot Operations")
    st.write("Status: Bot is running in background thread.")
    st.subheader("Configuration")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Llama 3 (Inference)", "Active" if hf_client else "Inactive")
    with col2:
        st.metric("Gemini 1.5 Flash (Multimodal)", "Active" if GOOGLE_API_KEY else "Inactive")

with tab2:
    st.header("Photo & Image Analysis (Gemini 1.5 Flash)")
    st.markdown("Upload photos or images to get detailed analysis and descriptions from Gemini 1.5 Flash.")
    uploaded_photo = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png", "webp"])
    prompt_text = st.text_input("Analysis Prompt", value="Describe this image in detail and highlight key features.")

    if uploaded_photo is not None:
        st.image(uploaded_photo, caption="Uploaded Image", use_container_width=True)
        if st.button("🔍 Analyze Photo"):
            with st.spinner("Analyzing photo with Gemini 1.5 Flash..."):
                file_bytes = uploaded_photo.getvalue()
                mime_type = uploaded_photo.type or "image/jpeg"
                res = analyze_photo_with_gemini(file_bytes, mime_type=mime_type, prompt=prompt_text)
                st.subheader("Analysis Result")
                st.write(res)

with tab3:
    st.header("LoRA Fine-Tuning Studio")
    st.markdown("Configure and trigger LoRA adapter training for Hugging Face LLMs.")

    col1, col2 = st.columns(2)

    with col1:
        model_presets = [
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "Qwen/Qwen2.5-7B-Instruct",
            "google/gemma-2-9b-it",
            "Custom..."
        ]
        selected_model_preset = st.selectbox("Base Model Preset", options=model_presets, index=0)
        if selected_model_preset == "Custom...":
            base_model = st.text_input("Custom Base Model Path/ID", value="meta-llama/Meta-Llama-3-8B-Instruct")
        else:
            base_model = selected_model_preset

        dataset_name = st.text_input("Dataset Name or Path", value="timdettmers/openassistant-guanaco")
        dataset_text_field = st.text_input("Dataset Text Field Column", value="text")
        output_dir = st.text_input("Output Directory", value="./lora_output")

    with col2:
        lora_r = st.number_input("LoRA Rank (r)", min_value=1, max_value=256, value=16)
        lora_alpha = st.number_input("LoRA Alpha", min_value=1, max_value=512, value=32)
        lora_dropout = st.slider("LoRA Dropout", min_value=0.0, max_value=0.5, value=0.05, step=0.01)
        learning_rate = st.select_slider("Learning Rate", options=[1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3], value=2e-4)
        num_epochs = st.number_input("Epochs", min_value=1, max_value=50, value=1)
        target_modules = st.text_input("Target Modules", value="q_proj,v_proj,k_proj,o_proj")

    st.subheader("Push to Hugging Face Hub (Optional)")
    push_to_hub = st.checkbox("Push trained adapter to HF Hub")
    hub_model_id = st.text_input("HF Hub Repository ID (e.g. username/my-lora-adapter)", value="")

    cmd = [
        "python3", "train_lora.py",
        f"--base_model={base_model}",
        f"--dataset_name={dataset_name}",
        f"--dataset_text_field={dataset_text_field}",
        f"--lora_r={lora_r}",
        f"--lora_alpha={lora_alpha}",
        f"--lora_dropout={lora_dropout}",
        f"--target_modules={target_modules}",
        f"--learning_rate={learning_rate}",
        f"--num_epochs={num_epochs}",
        f"--output_dir={output_dir}"
    ]

    if push_to_hub and hub_model_id:
        cmd.extend(["--push_to_hub", f"--hub_model_id={hub_model_id}"])

    generated_command = " ".join(cmd)
    st.subheader("Generated CLI Command")
    st.code(generated_command, language="bash")

    if st.button("🚀 Start LoRA Training"):
        st.info("Starting LoRA training subprocess...")
        log_area = st.empty()
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            logs = ""
            for line in iter(process.stdout.readline, ''):
                logs += line
                log_area.code(logs[-2000:], language="text")
            process.stdout.close()
            return_code = process.wait()
            if return_code == 0:
                st.success("LoRA training finished successfully!")
            else:
                st.error(f"Training failed with return code {return_code}")
        except Exception as e:
            st.error(f"Error executing training: {e}")
