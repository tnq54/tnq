import streamlit as st
import streamlit.components.v1 as components
import time
import os
import threading
import asyncio
import io
import logging
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pypdf import PdfReader
from telegram import Update
from telegram.error import NetworkError
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from huggingface_hub import InferenceClient

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

# Configure Streamlit Page
st.set_page_config(
    page_title="Adobe Photoshop CC Web Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Helper function to read asset files
def read_asset(filename):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# Inject Photoshop CC Dark Workspace CSS
css_content = read_asset("ps_theme.css")
st.markdown(f"""
<style>
{css_content}
</style>
<div class="ps-header">
    <div class="ps-brand">
        <span class="ps-logo">Ps</span>
        <span>Adobe Photoshop CC Web Studio</span>
    </div>
    <div style="color: #888; font-size: 12px;">
        File &nbsp;|&nbsp; Edit &nbsp;|&nbsp; Image &nbsp;|&nbsp; Layer &nbsp;|&nbsp; Type &nbsp;|&nbsp; Select &nbsp;|&nbsp; Filter &nbsp;|&nbsp; 3D &nbsp;|&nbsp; View &nbsp;|&nbsp; Window &nbsp;|&nbsp; Help
    </div>
</div>
""", unsafe_allow_html=True)

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

# Gemini Summarization & Multimodal Photo Analysis
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

def analyze_photo_with_gemini(image_bytes, prompt="Describe and analyze this image in detail for digital editing"):
    if not GOOGLE_API_KEY:
        return "Error: GOOGLE_API_KEY not found."
    if not genai:
        return "Error: google-genai library not installed."

    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        img = Image.open(io.BytesIO(image_bytes))
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=[prompt, img]
        )
        return response.text
    except Exception as e:
        logger.error(f"Gemini Photo Analysis Error: {e}")
        return f"Error analyzing photo: {e}"

# Telegram Bot Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Welcome to Photoshop CC Web Studio Bot!\n"
        "- Send text to chat with AI.\n"
        "- Send photos for AI visual analysis.\n"
        "- Send PDF documents for summarization."
    )

async def chat_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    if not hf_client:
        await update.message.reply_text("Llama 3 / HF Client is not available (HF_TOKEN missing).")
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

async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo_file = await update.message.photo[-1].get_file()
    photo_bytes = await photo_file.download_as_bytearray()
    status_msg = await update.message.reply_text("Analyzing photo with Gemini 1.5 Flash...")
    analysis = analyze_photo_with_gemini(photo_bytes)
    await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status_msg.message_id, text=analysis)

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
    logger.info("Waiting for network initialization...")
    time.sleep(5)

    if not TELEGRAM_TOKEN:
        logger.info("TELEGRAM_TOKEN is missing. Bot runner idle.")
        return

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    while True:
        try:
            application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
            application.add_handler(CommandHandler("start", start))
            application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat_handler))
            application.add_handler(MessageHandler(filters.PHOTO, photo_handler))
            application.add_handler(MessageHandler(filters.Document.PDF, document_handler))
            application.run_polling(stop_signals=None, close_loop=False)
        except NetworkError as e:
            logger.error(f"Network error during polling: {e}. Retrying in 10s...")
            time.sleep(10)
        except Exception as e:
            logger.error(f"Error during polling: {e}")
            time.sleep(10)

# Start background bot thread once
if "bot_thread" not in st.session_state:
    st.session_state.bot_thread = True
    thread = threading.Thread(target=run_bot, daemon=True)
    thread.start()

# --- PHOTOSHOP CC WORKSPACE TABS ---
tabs = st.tabs([
    "🎨 Photoshop CC Suite (Photopea)",
    "🖌️ Photoshop Canvas Tools",
    "✨ Firefly / AI Generator",
    "🔍 Gemini Smart Inspector",
    "⚙️ System Status"
])

# 1. PHOTOSHOP CC MAIN WORKSPACE (PHOTOPEA EMBED)
with tabs[0]:
    photopea_html = read_asset("photopea.html")
    components.html(photopea_html, height=890)

# 2. FULL PHOTOSHOP CANVAS TOOLS
with tabs[1]:
    st.markdown("#### Photoshop CC Toolset (Brush, Eraser, Shapes, Text, Fill, Eyedropper, Opacity)")
    canvas_html = read_asset("canvas.html")
    components.html(canvas_html, height=680)

# 3. AI FIREFLY / IMAGE GENERATOR
with tabs[2]:
    st.markdown("#### Adobe Firefly-style Generative Fill & Text-to-Image")
    prompt = st.text_area("Generative Prompt:", value="A surreal artistic photo of a dragon flying over ancient mountains at sunrise, 8k resolution, photorealistic", height=100)
    model_choice = st.selectbox("Generative Model:", [
        "black-forest-labs/FLUX.1-dev",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "runwayml/stable-diffusion-v1-5"
    ])

    if st.button("🚀 Generate Artwork"):
        if not hf_client:
            st.error("HF_TOKEN is missing in environment.")
        else:
            with st.spinner("Generating artwork via Hugging Face Inference API..."):
                try:
                    generated_image = hf_client.text_to_image(prompt, model=model_choice)
                    st.image(generated_image, caption=f"Generated via {model_choice}", use_column_width=True)
                    buf = io.BytesIO()
                    generated_image.save(buf, format="PNG")
                    st.download_button("💾 Download Layer PNG", data=buf.getvalue(), file_name="firefly_generated.png", mime="image/png")
                except Exception as e:
                    st.error(f"Generation error: {e}")

# 4. GEMINI SMART INSPECTOR
with tabs[3]:
    st.markdown("#### Gemini Multimodal Photoshop Analysis")
    uploaded_file = st.file_uploader("Upload Image Layer...", type=["jpg", "jpeg", "png", "webp"])
    analysis_prompt = st.text_input("Analysis Request:", value="Analyze color palette, lighting, subject placement, and suggest Photoshop CC retouches.")

    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        st.image(image_bytes, caption="Layer Preview", use_column_width=True)

        if st.button("🔍 Run Smart Analysis"):
            with st.spinner("Analyzing image layer with Gemini 1.5 Flash..."):
                analysis_result = analyze_photo_with_gemini(image_bytes, prompt=analysis_prompt)
                st.markdown("##### Analysis & Suggestions:")
                st.write(analysis_result)

# 5. SYSTEM STATUS
with tabs[4]:
    st.markdown("#### Photoshop CC Web System Status")
    st.write(f"- **Hugging Face Inference Token (HF_TOKEN)**: {'✅ Active' if HF_TOKEN else '❌ Missing'}")
    st.write(f"- **Google GenAI API Key (GOOGLE_API_KEY)**: {'✅ Active' if GOOGLE_API_KEY else '❌ Missing'}")
    st.write(f"- **Telegram Bot Token (TELEGRAM_TOKEN)**: {'✅ Active' if TELEGRAM_TOKEN else '❌ Missing'}")
    st.write("- **Port**: 7860 (Hugging Face Spaces Default)")
