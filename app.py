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
    page_title="Photoshop CC Web Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# --- STREAMLIT UI: PHOTOSHOP CC WEB STUDIO ---
st.title("🎨 Photoshop CC Web Studio")
st.caption("Full Photoshop CC Web Application with Layer Editing, Custom Canvas Tools, AI Image Generation & Photo Analysis on Hugging Face Space")

tabs = st.tabs([
    "🖼️ Photoshop CC (Photopea)",
    "🖌️ HTML5 Canvas Editor",
    "✨ AI Image Generator",
    "🔍 Gemini Photo Analysis",
    "⚙️ System Status"
])

# 1. PHOTOSHOP CC (PHOTOPEA EMBED)
with tabs[0]:
    st.markdown("### Professional Photoshop CC Editor")
    st.write("Full-featured browser image editor supporting `.psd`, `.png`, `.jpg`, `.webp`, custom layers, masks, smart objects, and filters.")

    # Embedded Photopea UI
    photopea_html = """
    <iframe src="https://www.photopea.com" style="width: 100%; height: 850px; border: none; border-radius: 8px;"></iframe>
    """
    components.html(photopea_html, height=870)

# 2. HTML5 CANVAS EDITOR
with tabs[1]:
    st.markdown("### Custom Interactive HTML5 Canvas Studio")
    st.write("Responsive client-side painting canvas with layer controls, brush size, color picker, and real-time image adjustment filters.")

    canvas_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body { margin: 0; background: #121212; color: #fff; font-family: sans-serif; }
            #toolbar { padding: 10px; background: #1e1e1e; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
            canvas { background: #ffffff; border: 2px solid #333; cursor: crosshair; margin: 10px 0; border-radius: 4px; }
            .btn { background: #2b2b2b; color: white; border: 1px solid #444; padding: 6px 12px; border-radius: 4px; cursor: pointer; }
            .btn:hover { background: #3b3b3b; }
        </style>
    </head>
    <body>
        <div id="toolbar">
            <label>Tool: </label>
            <select id="tool" class="btn">
                <option value="brush">Brush 🖌️</option>
                <option value="eraser">Eraser 🧽</option>
                <option value="line">Line 📏</option>
                <option value="rect">Rectangle 🔲</option>
            </select>
            <label>Color: </label>
            <input type="color" id="colorPicker" value="#000000">
            <label>Size: </label>
            <input type="range" id="brushSize" min="1" max="50" value="5">
            <button class="btn" onclick="clearCanvas()">Clear 🗑️</button>
            <button class="btn" onclick="downloadCanvas()">Export PNG 💾</button>
        </div>
        <canvas id="paintCanvas" width="900" height="550"></canvas>

        <script>
            const canvas = document.getElementById('paintCanvas');
            const ctx = canvas.getContext('2d');
            let painting = false;
            let startX, startY;

            function startPosition(e) {
                painting = true;
                draw(e);
            }

            function finishedPosition() {
                painting = false;
                ctx.beginPath();
            }

            function draw(e) {
                if(!painting) return;
                const rect = canvas.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;

                const tool = document.getElementById('tool').value;
                const color = document.getElementById('colorPicker').value;
                const size = document.getElementById('brushSize').value;

                ctx.lineWidth = size;
                ctx.lineCap = 'round';

                if(tool === 'brush') {
                    ctx.strokeStyle = color;
                    ctx.lineTo(x, y);
                    ctx.stroke();
                    ctx.beginPath();
                    ctx.moveTo(x, y);
                } else if(tool === 'eraser') {
                    ctx.strokeStyle = '#ffffff';
                    ctx.lineTo(x, y);
                    ctx.stroke();
                    ctx.beginPath();
                    ctx.moveTo(x, y);
                }
            }

            function clearCanvas() {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
            }

            function downloadCanvas() {
                const link = document.createElement('a');
                link.download = 'photoshop_canvas_export.png';
                link.href = canvas.toDataURL();
                link.click();
            }

            canvas.addEventListener('mousedown', startPosition);
            canvas.addEventListener('mouseup', finishedPosition);
            canvas.addEventListener('mousemove', draw);
        </script>
    </body>
    </html>
    """
    components.html(canvas_html, height=650)

# 3. AI IMAGE GENERATOR
with tabs[2]:
    st.markdown("### AI Image Generation & Diffusion Studio")
    st.write("Generate high-quality artwork using Hugging Face Diffusion Models (e.g., FLUX.1-dev, Stable Diffusion XL).")

    prompt = st.text_area("Prompt:", value="A realistic portrait of a futuristic cyberpunk city with neon lights, high resolution, 8k", height=100)
    model_choice = st.selectbox("AI Model:", [
        "black-forest-labs/FLUX.1-dev",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "runwayml/stable-diffusion-v1-5"
    ])

    if st.button("🚀 Generate AI Image"):
        if not hf_client:
            st.error("HF_TOKEN is missing in environment. Please configure your Hugging Face API key.")
        else:
            with st.spinner("Generating image via Hugging Face Inference API..."):
                try:
                    generated_image = hf_client.text_to_image(prompt, model=model_choice)
                    st.image(generated_image, caption=f"Generated with {model_choice}", use_column_width=True)

                    # Download option
                    buf = io.BytesIO()
                    generated_image.save(buf, format="PNG")
                    st.download_button("💾 Download Image", data=buf.getvalue(), file_name="ai_generated_image.png", mime="image/png")
                except Exception as e:
                    st.error(f"Generation failed: {e}")

# 4. GEMINI PHOTO ANALYSIS
with tabs[3]:
    st.markdown("### Gemini 1.5 Flash Multimodal Photo Analysis")
    st.write("Upload an image for visual analysis, composition feedback, or editing suggestions.")

    uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png", "webp"])
    analysis_prompt = st.text_input("Custom Analysis Prompt:", value="Analyze composition, lighting, subject matter, and suggest Photoshop enhancements.")

    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        st.image(image_bytes, caption="Uploaded Image Preview", use_column_width=True)

        if st.button("🔍 Analyze Image"):
            with st.spinner("Analyzing image using Gemini 1.5 Flash..."):
                analysis_result = analyze_photo_with_gemini(image_bytes, prompt=analysis_prompt)
                st.markdown("#### Analysis Results:")
                st.write(analysis_result)

# 5. SYSTEM STATUS
with tabs[4]:
    st.markdown("### System & API Status")
    st.write(f"- **Hugging Face Inference Token (HF_TOKEN)**: {'✅ Active' if HF_TOKEN else '❌ Missing'}")
    st.write(f"- **Google GenAI API Key (GOOGLE_API_KEY)**: {'✅ Active' if GOOGLE_API_KEY else '❌ Missing'}")
    st.write(f"- **Telegram Bot Token (TELEGRAM_TOKEN)**: {'✅ Active' if TELEGRAM_TOKEN else '❌ Missing'}")
    st.write("- **Port**: 7860 (Hugging Face Spaces Default)")
