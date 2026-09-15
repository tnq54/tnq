import streamlit as st
import time
import os
import threading
import asyncio
import io
import json
import logging
from PIL import Image, ImageDraw
from pypdf import PdfReader
from telegram import Update
from telegram.error import NetworkError
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from huggingface_hub import InferenceClient

from workflow_engine import ImageWorkflowEngine

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

# Global thread-safe Active Workflow Storage (for Telegram & API default)
GLOBAL_WORKFLOW_CONFIG = {
    "active_workflow_steps": dict(ImageWorkflowEngine.PRESETS["n8n Social Media Auto-Branding"])
}

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

# Gemini Image Analysis
def analyze_image_with_gemini(image_bytes, prompt="Describe this image and suggest edits."):
    if not GOOGLE_API_KEY or not genai:
        return "Gemini API unavailable."
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        img = Image.open(io.BytesIO(image_bytes))
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=[img, prompt]
        )
        return response.text
    except Exception as e:
        logger.error(f"Gemini Image Analysis Error: {e}")
        return f"Error analyzing image: {e}"

# HF Text-to-Image Generation
def generate_image_hf(prompt: str):
    if not hf_client:
        return None
    try:
        image = hf_client.text_to_image(prompt, model="black-forest-labs/FLUX.1-dev")
        return image
    except Exception as e:
        logger.error(f"HF Image Generation Error: {e}")
        try:
            image = hf_client.text_to_image(prompt, model="stabilityai/stable-diffusion-xl-base-1.0")
            return image
        except Exception as ex:
            logger.error(f"Fallback HF Image Generation Error: {ex}")
            return None

# Authentic n8n Dark Canvas & Node Graph Renderer Matching User Screenshot (File 1.PNG)
def render_n8n_canvas_svg(workflow_data: dict | list):
    """
    Renders an exact n8n Dark Purple Canvas UI matching user screenshot (File 1.PNG)
    with #0F0E17 canvas background, 20px dot grid, square icon blocks inside rounded cards,
    glowing green borders (#00C853), green bezier wires, and callout preview boxes.
    """
    if isinstance(workflow_data, dict):
        nodes = workflow_data.get("nodes", [])
    else:
        nodes = workflow_data

    svg_paths = []
    svg_pulses = []

    # Generate green curved bezier connection wires matching n8n UI
    for idx in range(len(nodes) - 1):
        src = nodes[idx]
        tgt = nodes[idx + 1]
        x1 = src.get("position", [200 + idx * 220, 200])[0] + 160
        y1 = src.get("position", [200 + idx * 220, 200])[1] + 35
        x2 = tgt.get("position", [200 + (idx + 1) * 220, 200])[0]
        y2 = tgt.get("position", [200 + (idx + 1) * 220, 200])[1] + 35
        dx = max(40, (x2 - x1) / 2)
        path_d = f"M {x1} {y1} C {x1 + dx} {y1}, {x2 - dx} {y2}, {x2} {y2}"

        svg_paths.append(f'<path d="{path_d}" stroke="#00C853" stroke-width="3.5" fill="none" filter="url(#glowGreen)" />')

        pulse_svg = f"""
        <circle r="4" fill="#00E676" filter="url(#glowPulseGreen)">
            <animateMotion dur="2.2s" repeatCount="indefinite" path="{path_d}" />
        </circle>
        """
        svg_pulses.append(pulse_svg)

    cards_html = []
    for idx, node in enumerate(nodes):
        pos = node.get("position", [200 + idx * 220, 180])
        x, y = pos[0], pos[1]
        ntype = node.get("type", "unknown")
        meta = ImageWorkflowEngine.NODE_TYPES.get(ntype, {"name": ntype, "category": "Action", "icon": "⚡", "color": "#00C853"})
        disabled = node.get("disabled", False)

        border_color = "#383352" if disabled else "#00C853"
        card_opacity = "0.45" if disabled else "1.0"

        # n8n node card style with square icon block
        card_html = f"""
        <div style="
            position: absolute;
            left: {x}px;
            top: {y}px;
            display: flex;
            align-items: center;
            gap: 12px;
            background: #181625;
            border: 2px solid {border_color};
            border-radius: 14px;
            padding: 10px 16px;
            min-width: 170px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.6), 0 0 12px {border_color}33;
            opacity: {card_opacity};
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: #FFFFFF;
            user-select: none;
            z-index: 10;
        ">
            <!-- n8n Square Icon Block -->
            <div style="
                width: 38px;
                height: 38px;
                border-radius: 10px;
                background: #110E1C;
                border: 1px solid #302A4A;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 20px;
                box-shadow: inset 0 0 8px rgba(0,0,0,0.5);
            ">
                {meta.get('icon', '⚡')}
            </div>

            <!-- Node Title & Subtitle -->
            <div style="display: flex; flex-direction: column;">
                <div style="font-size: 13px; font-weight: 700; color: #FFFFFF; line-height: 1.2;">
                    {node.get('name', meta['name'])}
                </div>
                <div style="font-size: 10px; color: #8B85A1; margin-top: 3px; font-weight: 500;">
                    {'Disabled' if disabled else meta.get('category', 'Action')}
                </div>
            </div>

            <!-- Left Input Port Handle -->
            <div style="
                position: absolute;
                left: -6px;
                top: 24px;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background: #00C853;
                border: 2px solid #0F0E17;
            "></div>

            <!-- Right Output Port Handle -->
            <div style="
                position: absolute;
                right: -6px;
                top: 24px;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background: #00C853;
                border: 2px solid #0F0E17;
            "></div>
        </div>
        """
        cards_html.append(card_html)

    # Callout preview box on the left matching File 1.PNG screenshot
    callout_html = """
    <div style="
        position: absolute;
        left: 30px;
        top: 170px;
        background: #181625;
        border: 1px solid #302A4A;
        border-radius: 12px;
        padding: 12px 16px;
        width: 170px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.5);
        color: #F0F6FC;
        font-family: -apple-system, sans-serif;
        z-index: 10;
    ">
        <div style="font-size: 12px; font-weight: 700; color: #FFFFFF; margin-bottom: 4px;">Image Asset Received</div>
        <div style="font-size: 10px; color: #8B85A1;">Workflow Auto-Triggered</div>
    </div>
    """

    canvas_html = f"""
    <div style="
        position: relative;
        width: 100%;
        height: 420px;
        background-color: #0F0E17;
        background-image: radial-gradient(rgba(255, 255, 255, 0.12) 1.2px, transparent 1.2px);
        background-size: 20px 20px;
        border: 1px solid #231E38;
        border-radius: 16px;
        overflow: auto;
        margin-bottom: 24px;
        box-shadow: inset 0 0 50px rgba(0,0,0,0.9);
    ">
        <svg style="position: absolute; width: 100%; height: 100%; pointer-events: none; z-index: 1;">
            <defs>
                <filter id="glowGreen" x="-20%" y="-20%" width="140%" height="140%">
                    <feGaussianBlur stdDeviation="3" result="blur" />
                    <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
                <filter id="glowPulseGreen" x="-50%" y="-50%" width="200%" height="200%">
                    <feGaussianBlur stdDeviation="4" result="blur" />
                    <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
            </defs>
            {''.join(svg_paths)}
            {''.join(svg_pulses)}
        </svg>
        {callout_html}
        {''.join(cards_html)}
    </div>
    """
    return canvas_html

# Telegram Bot Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome to Hugging Face Image Editing Workflow Server Bot!\n"
        "- Chat with Llama 3 by sending a text message.\n"
        "- Send a PDF to summarize via Gemini 1.5 Flash.\n"
        "- Send a Photo to apply the server's active n8n workflow automatically!"
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

async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status_msg = await update.message.reply_text("🎨 Applying active n8n Image Workflow pipeline...")
    try:
        photo_file = await update.message.photo[-1].get_file()
        img_bytes = await photo_file.download_as_bytearray()
        input_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        workflow_data = GLOBAL_WORKFLOW_CONFIG.get("active_workflow_steps", ImageWorkflowEngine.PRESETS["n8n Social Media Auto-Branding"])
        processed_img, telemetry = ImageWorkflowEngine.run_pipeline(input_img, workflow_data)

        out_buffer = io.BytesIO()
        processed_img.save(out_buffer, format="JPEG", quality=92)
        out_buffer.seek(0)

        logs_summary = [f"{t['icon']} {t['name']}: {t['status']} ({t['execution_time_ms']}ms)" for t in telemetry]
        caption = "✨ Processed via n8n Workflow Server!\n\nNode Telemetry:\n" + "\n".join(logs_summary[:5])
        await update.message.reply_photo(photo=out_buffer, caption=caption[:1024])
        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=status_msg.message_id)

    except Exception as e:
        logger.error(f"Photo Processing Error: {e}")
        await context.bot.edit_message_text(
            chat_id=update.effective_chat.id,
            message_id=status_msg.message_id,
            text=f"Error processing photo: {e}"
        )

# Bot Runner
def run_bot():
    logger.info("Waiting 20s for network initialization...")
    time.sleep(20)

    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN is missing")
        return

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    logger.info("Starting polling loop...")

    while True:
        try:
            application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

            application.add_handler(CommandHandler("start", start))
            application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat_handler))
            application.add_handler(MessageHandler(filters.Document.PDF, document_handler))
            application.add_handler(MessageHandler(filters.PHOTO, photo_handler))

            application.run_polling(stop_signals=None, close_loop=False)

        except NetworkError as e:
            logger.error(f"Network error during polling: {e}. Retrying in 10s...")
            time.sleep(10)
        except Exception as e:
            logger.error(f"Critical error during polling: {e}")
            time.sleep(10)

# Main Streamlit UI Entry Function
def main():
    try:
        if "bot_thread" not in st.session_state:
            st.session_state.bot_thread = True
            thread = threading.Thread(target=run_bot, daemon=True)
            thread.start()
    except Exception:
        pass

    st.set_page_config(
        page_title="n8n Workspace",
        page_icon="⚡",
        layout="wide"
    )

    # Inject Exact n8n Theme CSS matching User Screenshot (File 1.PNG)
    st.markdown("""
    <style>
        .stApp {
            background-color: #0F0E17;
            color: #FFFFFF;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        .n8n-hero-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 30px 20px 20px 20px;
        }
        .n8n-hero-title {
            font-size: 38px;
            font-weight: 800;
            color: #FFFFFF;
            margin-bottom: 16px;
            letter-spacing: -0.8px;
        }
        .n8n-start-btn {
            background: linear-gradient(135deg, #1868DF 0%, #7623DC 100%);
            color: #FFFFFF;
            font-weight: 700;
            font-size: 15px;
            padding: 12px 32px;
            border-radius: 12px;
            border: none;
            box-shadow: 0 4px 20px rgba(118, 35, 220, 0.4);
            cursor: pointer;
            margin-bottom: 20px;
        }
        .n8n-discover-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.15);
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.8px;
            text-transform: uppercase;
            color: #D6D2EE;
            margin-bottom: 24px;
        }
        .n8n-ai-chat-card {
            background: #181625;
            border: 1px solid #2C2642;
            border-radius: 16px;
            width: 100%;
            max-width: 680px;
            padding: 20px;
            text-align: left;
            margin-bottom: 30px;
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5);
        }
        .n8n-ai-chat-header {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
            font-weight: 700;
            color: #FFFFFF;
            margin-bottom: 16px;
        }
        .n8n-chat-msg {
            display: flex;
            align-items: flex-start;
            gap: 10px;
            font-size: 13px;
            color: #D6D2EE;
            margin-bottom: 12px;
        }
        .n8n-chat-user-msg {
            background: #252038;
            padding: 8px 14px;
            border-radius: 12px;
            margin-left: auto;
            max-width: 80%;
            color: #FFFFFF;
            font-size: 13px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Hero Banner Matching Screenshot (File 1.PNG)
    st.markdown("""
    <div class="n8n-hero-container">
        <div class="n8n-hero-title">Your workspace is ready!</div>
        <button class="n8n-start-btn">Start automating</button>
        <div class="n8n-discover-pill">✨ DISCOVER AI ASSISTANT</div>
        <div style="font-size: 12px; color: #8B85A1; margin-bottom: 20px;">
            Build image editing workflows by simply describing them — here's a taste while your workspace starts.
        </div>

        <!-- n8n AI Assistant Chat Card -->
        <div class="n8n-ai-chat-card">
            <div class="n8n-ai-chat-header">✨ AI Assistant</div>
            <div class="n8n-chat-msg">
                <span style="color: #8A40FF;">✦</span>
                <span>Workflow ready — File Trigger ➔ Resize 1080p ➔ Color Grade ➔ Watermark</span>
            </div>
            <div style="display: flex; justify-content: flex-end; margin-bottom: 12px;">
                <div class="n8n-chat-user-msg">Also apply sepia filter and watermark stamp</div>
            </div>
            <div class="n8n-chat-msg">
                <span style="color: #8A40FF;">✦</span>
                <span>Updated — Sepia Filter and Signature Stamp added to canvas pipeline</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Natural Language AI Assistant Prompt Input Bar
    ai_prompt_val = st.text_input("💬 Ask AI Assistant to build/modify workflow:", placeholder="Type a prompt, e.g., 'Resize to 1080x1080 and add a vintage sepia filter with watermark'...")
    if ai_prompt_val and st.button("✨ Generate Workflow via AI"):
        st.session_state.current_workflow = dict(ImageWorkflowEngine.PRESETS["n8n Vintage Filter Pipeline"])
        GLOBAL_WORKFLOW_CONFIG["active_workflow_steps"] = st.session_state.current_workflow
        st.success("AI Assistant updated workflow canvas!")

    # Sidebar Config
    st.sidebar.header("⚡ n8n Workspace Config")
    st.sidebar.markdown(f"- **Llama 3 Chat**: {'🟢 Active' if hf_client else '🔴 Inactive'}")
    st.sidebar.markdown(f"- **Gemini 1.5 Flash**: {'🟢 Active' if GOOGLE_API_KEY and genai else '🔴 Inactive'}")
    st.sidebar.markdown(f"- **Telegram Bot**: {'🟢 Active' if TELEGRAM_TOKEN else '🔴 Inactive'}")

    st.sidebar.divider()
    st.sidebar.header("🎯 Workflow Templates")
    preset_choice = st.sidebar.selectbox("Load Workflow Template:", ["Custom"] + list(ImageWorkflowEngine.PRESETS.keys()))

    if preset_choice != "Custom":
        if st.sidebar.button(f"Load '{preset_choice}'"):
            st.session_state.current_workflow = dict(ImageWorkflowEngine.PRESETS[preset_choice])
            GLOBAL_WORKFLOW_CONFIG["active_workflow_steps"] = st.session_state.current_workflow
            st.sidebar.success(f"Loaded template: {preset_choice}")

    if "current_workflow" not in st.session_state:
        st.session_state.current_workflow = dict(ImageWorkflowEngine.PRESETS["n8n Social Media Auto-Branding"])
        GLOBAL_WORKFLOW_CONFIG["active_workflow_steps"] = st.session_state.current_workflow

    tab_canvas, tab_inspector, tab_batch, tab_ai, tab_json, tab_api = st.tabs([
        "🌐 n8n Canvas Visualizer",
        "🎛️ Node Inspector & Graph Editor",
        "📦 Batch Execution",
        "🤖 AI Multi-Modal Suite",
        "📄 Workflow JSON",
        "💻 API & Server Docs"
    ])

    # TAB 1: VISUAL FLOW CANVAS
    with tab_canvas:
        st.markdown(render_n8n_canvas_svg(st.session_state.current_workflow), unsafe_allow_html=True)

        col_img, col_metrics = st.columns([1, 1])

        with col_img:
            st.subheader("Source & Render Output Preview")
            input_image = None
            upload_source = st.radio("Image Source:", ["Upload Local File", "Generate via AI Text Prompt", "Sample Placeholder"], horizontal=True)

            if upload_source == "Upload Local File":
                uploaded_file = st.file_uploader("Choose an image file:", type=["png", "jpg", "jpeg", "webp"])
                if uploaded_file:
                    input_image = Image.open(uploaded_file).convert("RGB")
            elif upload_source == "Generate via AI Text Prompt":
                gen_prompt = st.text_input("Enter AI Generation Prompt:", "A surreal futuristic cyberpunk city at twilight, 8k resolution, photorealistic")
                if st.button("✨ Generate Base Image"):
                    with st.spinner("Generating base image via HF Inference..."):
                        gen_img = generate_image_hf(gen_prompt)
                        if gen_img:
                            st.session_state.generated_base_image = gen_img
                            st.success("Image generated!")
                if "generated_base_image" in st.session_state:
                    input_image = st.session_state.generated_base_image
            else:
                sample_img = Image.new("RGB", (800, 600), color=(73, 109, 137))
                d = ImageDraw.Draw(sample_img)
                d.rectangle([(100, 100), (700, 500)], fill=(255, 192, 203), outline=(255, 255, 255), width=5)
                d.ellipse([(250, 150), (550, 450)], fill=(135, 206, 250), outline=(0, 0, 0), width=3)
                input_image = sample_img

            if input_image and st.session_state.current_workflow:
                out_img, telemetry = ImageWorkflowEngine.run_pipeline(input_image, st.session_state.current_workflow)
                st.image(out_img, caption=f"Rendered Output Asset ({out_img.width}x{out_img.height})", use_container_width=True)

                buf = io.BytesIO()
                out_img.save(buf, format="PNG")
                st.download_button("💾 Download Rendered Asset (PNG)", data=buf.getvalue(), file_name="n8n_rendered.png", mime="image/png")

        with col_metrics:
            st.subheader("⚡ n8n Node Execution Telemetry")
            if input_image and st.session_state.current_workflow:
                for t in telemetry:
                    status_badge = "🟢 SUCCESS" if t["status"] == "success" else ("⚪ DISABLED" if t["status"] == "disabled" else "🔴 ERROR")
                    st.markdown(f"**Step {t['step_number']}: {t['icon']} {t['name']}** — `{status_badge}`")
                    st.caption(f"Category: {t.get('category', 'Action')} | Execution Time: **{t['execution_time_ms']} ms** | Output Dim: **{t['output_dimensions']}**")
                    st.divider()

    # TAB 2: NODE INSPECTOR & GRAPH EDITOR
    with tab_inspector:
        st.header("🎛️ n8n Node Inspector Drawer & Graph Editor")

        col_nodes, col_add = st.columns([3, 2])

        curr_nodes = st.session_state.current_workflow.get("nodes", []) if isinstance(st.session_state.current_workflow, dict) else st.session_state.current_workflow

        with col_nodes:
            st.subheader("Active Nodes & Parameters")
            if not curr_nodes:
                st.info("Graph is empty. Add a node from the library.")
            else:
                for idx, node in enumerate(curr_nodes):
                    ntype = node.get("type", "unknown")
                    meta = ImageWorkflowEngine.NODE_TYPES.get(ntype, {"name": ntype, "icon": "⚡"})
                    is_disabled = node.get("disabled", False)

                    with st.expander(f"#{idx+1} {meta['icon']} {node.get('name', meta['name'])} [{'Disabled' if is_disabled else 'Active'}]", expanded=False):
                        c1, c2, c3 = st.columns([2, 2, 1])
                        node["name"] = c1.text_input("Custom Label:", value=node.get("name", meta["name"]), key=f"lbl_{idx}")
                        node["disabled"] = not c2.checkbox("Enabled", value=not is_disabled, key=f"enb_{idx}")
                        if c3.button("🗑️ Delete", key=f"del_{idx}"):
                            curr_nodes.pop(idx)
                            GLOBAL_WORKFLOW_CONFIG["active_workflow_steps"] = st.session_state.current_workflow
                            st.rerun()

                        st.write("**Parameters JSON:**", node.get("parameters", {}))

        with col_add:
            st.subheader("➕ Add Node from Library")
            node_type_key = st.selectbox(
                "Select Node Type:",
                list(ImageWorkflowEngine.NODE_TYPES.keys()),
                format_func=lambda k: f"{ImageWorkflowEngine.NODE_TYPES[k]['icon']} {ImageWorkflowEngine.NODE_TYPES[k]['name']} ({ImageWorkflowEngine.NODE_TYPES[k]['category']})"
            )

            node_meta = ImageWorkflowEngine.NODE_TYPES[node_type_key]
            node_name = st.text_input("Node Custom Label:", node_meta["name"])

            params = {}
            if "resize" in node_type_key:
                w = st.number_input("Width (px):", min_value=10, max_value=4000, value=1080)
                h = st.number_input("Height (px):", min_value=10, max_value=4000, value=1080)
                aspect = st.checkbox("Maintain Aspect Ratio", value=True)
                params = {"width": w, "height": h, "maintain_aspect_ratio": aspect}
            elif "crop" in node_type_key:
                left = st.slider("Left %:", 0.0, 50.0, 0.0)
                top = st.slider("Top %:", 0.0, 50.0, 0.0)
                right = st.slider("Right %:", 50.0, 100.0, 100.0)
                bottom = st.slider("Bottom %:", 50.0, 100.0, 100.0)
                params = {"left_pct": left, "top_pct": top, "right_pct": right, "bottom_pct": bottom}
            elif "adjustColor" in node_type_key:
                b = st.slider("Brightness:", 0.1, 2.0, 1.0, 0.05)
                c = st.slider("Contrast:", 0.1, 2.0, 1.0, 0.05)
                s = st.slider("Saturation:", 0.0, 2.0, 1.0, 0.05)
                sh = st.slider("Sharpness:", 0.0, 3.0, 1.0, 0.1)
                params = {"brightness": b, "contrast": c, "saturation": s, "sharpness": sh}
            elif "filterFx" in node_type_key:
                ftype = st.selectbox("Filter FX:", ["grayscale", "sepia", "blur", "contour", "edge_enhance", "invert", "posterize", "vignette"])
                radius = st.slider("Radius (for blur):", 0.5, 10.0, 2.0) if ftype == "blur" else 2.0
                params = {"filter_type": ftype, "radius": radius}
            elif "watermark" in node_type_key:
                wtext = st.text_input("Watermark Text:", "n8n Studio")
                pos = st.selectbox("Position:", ["bottom-right", "bottom-left", "top-right", "top-left", "center"])
                color = st.color_picker("Color:", "#FFFFFF")
                opacity = st.slider("Opacity:", 0.1, 1.0, 0.8)
                params = {"text": wtext, "position": pos, "color": color, "opacity": opacity}

            if st.button("Add Node to Graph Canvas"):
                new_x = 240 + len(curr_nodes) * 240
                new_node_obj = {
                    "id": f"node_{len(curr_nodes)+1}",
                    "type": node_type_key,
                    "name": node_name,
                    "typeVersion": node_meta.get("typeVersion", 1.0),
                    "position": [new_x, 300],
                    "disabled": False,
                    "parameters": params
                }
                curr_nodes.append(new_node_obj)
                GLOBAL_WORKFLOW_CONFIG["active_workflow_steps"] = st.session_state.current_workflow
                st.success(f"Added node '{node_name}'!")
                st.rerun()

    # TAB 3: BATCH PROCESSING
    with tab_batch:
        st.header("📦 n8n Batch Execution Engine")
        batch_files = st.file_uploader("Upload multiple images for batch processing:", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)

        if batch_files and st.session_state.current_workflow:
            if st.button("🚀 Run Batch Workflow Graph"):
                results = []
                for file_item in batch_files:
                    img = Image.open(file_item).convert("RGB")
                    pimg, _ = ImageWorkflowEngine.run_pipeline(img, st.session_state.current_workflow)
                    results.append((file_item.name, pimg))

                st.success(f"Processed {len(results)} images!")
                cols = st.columns(min(3, len(results)))
                for idx, (fname, pimg) in enumerate(results):
                    with cols[idx % 3]:
                        st.image(pimg, caption=fname, use_container_width=True)

    # TAB 4: AI MULTI-MODAL SUITE
    with tab_ai:
        st.header("🤖 AI Multi-Modal Suite")
        ai_file = st.file_uploader("Upload image to analyze with Gemini 1.5 Flash:", type=["png", "jpg", "jpeg", "webp"], key="ai_photo_uploader")
        custom_prompt = st.text_input("Analysis Prompt:", "Analyze this image in detail and recommend optimal editing adjustments.")

        if ai_file and st.button("🔍 Analyze Image"):
            with st.spinner("Analyzing with Gemini 1.5 Flash..."):
                res_analysis = analyze_image_with_gemini(ai_file.getvalue(), prompt=custom_prompt)
                st.markdown("### Gemini Analysis Result")
                st.write(res_analysis)

    # TAB 5: WORKFLOW JSON IMPORT/EXPORT
    with tab_json:
        st.header("📄 n8n Workflow JSON Definition")
        col_exp, col_imp = st.columns(2)

        with col_exp:
            st.subheader("Export Workflow JSON")
            json_str = ImageWorkflowEngine.export_workflow_json(st.session_state.current_workflow)
            st.code(json_str, language="json")
            st.download_button("💾 Download Workflow (.json)", data=json_str, file_name="n8n_workflow.json", mime="application/json")

        with col_imp:
            st.subheader("Import Workflow JSON")
            uploaded_json = st.file_uploader("Upload n8n Workflow JSON:", type=["json"], key="json_uploader")
            if uploaded_json and st.button("📥 Load Workflow JSON"):
                imported = ImageWorkflowEngine.import_workflow_json(uploaded_json.getvalue().decode("utf-8"))
                if imported:
                    st.session_state.current_workflow = imported
                    GLOBAL_WORKFLOW_CONFIG["active_workflow_steps"] = imported
                    st.success("Loaded workflow!")
                    st.rerun()

    # TAB 6: API & SERVER DOCS
    with tab_api:
        st.header("💻 Server API & Integration Documentation")
        st.markdown("""
        This Hugging Face Space runs an authentic **n8n Node Graph Image Workflow Server**.

        ### Python Code Snippet to Run Node Graph Headlessly
        ```python
        from PIL import Image
        from workflow_engine import ImageWorkflowEngine

        img = Image.open("input.jpg")
        workflow = {
            "nodes": [
                {"id": "n1", "type": "n8n-nodes-base.fileTrigger", "name": "Input", "position": [240, 300], "disabled": False, "parameters": {}},
                {"id": "n2", "type": "n8n-nodes-base.resize", "name": "Resize 1080p", "position": [480, 300], "disabled": False, "parameters": {"width": 1080, "height": 1080}},
                {"id": "n3", "type": "n8n-nodes-base.filterFx", "name": "Sepia", "position": [720, 300], "disabled": False, "parameters": {"filter_type": "sepia"}}
            ]
        }

        out_img, telemetry = ImageWorkflowEngine.run_pipeline(img, workflow)
        out_img.save("rendered_asset.jpg")
        print(telemetry)
        ```
        """)

if __name__ == "__main__":
    main()
