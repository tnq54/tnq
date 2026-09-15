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
    "active_workflow_steps": list(ImageWorkflowEngine.PRESETS["n8n Social Media Flow"])
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

# n8n Node Canvas Flowchart Renderer
def render_n8n_flowchart_html(nodes: list[dict]):
    """
    Renders an interactive n8n-style visual graph canvas showing node flow connections, status pills, and category colors.
    """
    html_cards = []
    for idx, node in enumerate(nodes):
        ntype = node.get("type", "unknown")
        meta = ImageWorkflowEngine.NODE_TYPES.get(ntype, {"name": ntype, "category": "Processor", "icon": "⚙️"})
        enabled = node.get("enabled", True)
        cat = meta.get("category", "Processor")

        # Category colors mimicking n8n theme
        bg_color = "#1F2937"
        border_color = "#374151"
        badge_bg = "#4B5563"

        if cat == "Trigger":
            border_color = "#10B981"  # Emerald
            badge_bg = "#065F46"
        elif cat == "Processor":
            border_color = "#3B82F6"  # Blue
            badge_bg = "#1E40AF"
        elif cat == "Output":
            border_color = "#8B5CF6"  # Purple
            badge_bg = "#5B21B6"

        opacity = "1.0" if enabled else "0.45"
        status_dot = "🟢" if enabled else "⚪ Bypassed"

        card_html = f"""
        <div style="
            display: inline-flex;
            flex-direction: column;
            align-items: center;
            background: {bg_color};
            border: 2px solid {border_color};
            border-radius: 12px;
            padding: 12px 16px;
            min-width: 170px;
            opacity: {opacity};
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            margin: 8px 4px;
            font-family: system-ui, sans-serif;
            color: #F9FAFB;
        ">
            <div style="font-size: 24px; margin-bottom: 4px;">{meta.get('icon', '⚡')}</div>
            <div style="font-size: 13px; font-weight: 700; text-align: center; margin-bottom: 4px;">{node.get('name', meta['name'])}</div>
            <div style="
                background: {badge_bg};
                font-size: 10px;
                padding: 2px 8px;
                border-radius: 10px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                font-weight: 600;
                margin-bottom: 6px;
            ">{cat}</div>
            <div style="font-size: 10px; color: #9CA3AF;">{status_dot}</div>
        </div>
        """
        html_cards.append(card_html)
        if idx < len(nodes) - 1:
            arrow_html = """
            <div style="display: inline-flex; align-items: center; margin: 0 6px; color: #FF6D5A; font-size: 20px; font-weight: bold;">
                ➔
            </div>
            """
            html_cards.append(arrow_html)

    flow_container = f"""
    <div style="
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        background-color: #0D1117;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #30363D;
        overflow-x: auto;
        margin-bottom: 20px;
    ">
        {''.join(html_cards)}
    </div>
    """
    return flow_container

# Telegram Bot Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome to Hugging Face Image Editing Workflow Server Bot!\n"
        "- Chat with Llama 3 by sending a text message.\n"
        "- Send a PDF to summarize via Gemini 1.5 Flash.\n"
        "- Send a Photo to apply the server's active image workflow automatically!"
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
    status_msg = await update.message.reply_text("🎨 Applying active Image Workflow pipeline...")
    try:
        photo_file = await update.message.photo[-1].get_file()
        img_bytes = await photo_file.download_as_bytearray()
        input_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        steps = GLOBAL_WORKFLOW_CONFIG.get("active_workflow_steps", ImageWorkflowEngine.PRESETS["n8n Social Media Flow"])
        processed_img, telemetry = ImageWorkflowEngine.run_pipeline(input_img, steps)

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
        page_title="n8n Image Workflow Server Studio",
        page_icon="⚡",
        layout="wide"
    )

    st.title("⚡ n8n Image Workflow Server Studio")
    st.markdown(
        "Build, visualize, and execute modular **n8n-style Node Graph Workflows** on Hugging Face Spaces with PIL, Gemini 1.5 Flash, and Llama 3."
    )

    st.sidebar.header("⚙️ Server Status & Config")
    st.sidebar.markdown(f"- **Llama 3 Chat**: {'🟢 Active' if hf_client else '🔴 Inactive (Missing HF_TOKEN)'}")
    st.sidebar.markdown(f"- **Gemini 1.5 Flash**: {'🟢 Active' if GOOGLE_API_KEY and genai else '🔴 Inactive'}")
    st.sidebar.markdown(f"- **Telegram Bot**: {'🟢 Active' if TELEGRAM_TOKEN else '🔴 Inactive'}")

    st.sidebar.divider()
    st.sidebar.header("🎯 n8n Workflow Templates")
    preset_choice = st.sidebar.selectbox("Load Preset Workflow:", ["Custom"] + list(ImageWorkflowEngine.PRESETS.keys()))

    if preset_choice != "Custom":
        if st.sidebar.button(f"Load '{preset_choice}'"):
            st.session_state.current_workflow = list(ImageWorkflowEngine.PRESETS[preset_choice])
            GLOBAL_WORKFLOW_CONFIG["active_workflow_steps"] = st.session_state.current_workflow
            st.sidebar.success(f"Loaded template: {preset_choice}")

    if "current_workflow" not in st.session_state:
        st.session_state.current_workflow = list(ImageWorkflowEngine.PRESETS["n8n Social Media Flow"])
        GLOBAL_WORKFLOW_CONFIG["active_workflow_steps"] = st.session_state.current_workflow

    tab_canvas, tab_inspector, tab_batch, tab_ai, tab_json, tab_api = st.tabs([
        "🌐 n8n Flow Visualizer",
        "🎛️ Node Inspector & Builder",
        "📦 Batch Processing",
        "🤖 AI Multi-Modal Suite",
        "📄 Workflow JSON",
        "💻 API & Server Docs"
    ])

    # TAB 1: VISUAL FLOW CANVAS
    with tab_canvas:
        st.header("Interactive n8n Node Flowchart Canvas")
        st.markdown(render_n8n_flowchart_html(st.session_state.current_workflow), unsafe_allow_html=True)

        col_img, col_metrics = st.columns([1, 1])

        with col_img:
            st.subheader("Source & Render Preview")
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
                st.image(out_img, caption=f"Final Output Asset ({out_img.width}x{out_img.height})", use_container_width=True)

                buf = io.BytesIO()
                out_img.save(buf, format="PNG")
                st.download_button("💾 Download Rendered Asset (PNG)", data=buf.getvalue(), file_name="n8n_processed.png", mime="image/png")

        with col_metrics:
            st.subheader("⚡ Per-Node Execution Telemetry Metrics")
            if input_image and st.session_state.current_workflow:
                for t in telemetry:
                    status_badge = "🟢 SUCCESS" if t["status"] == "success" else ("⚪ BYPASSED" if t["status"] == "bypassed" else "🔴 ERROR")
                    st.markdown(f"**Step {t['step_number']}: {t['icon']} {t['name']}** — `{status_badge}`")
                    st.caption(f"Category: {t.get('category', 'Processor')} | Time: **{t['execution_time_ms']} ms** | Output Dim: **{t['output_dimensions']}**")
                    st.divider()

    # TAB 2: NODE INSPECTOR & BUILDER
    with tab_inspector:
        st.header("🎛️ Node Graph Inspector & Builder")

        col_nodes, col_add = st.columns([3, 2])

        with col_nodes:
            st.subheader("Configured Nodes Sequence")
            if not st.session_state.current_workflow:
                st.info("Graph is empty. Add a node from the panel.")
            else:
                for idx, node in enumerate(st.session_state.current_workflow):
                    ntype = node.get("type", "unknown")
                    meta = ImageWorkflowEngine.NODE_TYPES.get(ntype, {"name": ntype, "icon": "⚙️"})

                    with st.expander(f"#{idx+1} {meta['icon']} {node.get('name', meta['name'])} [{'Active' if node.get('enabled', True) else 'Bypassed'}]", expanded=False):
                        c1, c2, c3 = st.columns([2, 2, 1])
                        node["name"] = c1.text_input("Node Custom Label:", value=node.get("name", meta["name"]), key=f"lbl_{idx}")
                        node["enabled"] = c2.checkbox("Enable Node", value=node.get("enabled", True), key=f"enb_{idx}")
                        if c3.button("🗑️ Delete", key=f"del_{idx}"):
                            st.session_state.current_workflow.pop(idx)
                            GLOBAL_WORKFLOW_CONFIG["active_workflow_steps"] = st.session_state.current_workflow
                            st.rerun()

                        st.write("**Parameters:**", node.get("params", {}))

        with col_add:
            st.subheader("➕ Add New n8n Node")
            node_type_key = st.selectbox("Select Node Type:", list(ImageWorkflowEngine.NODE_TYPES.keys()), format_func=lambda k: f"{ImageWorkflowEngine.NODE_TYPES[k]['icon']} {ImageWorkflowEngine.NODE_TYPES[k]['name']} ({ImageWorkflowEngine.NODE_TYPES[k]['category']})")

            node_meta = ImageWorkflowEngine.NODE_TYPES[node_type_key]
            node_name = st.text_input("Node Name:", node_meta["name"])

            params = {}
            if node_type_key == "resize":
                w = st.number_input("Width (px):", min_value=10, max_value=4000, value=1080)
                h = st.number_input("Height (px):", min_value=10, max_value=4000, value=1080)
                aspect = st.checkbox("Keep Aspect Ratio", value=True)
                params = {"width": w, "height": h, "maintain_aspect_ratio": aspect}
            elif node_type_key == "crop":
                left = st.slider("Left %:", 0.0, 50.0, 0.0)
                top = st.slider("Top %:", 0.0, 50.0, 0.0)
                right = st.slider("Right %:", 50.0, 100.0, 100.0)
                bottom = st.slider("Bottom %:", 50.0, 100.0, 100.0)
                params = {"left_pct": left, "top_pct": top, "right_pct": right, "bottom_pct": bottom}
            elif node_type_key == "adjust_color":
                b = st.slider("Brightness:", 0.1, 2.0, 1.0, 0.05)
                c = st.slider("Contrast:", 0.1, 2.0, 1.0, 0.05)
                s = st.slider("Saturation:", 0.0, 2.0, 1.0, 0.05)
                sh = st.slider("Sharpness:", 0.0, 3.0, 1.0, 0.1)
                params = {"brightness": b, "contrast": c, "saturation": s, "sharpness": sh}
            elif node_type_key == "filter":
                ftype = st.selectbox("Filter:", ["grayscale", "sepia", "blur", "contour", "edge_enhance", "invert", "posterize", "vignette"])
                radius = st.slider("Radius (for blur):", 0.5, 10.0, 2.0) if ftype == "blur" else 2.0
                params = {"filter_type": ftype, "radius": radius}
            elif node_type_key == "watermark":
                wtext = st.text_input("Watermark Text:", "n8n Workflow")
                pos = st.selectbox("Position:", ["bottom-right", "bottom-left", "top-right", "top-left", "center"])
                color = st.color_picker("Color:", "#FFFFFF")
                opacity = st.slider("Opacity:", 0.1, 1.0, 0.8)
                params = {"text": wtext, "position": pos, "color": color, "opacity": opacity}

            if st.button("Add Node to Graph"):
                new_node_obj = {
                    "id": f"node_{len(st.session_state.current_workflow)+1}",
                    "type": node_type_key,
                    "name": node_name,
                    "enabled": True,
                    "params": params
                }
                st.session_state.current_workflow.append(new_node_obj)
                GLOBAL_WORKFLOW_CONFIG["active_workflow_steps"] = st.session_state.current_workflow
                st.success(f"Added node '{node_name}'!")
                st.rerun()

    # TAB 3: BATCH PROCESSING
    with tab_batch:
        st.header("📦 Batch Image Processing Server")
        batch_files = st.file_uploader("Upload multiple images for batch processing:", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)

        if batch_files and st.session_state.current_workflow:
            if st.button("🚀 Execute Batch Node Workflow"):
                results = []
                for i, file_item in enumerate(batch_files):
                    img = Image.open(file_item).convert("RGB")
                    pimg, _ = ImageWorkflowEngine.run_pipeline(img, st.session_state.current_workflow)
                    results.append((file_item.name, pimg))

                st.success(f"Processed {len(results)} images successfully!")
                cols = st.columns(min(3, len(results)))
                for idx, (fname, pimg) in enumerate(results):
                    with cols[idx % 3]:
                        st.image(pimg, caption=fname, use_container_width=True)

    # TAB 4: AI MULTI-MODAL SUITE
    with tab_ai:
        st.header("🤖 AI Multi-Modal & Photo Assistant")
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
            st.subheader("Export Graph JSON")
            json_str = ImageWorkflowEngine.export_workflow_json(st.session_state.current_workflow)
            st.code(json_str, language="json")
            st.download_button("💾 Download Workflow (.json)", data=json_str, file_name="n8n_image_workflow.json", mime="application/json")

        with col_imp:
            st.subheader("Import Graph JSON")
            uploaded_json = st.file_uploader("Upload Workflow JSON:", type=["json"], key="json_uploader")
            if uploaded_json and st.button("📥 Import JSON"):
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
        This Hugging Face Space operates an **n8n-style Node Graph Image Workflow Server**.

        ### Python Code Snippet to Run Node Graph Headlessly
        ```python
        from PIL import Image
        from workflow_engine import ImageWorkflowEngine

        img = Image.open("input.jpg")
        nodes = [
            {"id": "n1", "type": "trigger_file", "name": "Input Image", "enabled": True, "params": {}},
            {"id": "n2", "type": "resize", "name": "Resize 1080p", "enabled": True, "params": {"width": 1080, "height": 1080, "maintain_aspect_ratio": True}},
            {"id": "n3", "type": "filter", "name": "Sepia Tone", "enabled": True, "params": {"filter_type": "sepia"}},
            {"id": "n4", "type": "watermark", "name": "Watermark", "enabled": True, "params": {"text": "n8n Server"}}
        ]

        out_img, telemetry = ImageWorkflowEngine.run_pipeline(img, nodes)
        out_img.save("processed_output.jpg")
        print(telemetry)
        ```
        """)

if __name__ == "__main__":
    main()
