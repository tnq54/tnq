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
    "active_workflow_steps": list(ImageWorkflowEngine.PRESETS["Social Media Post"])
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

        # Access global thread-safe config instead of st.session_state
        steps = GLOBAL_WORKFLOW_CONFIG.get("active_workflow_steps", ImageWorkflowEngine.PRESETS["Social Media Post"])
        processed_img, logs = ImageWorkflowEngine.run_pipeline(input_img, steps)

        out_buffer = io.BytesIO()
        processed_img.save(out_buffer, format="JPEG", quality=92)
        out_buffer.seek(0)

        caption = "✨ Processed via HF Workflow Server!\n\nExecution log:\n" + "\n".join(logs[:5])
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
    # Background Bot Thread Initializer
    try:
        if "bot_thread" not in st.session_state:
            st.session_state.bot_thread = True
            thread = threading.Thread(target=run_bot, daemon=True)
            thread.start()
    except Exception:
        pass

    # Streamlit Page Config
    st.set_page_config(
        page_title="Hugging Face Image Editing Workflow Server",
        page_icon="🖼️",
        layout="wide"
    )

    st.title("⚡ Hugging Face Image Editing Workflow Server")
    st.markdown(
        "Automate, build, test, and execute modular image editing workflows on Hugging Face Spaces with PIL, Gemini AI, and Llama 3."
    )

    # Sidebar System Specs
    st.sidebar.header("⚙️ Server Status & Config")
    st.sidebar.markdown(f"- **Llama 3 Chat**: {'🟢 Active' if hf_client else '🔴 Inactive (Missing HF_TOKEN)'}")
    st.sidebar.markdown(f"- **Gemini 1.5 Flash**: {'🟢 Active' if GOOGLE_API_KEY and genai else '🔴 Inactive'}")
    st.sidebar.markdown(f"- **Telegram Bot**: {'🟢 Active' if TELEGRAM_TOKEN else '🔴 Inactive'}")

    st.sidebar.divider()
    st.sidebar.header("🎯 Workflow Presets")
    preset_choice = st.sidebar.selectbox("Load Preset Workflow:", ["Custom"] + list(ImageWorkflowEngine.PRESETS.keys()))

    if preset_choice != "Custom":
        if st.sidebar.button(f"Load '{preset_choice}'"):
            st.session_state.current_workflow = list(ImageWorkflowEngine.PRESETS[preset_choice])
            GLOBAL_WORKFLOW_CONFIG["active_workflow_steps"] = st.session_state.current_workflow
            st.sidebar.success(f"Loaded preset: {preset_choice}")

    if "current_workflow" not in st.session_state:
        st.session_state.current_workflow = list(ImageWorkflowEngine.PRESETS["Social Media Post"])
        GLOBAL_WORKFLOW_CONFIG["active_workflow_steps"] = st.session_state.current_workflow

    # Navigation Tabs
    tab_builder, tab_batch, tab_ai, tab_json, tab_api = st.tabs([
        "🎨 Workflow Builder",
        "📦 Batch Processing",
        "🤖 AI Multi-Modal Suite",
        "📄 Workflow JSON",
        "💻 API & Server Docs"
    ])

    # TAB 1: WORKFLOW BUILDER
    with tab_builder:
        st.header("Visual Workflow Engine Builder")
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("1. Source Image Input")
            upload_source = st.radio("Image Source:", ["Upload Local File", "Generate via AI Text Prompt", "Sample Placeholder"], horizontal=True)

            input_image = None
            if upload_source == "Upload Local File":
                uploaded_file = st.file_uploader("Choose an image file:", type=["png", "jpg", "jpeg", "webp"])
                if uploaded_file:
                    input_image = Image.open(uploaded_file).convert("RGB")
            elif upload_source == "Generate via AI Text Prompt":
                gen_prompt = st.text_input("Enter AI Image Generation Prompt:", "A surreal futuristic cyberpunk city at twilight, 8k resolution, photorealistic")
                if st.button("✨ Generate Base Image"):
                    with st.spinner("Generating image via Hugging Face Inference..."):
                        gen_img = generate_image_hf(gen_prompt)
                        if gen_img:
                            st.session_state.generated_base_image = gen_img
                            st.success("Image generated successfully!")
                        else:
                            st.error("Failed to generate image. Please check HF_TOKEN.")
                if "generated_base_image" in st.session_state:
                    input_image = st.session_state.generated_base_image
            else:  # Sample placeholder
                sample_img = Image.new("RGB", (800, 600), color=(73, 109, 137))
                d = ImageDraw.Draw(sample_img)
                d.rectangle([(100, 100), (700, 500)], fill=(255, 192, 203), outline=(255, 255, 255), width=5)
                d.ellipse([(250, 150), (550, 450)], fill=(135, 206, 250), outline=(0, 0, 0), width=3)
                input_image = sample_img

            if input_image:
                st.image(input_image, caption=f"Original Image ({input_image.width}x{input_image.height})", use_container_width=True)

        with col_right:
            st.subheader("2. Configure Workflow Pipeline Steps")

            with st.expander("➕ Add Step to Workflow", expanded=True):
                step_type = st.selectbox("Select Action Type:", ["Resize", "Crop", "Color Adjustments", "Rotate & Flip", "Artistic Filter", "Watermark Text"])

                new_step = None
                if step_type == "Resize":
                    w = st.number_input("Target Width (px):", min_value=10, max_value=4000, value=1080, step=10)
                    h = st.number_input("Target Height (px):", min_value=10, max_value=4000, value=1080, step=10)
                    aspect = st.checkbox("Maintain Aspect Ratio", value=True)
                    new_step = {"type": "resize", "params": {"width": w, "height": h, "maintain_aspect_ratio": aspect}}

                elif step_type == "Crop":
                    left = st.slider("Crop Left %:", 0.0, 50.0, 0.0)
                    top = st.slider("Crop Top %:", 0.0, 50.0, 0.0)
                    right = st.slider("Crop Right %:", 50.0, 100.0, 100.0)
                    bottom = st.slider("Crop Bottom %:", 50.0, 100.0, 100.0)
                    new_step = {"type": "crop", "params": {"left_pct": left, "top_pct": top, "right_pct": right, "bottom_pct": bottom}}

                elif step_type == "Color Adjustments":
                    b = st.slider("Brightness:", 0.1, 2.0, 1.0, 0.05)
                    c = st.slider("Contrast:", 0.1, 2.0, 1.0, 0.05)
                    s = st.slider("Saturation:", 0.0, 2.0, 1.0, 0.05)
                    sh = st.slider("Sharpness:", 0.0, 3.0, 1.0, 0.1)
                    new_step = {"type": "adjust_color", "params": {"brightness": b, "contrast": c, "saturation": s, "sharpness": sh}}

                elif step_type == "Rotate & Flip":
                    angle = st.selectbox("Rotation Angle:", [0, 90, 180, 270])
                    fh = st.checkbox("Flip Horizontal (Mirror)")
                    fv = st.checkbox("Flip Vertical")
                    new_step = {"type": "rotate_flip", "params": {"angle": angle, "flip_h": fh, "flip_v": fv}}

                elif step_type == "Artistic Filter":
                    ftype = st.selectbox("Filter Type:", ["grayscale", "sepia", "blur", "contour", "edge_enhance", "invert", "posterize", "vignette"])
                    radius = 2.0
                    if ftype == "blur":
                        radius = st.slider("Blur Radius:", 0.5, 10.0, 2.0, 0.5)
                    new_step = {"type": "filter", "params": {"filter_type": ftype, "radius": radius}}

                elif step_type == "Watermark Text":
                    wtext = st.text_input("Watermark Text:", "HF Workflow Engine")
                    pos = st.selectbox("Position:", ["bottom-right", "bottom-left", "top-right", "top-left", "center"])
                    color = st.color_picker("Text Color:", "#FFFFFF")
                    opacity = st.slider("Opacity:", 0.1, 1.0, 0.8, 0.05)
                    new_step = {"type": "watermark", "params": {"text": wtext, "position": pos, "color": color, "opacity": opacity}}

                if st.button("Add Step to Pipeline"):
                    st.session_state.current_workflow.append(new_step)
                    GLOBAL_WORKFLOW_CONFIG["active_workflow_steps"] = st.session_state.current_workflow
                    st.success(f"Added '{step_type}' to pipeline!")

            st.subheader("Current Pipeline Sequence")
            if not st.session_state.current_workflow:
                st.info("No steps in pipeline. Add steps above or select a preset.")
            else:
                for idx, step in enumerate(st.session_state.current_workflow):
                    c1, c2 = st.columns([4, 1])
                    c1.write(f"**Step {idx+1}:** `{step['type']}` — `{step['params']}`")
                    if c2.button("🗑️", key=f"del_step_{idx}"):
                        st.session_state.current_workflow.pop(idx)
                        GLOBAL_WORKFLOW_CONFIG["active_workflow_steps"] = st.session_state.current_workflow
                        st.rerun()

                if st.button("🧹 Clear All Steps"):
                    st.session_state.current_workflow = []
                    GLOBAL_WORKFLOW_CONFIG["active_workflow_steps"] = []
                    st.rerun()

        st.divider()
        st.subheader("3. Rendered Output Preview & Pipeline Logs")
        if input_image and st.session_state.current_workflow:
            output_img, exec_logs = ImageWorkflowEngine.run_pipeline(input_image, st.session_state.current_workflow)

            prev_col1, prev_col2 = st.columns(2)
            with prev_col1:
                st.image(input_image, caption="Before (Original)", use_container_width=True)
            with prev_col2:
                st.image(output_img, caption=f"After Processed ({output_img.width}x{output_img.height})", use_container_width=True)

                buf = io.BytesIO()
                output_img.save(buf, format="PNG")
                st.download_button("💾 Download Output Image (PNG)", data=buf.getvalue(), file_name="workflow_result.png", mime="image/png")

            with st.expander("📋 Execution Logs", expanded=False):
                for log in exec_logs:
                    st.text(log)
        elif not input_image:
            st.warning("Please select or upload a source image.")

    # TAB 2: BATCH PROCESSING
    with tab_batch:
        st.header("📦 Batch Image Processing Server")
        st.write("Apply your current active workflow to multiple images at once.")

        batch_files = st.file_uploader("Upload multiple images for batch processing:", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)

        if batch_files and st.session_state.current_workflow:
            if st.button("🚀 Process Batch Images"):
                st.info(f"Processing {len(batch_files)} images using active workflow...")
                progress_bar = st.progress(0)

                results = []
                for i, file_item in enumerate(batch_files):
                    img = Image.open(file_item).convert("RGB")
                    processed_img, _ = ImageWorkflowEngine.run_pipeline(img, st.session_state.current_workflow)
                    results.append((file_item.name, processed_img))
                    progress_bar.progress((i + 1) / len(batch_files))

                st.success(f"Successfully processed {len(results)} images!")

                cols = st.columns(min(3, len(results)))
                for idx, (fname, pimg) in enumerate(results):
                    with cols[idx % 3]:
                        st.image(pimg, caption=f"Processed: {fname}", use_container_width=True)
                        buf = io.BytesIO()
                        pimg.save(buf, format="JPEG", quality=90)
                        st.download_button(f"Download {fname}", data=buf.getvalue(), file_name=f"processed_{fname}", mime="image/jpeg", key=f"dl_batch_{idx}")

    # TAB 3: AI MULTI-MODAL SUITE
    with tab_ai:
        st.header("🤖 AI Multi-Modal & Photo Assistant")

        st.subheader("Gemini 1.5 Flash Photo Analyzer")
        ai_file = st.file_uploader("Upload image to analyze with Gemini 1.5 Flash:", type=["png", "jpg", "jpeg", "webp"], key="ai_photo_uploader")
        custom_prompt = st.text_input("Analysis Prompt:", "Analyze this image in detail and recommend optimal editing adjustments.")

        if ai_file and st.button("🔍 Analyze Image"):
            with st.spinner("Analyzing with Gemini 1.5 Flash..."):
                res_analysis = analyze_image_with_gemini(ai_file.getvalue(), prompt=custom_prompt)
                st.markdown("### Gemini Analysis Result")
                st.write(res_analysis)

    # TAB 4: WORKFLOW JSON IMPORT/EXPORT
    with tab_json:
        st.header("📄 Workflow JSON Configuration")
        st.write("Export your active workflow to JSON, or import an existing workflow JSON definition.")

        col_exp, col_imp = st.columns(2)

        with col_exp:
            st.subheader("Export Workflow JSON")
            json_str = ImageWorkflowEngine.export_workflow_json(st.session_state.current_workflow)
            st.code(json_str, language="json")
            st.download_button("💾 Download Workflow (.json)", data=json_str, file_name="image_workflow.json", mime="application/json")

        with col_imp:
            st.subheader("Import Workflow JSON")
            uploaded_json = st.file_uploader("Upload Workflow JSON File:", type=["json"], key="json_uploader")
            raw_json_input = st.text_area("Or Paste JSON String here:", height=150)

            if st.button("📥 Load Workflow JSON"):
                content_to_parse = ""
                if uploaded_json:
                    content_to_parse = uploaded_json.getvalue().decode("utf-8")
                elif raw_json_input:
                    content_to_parse = raw_json_input

                if content_to_parse:
                    imported_steps = ImageWorkflowEngine.import_workflow_json(content_to_parse)
                    if imported_steps:
                        st.session_state.current_workflow = imported_steps
                        GLOBAL_WORKFLOW_CONFIG["active_workflow_steps"] = imported_steps
                        st.success(f"Successfully loaded {len(imported_steps)} workflow steps!")
                        st.rerun()
                    else:
                        st.error("Failed to parse workflow JSON. Please check formatting.")

    # TAB 5: API & SERVER DOCS
    with tab_api:
        st.header("💻 Server API & Integration Documentation")
        st.markdown("""
        This Hugging Face Space operates as a **Image Editing Workflow Server**.
        You can trigger image workflows via Python, cURL, or directly via Telegram photos.

        ### Telegram Integration
        - Send any photo directly to your connected Telegram Bot.
        - The bot automatically applies the active workflow pipeline configured in this server and replies with the processed image.

        ### Python Code Snippet to Run Engine Headlessly
        ```python
        from PIL import Image
        from workflow_engine import ImageWorkflowEngine

        # Load input image
        img = Image.open("input.jpg")

        # Define Workflow Steps
        workflow = [
            {"type": "resize", "params": {"width": 1080, "height": 1080, "maintain_aspect_ratio": True}},
            {"type": "adjust_color", "params": {"brightness": 1.1, "contrast": 1.2, "saturation": 1.15}},
            {"type": "filter", "params": {"filter_type": "vignette"}},
            {"type": "watermark", "params": {"text": "HF Workflow Server", "position": "bottom-right"}}
        ]

        # Run Pipeline
        processed_img, logs = ImageWorkflowEngine.run_pipeline(img, workflow)

        # Save Output
        processed_img.save("output_processed.jpg")
        print(logs)
        ```
        """)

if __name__ == "__main__":
    main()
