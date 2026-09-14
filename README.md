---
title: Image Editing Workflow Server
emoji: 🖼️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

# Image Editing Workflow Server (Hugging Face Space)

A full-featured server and workflow builder interface for image editing, transformation, batch processing, and AI multi-modal analysis deployed on Hugging Face Spaces.

## 🚀 Key Features

1. **Modular Image Pipeline Engine (`workflow_engine.py`)**:
   - **Resize**: Custom width/height and aspect ratio constraints.
   - **Crop**: Precision percentage-based cropping.
   - **Color Adjustments**: Real-time Brightness, Contrast, Saturation, and Sharpness controls.
   - **Rotate & Flip**: 90/180/270 degree rotation, horizontal mirror, and vertical flipping.
   - **Artistic Filters**: Grayscale, Sepia, Gaussian Blur, Contour, Edge Enhancement, Invert, Posterize, and Vignette.
   - **Watermarking**: Dynamic overlay text with custom positioning, font sizing, color picker, and alpha opacity.

2. **Preset Workflow Library**:
   - Social Media Post (1080x1080 square format, vibrant color grade, watermark)
   - Aesthetic Vintage (Sepia tones, vignette shading, warm contrast)
   - High Contrast Monochrome
   - Soft Portrait Blur
   - Edge Art Sketch

3. **Streamlit Interactive UI (`app.py`)**:
   - **🎨 Workflow Builder**: Visual pipeline assembly, node addition/removal, and live before/after image previews.
   - **📦 Batch Processing**: Apply active workflow pipelines across multiple uploaded images simultaneously.
   - **🤖 AI Multi-Modal Suite**: Gemini 1.5 Flash image inspection and Hugging Face FLUX/SDXL image generation.
   - **📄 Workflow JSON Import/Export**: Export workflows to JSON or import workflow configurations.
   - **💻 API & Server Docs**: Code examples for headless Python integration and server usage.

4. **Telegram Bot Integration**:
   - Send photos directly to your Telegram bot.
   - Automatically executes the active server workflow on incoming photos and responds with the edited image and execution logs.

## 🛠️ Installation & Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Streamlit Server Locally**:
   ```bash
   streamlit run app.py
   ```

3. **Environment Variables** (Optional):
   - `HF_TOKEN`: Hugging Face Token for Llama 3 Chat and FLUX/SDXL Image Generation.
   - `TELEGRAM_TOKEN`: Telegram Bot Token for background bot execution.
   - `GOOGLE_API_KEY`: Google GenAI Key for Gemini 1.5 Flash summarization and image analysis.

## 🧪 Testing

Run the automated test suite with pytest:
```bash
pytest test_workflow.py
```
