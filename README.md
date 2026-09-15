---
title: n8n Image Workflow Server Studio
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

# ⚡ n8n Image Workflow Server Studio (Hugging Face Space)

A full-featured **n8n-style Node Graph image editing workflow server** hosted on Hugging Face Spaces. Design, visualize, inspect, and execute node-based image pipelines with live canvas rendering, node bypass toggles, per-node telemetry, batch processing, and AI multi-modal integrations.

## 🚀 Key Features

1. **n8n-Style Node Graph Engine (`workflow_engine.py`)**:
   - **Trigger Nodes**: File Upload Trigger (`trigger_file`), Telegram Photo Trigger (`trigger_telegram`), AI Generation Trigger (`trigger_ai_gen`).
   - **Processor Nodes**: Resize Node (`resize`), Crop Node (`crop`), Color Adjustment Node (`adjust_color`), Rotate & Flip Node (`rotate_flip`), Artistic Filter Node (`filter`), Gemini AI Analyzer Node (`gemini_analysis`).
   - **Output Nodes**: Watermark Overlay Node (`watermark`), Download Output Asset (`output_download`).
   - **Node Bypass & Telemetry**: Enable/Disable toggles per node, execution timing (`ms`), output image dimensions, and node status indicators (`🟢 SUCCESS`, `⚪ BYPASSED`, `🔴 ERROR`).

2. **n8n Preset Workflow Templates**:
   - `n8n Social Media Flow` (Square formatting, vibrant color grade, brand watermark overlay).
   - `n8n Vintage Film Flow` (Warm film tone, sepia filter node, vignette shading node).
   - `n8n Monochrome Art Flow` (Grayscale filter node, high contrast boost, signature stamp).
   - `n8n Edge Sketch Flow` (Contour edge detect, edge sharpening).

3. **Streamlit Interactive UI (`app.py`)**:
   - **🌐 n8n Flow Visualizer**: Interactive visual node flowchart canvas (`[Trigger Node] ➔ [Processor Nodes] ➔ [Output Node]`).
   - **🎛️ Node Inspector & Builder**: Inspect node parameters, toggle enable/disable, customize node labels, and add new n8n nodes.
   - **📦 Batch Processing**: Run configured n8n node graphs across multiple images simultaneously.
   - **🤖 AI Multi-Modal Suite**: Gemini 1.5 Flash image inspection and Hugging Face FLUX/SDXL base image generation.
   - **📄 Workflow JSON Import/Export**: Save, export, and load n8n graph JSON definitions.
   - **💻 API & Server Docs**: Code examples for headless Python integration.

4. **Telegram Bot Integration**:
   - Send photos to your Telegram bot.
   - Automatically executes the active server n8n node graph on incoming photos and responds with the processed asset and per-node telemetry log.

## 🛠️ Installation & Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Streamlit App Locally**:
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
