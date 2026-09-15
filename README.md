---
title: n8n Node Graph Image Editing Server
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

# ⚡ n8n Node Graph Image Editing Server (Hugging Face Space)

A full-featured **n8n-style Node Graph image editing server studio** hosted on Hugging Face Spaces. Build, inspect, edit, and execute node-based image pipelines on an authentic **n8n Dark Canvas** (`#090C10`) with SVG cubic bezier curved connections, port handles, node disabled state toggles, real-time per-node telemetry, batch execution, and AI multi-modal integrations.

## 🚀 Key Features

1. **Authentic n8n Node Graph Execution Engine (`workflow_engine.py`)**:
   - **n8n Node Schemas**: Every node follows standard n8n format (`id`, `type`, `name`, `typeVersion`, `position` [x, y], `parameters`, `disabled`).
   - **Node Categories & Color Themes**:
     - **Triggers (Emerald `#10B981`)**: File Upload (`n8n-nodes-base.fileTrigger`), Telegram Photo (`n8n-nodes-base.telegramTrigger`), AI Generation (`n8n-nodes-base.aiGenTrigger`).
     - **Actions (Blue `#3B82F6`)**: Resize (`n8n-nodes-base.resize`), Crop (`n8n-nodes-base.crop`), Color Grade (`n8n-nodes-base.adjustColor`), Rotate & Mirror (`n8n-nodes-base.rotateFlip`), Artistic Filter FX (`n8n-nodes-base.filterFx`).
     - **AI (Pink `#EC4899`)**: Gemini 1.5 AI Vision Inspector (`n8n-nodes-base.geminiVision`).
     - **Outputs (Purple `#8B5CF6`)**: Brand Watermark Stamp (`n8n-nodes-base.watermark`), Export Rendered Asset (`n8n-nodes-base.downloadOutput`).
   - **Edge Connections & Node Bypassing**: Pass image payloads along graph edges, skip disabled nodes, and capture telemetry (execution time in `ms`, output image dimensions, status).

2. **n8n Preset Workflow Templates**:
   - `n8n Social Media Auto-Branding` (1080p square formatting, vibrant color boost, watermark overlay stamp).
   - `n8n Vintage Filter Pipeline` (Warm contrast, sepia tone FX, vignette shading).

3. **Streamlit UI Studio (`app.py`)**:
   - **🌐 n8n Flow Canvas**: Interactive dark grid canvas (`#090C10`) rendering position-based node cards and SVG cubic bezier curved connection paths between input/output ports.
   - **🎛️ Node Inspector Drawer & Graph Editor**: Inspect node parameters, edit custom labels, toggle active/disabled states, delete nodes, and add new nodes from the library.
   - **📦 Batch Execution**: Run configured n8n node graphs across multiple image files in parallel.
   - **🤖 AI Multi-Modal Suite**: Gemini 1.5 Flash image inspection & HF FLUX/SDXL text-to-image base generation.
   - **📄 Workflow JSON Import/Export**: Import and export native n8n graph JSON definitions.
   - **💻 API & Server Docs**: Code examples for headless Python integration.

4. **Telegram Bot Integration**:
   - Send photos to your Telegram bot to execute the server's active n8n workflow graph automatically.

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
