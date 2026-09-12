---
title: Hugging Face Image LoRA Training Studio
emoji: 🎨
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

# 🎨 Hugging Face Image LoRA Training Studio

A complete, production-ready Streamlit Hugging Face Space application for training Image LoRA adapters on FLUX.1, SDXL, SD 1.5, and Vision-Language models.

## ✨ Features

- **📸 Image Dataset Management & Gemini 1.5 Auto-Captioning**:
  - Upload dataset images (`.png`, `.jpg`, `.jpeg`, `.webp`).
  - Auto-generate detailed prompt captions using Gemini 1.5 Flash.
  - Interactive caption editor and dataset ZIP package exporter/importer.

- **⚙️ LoRA Hyperparameters & Precision Studio**:
  - Presets for `black-forest-labs/FLUX.1-dev`, `stabilityai/stable-diffusion-xl-base-1.0`, `runwayml/stable-diffusion-v1-5`, and custom repositories.
  - Configurable resolution, LoRA rank (`r`), LoRA alpha, UNet LR, Text Encoder LR, repeat count, and epochs.
  - Performance optimizations: Aspect Ratio Bucketing (ARB), 4-bit QLoRA, Gradient Checkpointing, SafeTensors export, Conv LoRA dimensions, and Min-SNR Gamma weighting.
  - Export & import training configurations as JSON (`training_config.json`).

- **🚀 Real-time Training Execution & Monitoring**:
  - Launch `train_lora.py` directly from the Streamlit UI.
  - View live progress, real-time training logs, and loss telemetry.

- **📦 Direct Hugging Face Hub Export & Publish**:
  - Download `.safetensors` LoRA adapter checkpoints.
  - Publish trained LoRA adapters directly to Hugging Face Hub repos with a single click.

- **🤖 Hybrid AI Bot System**:
  - Llama 3 chat inference via Hugging Face Inference API.
  - PDF document extraction & summarization via Gemini 1.5 Flash.
  - Telegram Bot integration (`python-telegram-bot`).

## 🛠️ Environment Variables

Configure the following secrets in your Hugging Face Space settings:

| Variable | Description |
|---|---|
| `HF_TOKEN` | Hugging Face User Access Token (read/write access for pushing adapters) |
| `GOOGLE_API_KEY` | Google Gemini API Key for auto-captioning and summarization |
| `TELEGRAM_TOKEN` | Telegram Bot API Token for background bot operations |

## 🚀 Running Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

3. Or run command-line LoRA training directly:
   ```bash
   python train_lora.py --base_model "black-forest-labs/FLUX.1-dev" --dataset_dir "./dataset" --resolution 1024 --lora_r 16
   ```

## 🧪 Testing

Run the automated test suite:
```bash
python test_lora.py
```
