import sys
import os
import streamlit as st

if __name__ == "__main__":
    # If the script is executed directly via `python app.py` instead of `streamlit run app.py`
    if not st.runtime.exists():
        port = os.environ.get("PORT", "7860")
        cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", port, "--server.address", "0.0.0.0"]
        print(f"Direct python execution detected. Relaunching under Streamlit: {' '.join(cmd)}")
        os.execv(sys.executable, cmd)
import time
import threading
import asyncio
import io
import logging
import random
import socket
import pandas as pd
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

# Suppress Streamlit's missing ScriptRunContext warning at import time / bare modes
class ScriptRunContextFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "missing ScriptRunContext" in msg:
            return False
        if "Session state does not function" in msg:
            return False
        return True

logging.getLogger().addFilter(ScriptRunContextFilter())
# Apply filter to all current and future loggers in streamlit namespace
for name in ["streamlit", "streamlit.runtime.scriptrunner_utils.script_run_context", "streamlit.runtime.scriptrunner", "streamlit.runtime.state.session_state_proxy"]:
    logging.getLogger(name).addFilter(ScriptRunContextFilter())

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

# Network Check Utility
def check_network(host="8.8.8.8", port=53, timeout=1):
    try:
        socket.create_connection((host, port), timeout=timeout)
        return True
    except OSError:
        return False

# PDF Text Extraction
def extract_pdf_text(file_bytes):
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages_text = []
        for page in reader.pages:
            pages_text.append(page.extract_text() or "")
        return "".join(pages_text)
    except Exception as e:
        logger.error(f"PDF Extraction Error: {e}")
        return None

# Gemini Summarization
def summarize_with_gemini(text):
    if not GOOGLE_API_KEY:
        return "Lỗi: Chưa cấu hình GOOGLE_API_KEY."
    if not genai:
        return "Lỗi: Thư viện google-genai chưa được cài đặt."

    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=f"Summarize this document:\n\n{text[:30000]}"
        )
        return response.text
    except Exception as e:
        logger.error(f"Gemini Error: {e}")
        return f"Lỗi tóm tắt văn bản: {e}"

# Telegram Bot Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Chào mừng bạn đến với VBot1!\n"
        "- Chat với tôi (Llama 3).\n"
        "- Gửi tệp PDF để tóm tắt nội dung (Gemini 1.5 Flash)."
    )

async def chat_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    if not hf_client:
        await update.message.reply_text("Llama 3 hiện không khả dụng (thiếu HF_TOKEN).")
        return

    status_msg = await update.message.reply_text("Đang suy nghĩ...")
    try:
        messages = [{"role": "user", "content": user_text}]
        # Run blocking hf call in executor thread to keep event loop responsive
        completion = await asyncio.to_thread(
            hf_client.chat_completion,
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=messages,
            max_tokens=500
        )
        reply = completion.choices[0].message.content
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status_msg.message_id, text=reply)
    except Exception as e:
        logger.error(f"Llama 3 Error: {e}")
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status_msg.message_id, text=f"Lỗi: {e}")

async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if doc.mime_type != 'application/pdf':
        await update.message.reply_text("Vui lòng gửi tệp định dạng PDF.")
        return

    status_msg = await update.message.reply_text("Đang tải xuống tệp PDF...")
    try:
        file = await context.bot.get_file(doc.file_id)
        file_bytes = await file.download_as_bytearray()

        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status_msg.message_id, text="Đang trích xuất văn bản...")
        # Offload CPU-heavy text extraction
        text = await asyncio.to_thread(extract_pdf_text, file_bytes)

        if not text:
            await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status_msg.message_id, text="Không tìm thấy văn bản trong tệp PDF.")
            return

        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status_msg.message_id, text="Đang tóm tắt nội dung bằng Gemini 1.5 Flash...")
        # Offload network-blocking Gemini call
        summary = await asyncio.to_thread(summarize_with_gemini, text)

        # Gửi phản hồi, chia nhỏ tin nhắn nếu quá dài
        if len(summary) > 4000:
            for i in range(0, len(summary), 4000):
                await update.message.reply_text(summary[i:i+4000])
        else:
            await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status_msg.message_id, text=summary)

    except Exception as e:
        logger.error(f"Document Error: {e}")
        await update.message.reply_text(f"Gặp lỗi khi xử lý tệp: {e}")

# Bulletproof Telegram Bot Runner
def run_bot():
    logger.info("Initializing Telegram Bot Runner thread...")

    # Wait for network using robust check
    backoff = 2
    max_backoff = 60
    while not check_network():
        logger.warning(f"Network unavailable. Retrying connection check in {backoff} seconds...")
        time.sleep(backoff)
        backoff = min(backoff * 2, max_backoff)

    logger.info("Network check passed successfully!")

    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN is missing. Bot cannot be started.")
        return

    poll_backoff = 2
    max_poll_backoff = 60

    while True:
        loop = None
        try:
            # Create fresh event loop inside retry loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Re-instantiate Application
            application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

            application.add_handler(CommandHandler("start", start))
            application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat_handler))
            application.add_handler(MessageHandler(filters.Document.PDF, document_handler))

            logger.info("Starting Telegram Bot long-polling...")
            # Run polling with close_loop=True
            application.run_polling(stop_signals=None, close_loop=True)

            # Reset backoff upon successful running
            poll_backoff = 2

        except NetworkError as e:
            logger.error(f"Telegram NetworkError encountered: {e}. Retrying in {poll_backoff}s...")
            time.sleep(poll_backoff)
            poll_backoff = min(poll_backoff * 2, max_poll_backoff)
        except Exception as e:
            logger.error(f"Telegram Bot exception raised: {e}. Retrying in {poll_backoff}s...")
            time.sleep(poll_backoff)
            poll_backoff = min(poll_backoff * 2, max_poll_backoff)
        finally:
            if loop and not loop.is_closed():
                try:
                    loop.close()
                except Exception as ex:
                    logger.error(f"Failed to close loop: {ex}")


# ----------------- BRAIN SIMULATOR GAME LOGIC -----------------

GRID_SIZE = 4 # 3D grid size 4x4x4 (64 nodes)

def serialize_grid(grid):
    dir_map = {"All": "A", "Up": "U", "Right": "R", "Down": "D", "Left": "L", "Front": "F", "Back": "B"}
    cells = []
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            for z in range(GRID_SIZE):
                cell = grid[x][y][z]
                t_char = cell["type"][0] # E, S, I, M
                d_char = dir_map.get(cell.get("direction", "All"), "A")
                weight_val = int(cell.get("weight", 1.0))
                cells.append(f"{t_char}{d_char}{weight_val}")
    return "-".join(cells)

def deserialize_grid(code_string):
    if not isinstance(code_string, str):
        return None
    code_string = code_string.strip()
    if not code_string:
        return None
    parts = code_string.split("-")
    if len(parts) != GRID_SIZE * GRID_SIZE * GRID_SIZE:
        return None

    type_map = {"E": "Empty", "S": "Sensory", "I": "Interneuron", "M": "Motor"}
    dir_map = {"A": "All", "U": "Up", "R": "Right", "D": "Down", "L": "Left", "F": "Front", "B": "Back"}

    new_grid = [[[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    idx = 0
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            for z in range(GRID_SIZE):
                part = parts[idx]
                if len(part) < 2:
                    return None
                t_char, d_char = part[0], part[1]
                weight_val = 1.0
                if len(part) >= 3:
                    try:
                        weight_val = float(part[2:])
                    except ValueError:
                        weight_val = 1.0
                t_name = type_map.get(t_char, "Empty")
                d_name = dir_map.get(d_char, "All")

                new_grid[x][y][z] = {
                    "type": t_name,
                    "charge": 0.0,
                    "threshold": 0.4 if t_name == "Sensory" else (0.6 if t_name == "Motor" else 0.5),
                    "fire_rate": 0.2 if t_name == "Sensory" else 0.0,
                    "last_fired": -1,
                    "direction": d_name,
                    "weight": weight_val,
                    "amyloid_plaque": False
                }
                idx += 1
    return new_grid

def init_game_state():
    if "game_initialized" not in st.session_state or not st.session_state.game_initialized:
        st.session_state.game_initialized = True

        # Initialize 3D 4x4x4 grid
        grid = []
        for x in range(GRID_SIZE):
            plane = []
            for y in range(GRID_SIZE):
                row = []
                for z in range(GRID_SIZE):
                    row.append({
                        "type": "Empty",
                        "charge": 0.0,
                        "threshold": 0.5,
                        "fire_rate": 0.25,
                        "last_fired": -1,
                        "direction": "All",
                        "weight": 1.0,
                        "amyloid_plaque": False
                    })
                plane.append(row)
            grid.append(plane)

        # Default starting network structure
        grid[0][0][0] = {"type": "Sensory", "charge": 0.0, "threshold": 0.4, "fire_rate": 0.35, "last_fired": -1, "direction": "All", "weight": 1.0, "amyloid_plaque": False}
        grid[1][1][1] = {"type": "Interneuron", "charge": 0.0, "threshold": 0.5, "fire_rate": 0.0, "last_fired": -1, "direction": "All", "weight": 1.0, "amyloid_plaque": False}
        grid[3][3][3] = {"type": "Motor", "charge": 0.0, "threshold": 0.6, "fire_rate": 0.0, "last_fired": -1, "direction": "All", "weight": 1.0, "amyloid_plaque": False}

        st.session_state.neuron_grid = grid

        # Chemistry metrics (added melatonin, neuro_inflammation, norepinephrine, gaba, neuro_nutrients)
        st.session_state.chemicals = {
            "dopamine": 50.0,
            "serotonin": 50.0,
            "acetylcholine": 50.0,
            "energy": 100.0,
            "stress": 10.0,
            "sanity": 100.0,
            "melatonin": 10.0,
            "neuro_inflammation": 10.0,
            "norepinephrine": 10.0,
            "gaba": 30.0,
            "neuro_nutrients": 80.0
        }

        # 3D Index stats (glowing WebGPU diagnostics)
        st.session_state.csi = 0.0 # Cognitive Sync Index
        st.session_state.pdi = 0.0 # Plasticity Density Index
        st.session_state.vpi = 80.0 # Vascular Perfusion Index

        # Advanced Game Modes
        st.session_state.game_mode = "Normal"
        st.session_state.active_genes = []
        st.session_state.cellular_evolution = True

        # Active ongoing buffs (seconds)
        st.session_state.active_buffs = {
            "doping": 0,
            "ssri": 0,
            "focus": 0,
            "tyrosine": 0,
            "tryptophan": 0,
            "choline": 0,
            "glutamate": 0,
            "somatosensory_gating": 0,
            "oxytocin": 0
        }

        # Cooldowns
        st.session_state.cooldowns = {
            "doping": 0,
            "ssri": 0,
            "focus": 0,
            "rtms": 0,
            "opto": 0,
            "cortisol": 0,
            "propranolol": 0,
            "sprouting": 0,
            "vns": 0
        }

        st.session_state.audio_trigger = None

        # Challenges & Missions System
        st.session_state.missions = {
            "reflex": {"name": "⚡ Cung Phản Xạ Sinh Học 3D", "target": "Đặt ít nhất 1 Sensory và 1 Motor trên lưới 3D", "status": "In Progress", "reward_claimed": False, "desc": "+100 MB Trí nhớ"},
            "loop": {"name": "🧠 Vòng Lặp Phản Hồi Tự Trị 3D", "target": "Đặt ít nhất 6 nơ-ron hoạt động trên lưới 3D", "status": "In Progress", "reward_claimed": False, "desc": "+300 IQ"},
            "zen": {"name": "🧘 Thiền Tĩnh Tâm Trị Liệu", "target": "Căng thẳng dưới 5% và Tỉnh táo trên 95%", "status": "In Progress", "reward_claimed": False, "desc": "+40 Dopamine & +40 Serotonin"},
            "marathon": {"name": "🏆 Chạy Đua Nhận Thức Siêu Phàm", "target": "Tích lũy tối thiểu 500 điểm IQ nhận thức", "status": "In Progress", "reward_claimed": False, "desc": "+50 Acetylcholine & +200 MB Trí nhớ"}
        }

        # Progression stats
        st.session_state.stats = {
            "iq": 0.0,
            "memory": 10.0,
            "ticks": 0,
            "evolution_stage": "Đơn bào (Single-Cell)",
            "burnout_count": 0,
            "burnout_streak": 0,
            "max_streak": 0,
            "high_score_iq": 0.0,
            "max_memory": 10.0,
            "circadian_cycle": "Day",
            "sleep_state": False,
            "glycogen_pool": 30.0
        }

        # Upgrades
        st.session_state.upgrades = {
            "brainstem": 1,
            "cerebellum": 1,
            "hippocampus": 1,
            "cortex": 1,
            "myelin": 0,
            "plasticity": 0,
            "pruning": 0,
            "pfc": 0,
            "amygdala": 0,
            "thalamus": 0,
            "glycogen_shunt": 0,
            "dentate_gyrus": 0,
            "occipital_lobe": 0,
            "temporal_lobe": 0,
            "ltp_consolidator": 0,
            "parietal_lobe": 0,
            "pituitary_gland": 0,
            "blood_brain_barrier": 0
        }

        st.session_state.save_slots = {
            "Slot 1": None,
            "Slot 2": None,
            "Slot 3": None
        }

        st.session_state.game_log = ["Khởi tạo bộ não 3D WebGPU thành công. Trạng thái tiến hóa: Đơn bào (Single-Cell)."]
        st.session_state.playing = False
        st.session_state.tick_speed = 1.0
        st.session_state.selected_cell = (0, 0, 0) # 3D Coordinate
        st.session_state.current_event = None
        st.session_state.history_data = {
            "tick": [0],
            "sanity": [100.0],
            "energy": [100.0],
            "dopamine": [50.0],
            "stress": [10.0],
            "norepinephrine": [10.0],
            "gaba": [30.0],
            "csi": [0.0],
            "pdi": [0.0],
            "vpi": [80.0]
        }

def get_evolution_stage(iq):
    if iq < 100:
        return "Đơn bào (Single-Cell)"
    elif iq < 300:
        return "Đa bào (Multi-Cell)"
    elif iq < 1000:
        return "Sứa biển (Coelenterate)"
    elif iq < 3000:
        return "Bò sát (Instinct)"
    elif iq < 10000:
        return "Thú cổ (Emotional)"
    else:
        return "Người tinh khôn (Logical)"

def trigger_random_event():
    events = [
        {
            "title": "☕ Cốc Espresso Đậm Đặc",
            "desc": "Bạn nạp một liều caffeine cực mạnh vào cơ thể để tăng tốc độ xử lý thông tin.",
            "choices": [
                {
                    "label": "Uống cạn ly (Tăng +15 MB Trí nhớ, +15% Căng thẳng)",
                    "effect": "None",
                    "apply": lambda: apply_event_effects(da=5.0, ach=10.0, stress=15.0, mem_gain=15.0, log_msg="Uống Espresso: Khuyếch đại truyền dẫn acetylcholine và nâng cao trí lực ngắn hạn.")
                },
                {
                    "label": "Chọn trà xanh thanh lọc (+5% Tỉnh táo, +5% GABA)",
                    "effect": "None",
                    "apply": lambda: apply_event_effects(se=8.0, sanity=5.0, stress=-10.0, log_msg="Chọn trà xanh: L-Theanine làm dịu tâm trí, sạc nhẹ GABA ức chế bớt quá tải.")
                }
            ]
        },
        {
            "title": "⚡ Cơn Bão Từ Trường Toàn Cầu",
            "desc": "Bão địa từ quét qua Trái Đất gây nhiễu loạn nhẹ dòng điện áp sinh học vỏ não.",
            "choices": [
                {
                    "label": "Mở rộng liên kết đón bão (Nhận ngẫu nhiên +10.0% điện tích các ô)",
                    "effect": "None",
                    "apply": lambda: apply_magnetic_storm(0.10, "Từ trường cộng hưởng nhẹ: Gia tăng điện áp tích lũy nơ-ron.")
                },
                {
                    "label": "Đóng mạch bảo vệ tế bào (Tiêu tốn -20% Glucose năng lượng)",
                    "effect": "None",
                    "apply": lambda: apply_event_effects(energy=-20.0, stress=-15.0, log_msg="Mạch nơ-ron đóng kẹp an toàn: Bảo toàn điện thế ổn định, tiêu hao glucose dự phòng.")
                }
            ]
        }
    ]
    st.session_state.current_event = random.choice(events)
    add_log(f"💥 BIẾN CỐ BẤT NGỜ: {st.session_state.current_event['title']} đã xảy ra!")

def apply_event_effects(da=0.0, ach=0.0, stress=0.0, energy=0.0, se=0.0, sanity=0.0, log_msg="", iq_gain=0.0, mem_gain=0.0):
    chems = st.session_state.chemicals
    chems["dopamine"] = max(0.0, min(100.0, chems["dopamine"] + da))
    chems["acetylcholine"] = max(0.0, min(100.0, chems["acetylcholine"] + ach))
    chems["stress"] = max(0.0, min(100.0, chems["stress"] + stress))
    chems["energy"] = max(0.0, min(100.0, chems["energy"] + energy))
    chems["serotonin"] = max(0.0, min(100.0, chems["serotonin"] + se))
    chems["sanity"] = max(0.0, min(100.0, chems["sanity"] + sanity))

    st.session_state.stats["iq"] += iq_gain
    st.session_state.stats["memory"] = min(st.session_state.stats["max_memory"], st.session_state.stats["memory"] + mem_gain)
    if log_msg:
        add_log(f"🔔 {log_msg}")

def apply_magnetic_storm(val, msg):
    grid = st.session_state.neuron_grid
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            for z in range(GRID_SIZE):
                if grid[x][y][z]["type"] != "Empty":
                    grid[x][y][z]["charge"] = min(1.0, grid[x][y][z]["charge"] + val)
    add_log(f"🌀 Bão từ trường: {msg}")

def add_log(msg):
    st.session_state.game_log.append(msg)
    if len(st.session_state.game_log) > 40:
        st.session_state.game_log.pop(0)

def record_history(tick, chems):
    hist = st.session_state.history_data
    hist["tick"].append(tick)
    hist["sanity"].append(chems["sanity"])
    hist["energy"].append(chems["energy"])
    hist["dopamine"].append(chems["dopamine"])
    hist["stress"].append(chems["stress"])
    hist["norepinephrine"].append(chems.get("norepinephrine", 10.0))
    hist["gaba"].append(chems.get("gaba", 30.0))
    hist["csi"].append(st.session_state.get("csi", 0.0))
    hist["pdi"].append(st.session_state.get("pdi", 0.0))
    hist["vpi"].append(st.session_state.get("vpi", 80.0))

    for k in hist:
        if len(hist[k]) > 60:
            hist[k].pop(0)

def check_mission_statuses():
    missions = st.session_state.missions
    grid = st.session_state.neuron_grid
    chems = st.session_state.chemicals
    stats = st.session_state.stats

    has_sensory = False
    has_motor = False
    active_count = 0

    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            for z in range(GRID_SIZE):
                t = grid[x][y][z]["type"]
                if t == "Sensory":
                    has_sensory = True
                if t == "Motor":
                    has_motor = True
                if t != "Empty":
                    active_count += 1

    if missions["reflex"]["status"] == "In Progress" and has_sensory and has_motor:
        missions["reflex"]["status"] = "Completed"
        add_log("🎉 NHIỆM VỤ HOÀN THÀNH: [⚡ Cung Phản Xạ Sinh Học 3D]! Đã nhận thưởng +100 MB Trí nhớ.")

    if missions["loop"]["status"] == "In Progress" and active_count >= 6:
        missions["loop"]["status"] = "Completed"
        add_log("🎉 NHIỆM VỤ HOÀN THÀNH: [🧠 Vòng Lặp Phản Hồi Tự Trị 3D]! Đã nhận thưởng +300 IQ.")

    if missions["zen"]["status"] == "In Progress" and chems["stress"] < 5.0 and chems["sanity"] > 95.0:
        missions["zen"]["status"] = "Completed"
        add_log("🎉 NHIỆM VỤ HOÀN THÀNH: [🧘 Thiền Tĩnh Tâm Trị Liệu]! Đã nhận thưởng +40 Dopamine & +40 Serotonin.")

    if missions["marathon"]["status"] == "In Progress" and stats["iq"] >= 500.0:
        missions["marathon"]["status"] = "Completed"
        add_log("🎉 NHIỆM VỤ HOÀN THÀNH: [🏆 Chạy Đua Nhận Thức Siêu Phàm]! Đã nhận thưởng +50 Acetylcholine & +200 MB Trí nhớ.")

def run_simulation_tick():
    grid = st.session_state.neuron_grid
    chems = st.session_state.chemicals
    upgrades = st.session_state.upgrades
    mode = st.session_state.get("game_mode", "Normal")
    genes = st.session_state.get("active_genes", [])
    buffs = st.session_state.active_buffs

    st.session_state.stats["ticks"] += 1
    ticks = st.session_state.stats["ticks"]

    cooldowns = st.session_state.get("cooldowns", {"doping": 0, "ssri": 0, "focus": 0, "rtms": 0, "opto": 0, "cortisol": 0, "propranolol": 0, "sprouting": 0, "vns": 0})
    for k in cooldowns:
        if cooldowns[k] > 0:
            cooldowns[k] -= 1

    for k in buffs:
        if buffs[k] > 0:
            buffs[k] -= 1

    cycle_time = ticks % 40
    if cycle_time < 30:
        st.session_state.stats["circadian_cycle"] = "Day"
        chems["melatonin"] = max(5.0, chems["melatonin"] - 1.0)
    else:
        st.session_state.stats["circadian_cycle"] = "Night"
        chems["melatonin"] = min(100.0, chems["melatonin"] + 4.0)

    if upgrades.get("pfc", 0) >= 1 and st.session_state.current_event:
        evt = st.session_state.current_event
        best_choice = evt["choices"][0]
        best_choice["apply"]()
        add_log(f"🤖 [Vỏ Não Trước Trán PFC] Tự động giải quyết biến cố tối ưu: '{best_choice['label']}'")
        st.session_state.current_event = None

    if not st.session_state.current_event and random.random() < 0.03:
        trigger_random_event()

    if upgrades.get("pruning", 0) >= 1:
        pruned_count = 0
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                for z in range(GRID_SIZE):
                    cell = grid[x][y][z]
                    if cell["type"] == "Interneuron" and cell["last_fired"] != -1:
                        if (ticks - cell["last_fired"]) > 15:
                            refund = int(15 * 0.75)
                            st.session_state.stats["memory"] = min(st.session_state.stats["max_memory"], st.session_state.stats["memory"] + refund)
                            grid[x][y][z] = {
                                "type": "Empty",
                                "charge": 0.0,
                                "threshold": 0.5,
                                "fire_rate": 0.0,
                                "last_fired": -1,
                                "direction": "All",
                                "weight": 1.0,
                                "amyloid_plaque": False
                            }
                            pruned_count += 1
        if pruned_count > 0:
            add_log(f"✂️ [Cắt tỉa 3D] Đã tự động cắt tỉa {pruned_count} liên kết nơ-ron nhàn rỗi (>15 ticks) và hoàn phí +75% MB.")

    nutrients_decay = 1.0 if upgrades.get("blood_brain_barrier", 0) >= 1 else 1.5
    chems["neuro_nutrients"] = max(0.0, min(100.0, chems.get("neuro_nutrients", 80.0) - nutrients_decay))

    energy_generation = 4.0 + upgrades["brainstem"] * 2.0
    if chems.get("neuro_nutrients", 80.0) < 30.0:
        energy_generation *= 0.5

    neuron_count = 0
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            for z in range(GRID_SIZE):
                if grid[x][y][z]["type"] != "Empty":
                    neuron_count += 1

    metabolic_cost = 1.0 + (neuron_count * 0.4)
    if "CHRNA7" in genes:
        metabolic_cost *= 1.15

    norepi_val = chems.get("norepinephrine", 10.0)
    metabolic_cost += (norepi_val / 100.0) * 3.0

    if buffs.get("somatosensory_gating", 0) > 0:
        metabolic_cost *= 0.5

    max_energy = 100.0
    if upgrades.get("glycogen_shunt", 0) == 1:
        max_energy = 150.0
    if "PGC-1alpha" in genes:
        max_energy += 40.0

    if chems["energy"] < 15.0 and st.session_state.stats["glycogen_pool"] > 0:
        chems["energy"] = min(max_energy, chems["energy"] + 30.0)
        st.session_state.stats["glycogen_pool"] = max(0.0, st.session_state.stats["glycogen_pool"] - 30.0)
        add_log("🔋 [Glycogen Shunt 3D] Năng lượng dưới 15%! Tự động xuất kho Glycogen khẩn cấp từ tế bào hình sao (+30 Energy).")

    chems["energy"] = max(0.0, min(max_energy, chems["energy"] + energy_generation - metabolic_cost))

    if chems["energy"] <= 0.0:
        add_log("⚠️ Cảnh báo: Bộ não cạn kiệt Glucose và Oxy! Không thể truyền tín hiệu.")

    da_delta = 0.0
    se_delta = 0.0
    ach_delta = 0.0
    gaba_delta = 0.0

    if upgrades["brainstem"] >= 2: da_delta += 1.0
    if upgrades["brainstem"] >= 3: da_delta += 1.5
    if upgrades["cerebellum"] >= 2: se_delta += 1.0
    if upgrades["cerebellum"] >= 3: se_delta += 1.5
    if upgrades["hippocampus"] >= 2: ach_delta += 1.0
    if upgrades["hippocampus"] >= 3: ach_delta += 1.5

    if buffs.get("doping", 0) > 0: da_delta += 5.0
    if buffs.get("ssri", 0) > 0: se_delta += 3.0
    if buffs.get("focus", 0) > 0: ach_delta += 4.0

    if buffs.get("tyrosine", 0) > 0: da_delta += 3.0
    if buffs.get("tryptophan", 0) > 0: se_delta += 2.0
    if buffs.get("choline", 0) > 0: ach_delta += 2.5
    if buffs.get("glutamate", 0) > 0: gaba_delta += 3.0

    decay_rate = 0.08
    if "MAOA" in genes:
        decay_rate *= 0.7

    ach_mult = 1.25 if "CHRNA7" in genes else 1.0

    chems["dopamine"] = max(0.0, min(100.0, chems["dopamine"] + da_delta + (50.0 - chems["dopamine"]) * decay_rate))
    chems["serotonin"] = max(0.0, min(100.0, chems["serotonin"] + se_delta + (50.0 - chems["serotonin"]) * decay_rate))
    chems["acetylcholine"] = max(0.0, min(100.0, chems["acetylcholine"] + ach_delta + (50.0 - chems["acetylcholine"]) * 0.08 * ach_mult))
    chems["gaba"] = max(0.0, min(100.0, chems["gaba"] + gaba_delta + (30.0 - chems["gaba"]) * 0.10))

    if st.session_state.stats.get("sleep_state", False):
        chems["energy"] = min(max_energy, chems["energy"] + 15.0)
        chems["sanity"] = min(100.0, chems["sanity"] + 8.0)
        chems["stress"] = max(0.0, chems["stress"] - 12.0)

        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                for z in range(GRID_SIZE):
                    grid[x][y][z]["charge"] = 0.0

        chems["dopamine"] += (50.0 - chems["dopamine"]) * 0.35
        chems["serotonin"] += (50.0 - chems["serotonin"]) * 0.35
        chems["neuro_inflammation"] = max(5.0, chems["neuro_inflammation"] - 6.0)
        chems["norepinephrine"] = max(5.0, chems.get("norepinephrine", 10.0) - 8.0)

        if cycle_time == 0:
            st.session_state.stats["sleep_state"] = False
            add_log("🌞 [Circadian] Mặt trời lên! Bộ não tự động tỉnh giấc, khôi phục hệ thống kích thích.")

        update_3d_indices(grid, chems, upgrades)
        record_history(ticks, chems)
        return

    sensory_fires = 0
    visual_boost_active = False
    visual_spark = st.session_state.get("visual_spark", None)

    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            for z in range(GRID_SIZE):
                cell = grid[x][y][z]
                if cell["type"] == "Sensory":
                    drd2_mult = 1.5 if "DRD2" in genes else 1.0
                    boost = 1.0 + (chems["dopamine"] / 100.0) * drd2_mult
                    thalamus_level = upgrades.get("thalamus", 0)
                    boost *= (1.0 + thalamus_level * 0.2)

                    norepi_boost = 1.0 + (chems.get("norepinephrine", 10.0) / 100.0) * 0.4
                    boost *= norepi_boost

                    if visual_spark and visual_spark["pos"] == (x, y, z):
                        if cell.get("direction", "All") == visual_spark["dir"]:
                            boost *= 2.0
                            visual_boost_active = True
                            add_log(f"👁️ [Thùy Chẩm] Khớp hướng thành công tại [{x+1},{y+1},{z+1}]! Sensory nhận gia tốc 2.0x.")

                    cell["charge"] += cell["fire_rate"] * boost
                    if cell["charge"] >= cell["threshold"]:
                        sensory_fires += 1

    st.session_state.visual_boost_active = visual_boost_active

    next_charges = [[[grid[x][y][z]["charge"] for z in range(GRID_SIZE)] for y in range(GRID_SIZE)] for x in range(GRID_SIZE)]
    signals_fired = 0
    fired_cells = set()

    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            for z in range(GRID_SIZE):
                cell = grid[x][y][z]
                if cell["type"] != "Empty" and cell["charge"] >= cell["threshold"]:
                    fired_cells.add((x, y, z))
                    cell["last_fired"] = ticks

                    gate = st.session_state.get("spatial_gate", None)
                    if upgrades.get("parietal_lobe", 0) >= 1 and gate == (x, y, z):
                        chems["stress"] = max(0.0, chems["stress"] - 20.0)
                        st.session_state.active_buffs["somatosensory_gating"] = 3
                        add_log(f"🧭 [Thùy Đỉnh] Luồng điện tích khớp vị trí Gating [{x+1},{y+1},{z+1}]! Giải tỏa stress lập tức và kích hoạt Somatosensory Gating (giảm 50% tiêu hao năng lượng).")
                    carry_over = 0.05 * upgrades["plasticity"] if cell["type"] == "Interneuron" else 0.0
                    next_charges[x][y][z] = carry_over

                    dir_deltas = {
                        "Up": [(0, 1, 0)],
                        "Down": [(0, -1, 0)],
                        "Left": [(-1, 0, 0)],
                        "Right": [(1, 0, 0)],
                        "Front": [(0, 0, 1)],
                        "Back": [(0, 0, -1)],
                        "All": [
                            (0, 1, 0), (0, -1, 0),
                            (-1, 0, 0), (1, 0, 0),
                            (0, 0, 1), (0, 0, -1)
                        ]
                    }
                    allowed_deltas = dir_deltas.get(cell.get("direction", "All"), dir_deltas["All"])

                    neighbors = []
                    for dx, dy, dz in allowed_deltas:
                        nx, ny, nz = x + dx, y + dy, z + dz
                        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and 0 <= nz < GRID_SIZE:
                            if grid[nx][ny][nz]["type"] != "Empty":
                                neighbors.append((nx, ny, nz))

                    if neighbors:
                        shank3_bonus = 0.15 if "SHANK3" in genes else 0.0
                        signal_efficiency = 0.35 + (upgrades["myelin"] * 0.05) + shank3_bonus

                        if mode == "Epilepsy":
                            signal_efficiency *= 1.35

                        cell_weight = cell.get("weight", 1.0)
                        if cell.get("amyloid_plaque", False):
                            cell_weight *= 0.5
                        transfer_charge = (cell["charge"] * signal_efficiency * cell_weight) / len(neighbors)

                        for nx, ny, nz in neighbors:
                            next_charges[nx][ny][nz] = min(1.0, next_charges[nx][ny][nz] + transfer_charge)
                            if upgrades["plasticity"] > 0 and grid[nx][ny][nz]["charge"] > 0.3 and chems.get("neuro_nutrients", 80.0) >= 30.0:
                                learn_rate = 0.015 if "BDNF" in genes else 0.01
                                grid[nx][ny][nz]["threshold"] = max(0.2, grid[nx][ny][nz]["threshold"] - learn_rate)

                    signals_fired += 1

    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            for z in range(GRID_SIZE):
                grid[x][y][z]["charge"] = next_charges[x][y][z]

    motor_yield_iq = 0.0
    motor_yield_mem = 0.0
    motor_fired_count = 0

    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            for z in range(GRID_SIZE):
                cell = grid[x][y][z]
                if cell["type"] == "Motor" and (x, y, z) in fired_cells:
                    motor_fired_count += 1
                    iq_multiplier = 1.0 + (upgrades["cortex"] * 0.6)
                    focus_bonus = 1.0 + (chems["acetylcholine"] / 100.0)
                    visual_iq_bonus = 2.0 if st.session_state.get("visual_boost_active", False) else 1.0
                    drd4_iq_mult = 2.0 if "DRD4" in genes else 1.0
                    comt_met_mult = 1.30 if "COMT-Met" in genes else 1.0

                    motor_yield_iq += 5.0 * iq_multiplier * focus_bonus * visual_iq_bonus * drd4_iq_mult * comt_met_mult

                    mem_multiplier = 1.0 + (upgrades["hippocampus"] * 0.5)
                    is_resonant = (upgrades.get("temporal_lobe", 0) >= 1 and 400 <= st.session_state.get("auditory_freq", 0) <= 500)
                    auditory_resonance_mult = 3.0 if is_resonant else 1.0

                    motor_yield_mem += 2.0 * mem_multiplier * auditory_resonance_mult

    if motor_fired_count >= 3:
        motor_yield_mem *= 1.5
        chems["stress"] = max(0.0, chems["stress"] - 15.0)
        add_log("🔥 [DỒNG BỘ NHẬN THỨC] Kích hoạt Cognitive Sync Combo! +50% sản lượng Trí nhớ và triệt tiêu -15% Stress.")

    if motor_fired_count > 0:
        drd4_da_reward = 16.0 if "DRD4" in genes else 8.0
        chems["dopamine"] = min(100.0, chems["dopamine"] + drd4_da_reward)
        chems["sanity"] = min(100.0, chems["sanity"] + 3.0)

        st.session_state.stats["iq"] += motor_yield_iq
        st.session_state.stats["memory"] = min(st.session_state.stats["max_memory"], st.session_state.stats["memory"] + motor_yield_mem)

        add_log(f"💪 Xung động vận động phát hỏa thành công! Thu hoạch +{motor_yield_iq:.1f} IQ, +{motor_yield_mem:.1f} MB Trí nhớ.")
        st.session_state.audio_trigger = {"sensory": sensory_fires, "motor": motor_fired_count}

    if mode == "Alzheimer":
        if ticks % 10 == 0:
            apoe4_speed = 0.08 if "APOE4" in genes else 0.04
            degraded_count = 0
            for x in range(GRID_SIZE):
                for y in range(GRID_SIZE):
                    for z in range(GRID_SIZE):
                        cell = grid[x][y][z]
                        if cell["type"] != "Empty" and cell["threshold"] < 0.9:
                            cell["threshold"] = min(0.9, cell["threshold"] + apoe4_speed)
                            degraded_count += 1
            if degraded_count > 0:
                add_log(f"👵 [Alzheimer xơ hóa] Mạch nơ-ron chai lỳ điện thế! Tăng ngưỡng kích hoạt {degraded_count} tế bào (+{apoe4_speed:.2f}).")

            empty_plaques = []
            for x in range(GRID_SIZE):
                for y in range(GRID_SIZE):
                    for z in range(GRID_SIZE):
                        if grid[x][y][z]["type"] != "Empty" and not grid[x][y][z].get("amyloid_plaque", False):
                            empty_plaques.append((x, y, z))
            if empty_plaques:
                px, py, pz = random.choice(empty_plaques)
                grid[px][py][pz]["amyloid_plaque"] = True
                add_log(f"🧬 [Amyloid Plaques] Xuất hiện mảng xơ hóa beta-amyloid bám phủ tế bào [{px+1},{py+1},{pz+1}], giảm 50% hiệu suất dẫn truyền.")

    if chems["acetylcholine"] > 60.0 and chems["serotonin"] > 60.0:
        clear_prob = 0.40 if "TREM2" in genes else 0.20
        if random.random() < clear_prob:
            plaque_cells = []
            for x in range(GRID_SIZE):
                for y in range(GRID_SIZE):
                    for z in range(GRID_SIZE):
                        if grid[x][y][z].get("amyloid_plaque", False):
                            plaque_cells.append((x, y, z))
            if plaque_cells:
                cx, cy, cz = random.choice(plaque_cells)
                grid[cx][cy][cz]["amyloid_plaque"] = False
                add_log(f"✨ [Microglia Phục Hồi] Đại thực bào dọn dẹp sạch mảng bám Amyloid tại nơ-ron [{cx+1},{cy+1},{cz+1}]!")

    if chems["neuro_inflammation"] > 80.0:
        degraded = 0
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                for z in range(GRID_SIZE):
                    cell = grid[x][y][z]
                    if cell["type"] != "Empty" and cell["threshold"] < 0.9:
                        cell["threshold"] = min(0.9, cell["threshold"] + 0.02)
                        degraded += 1
        if degraded > 0:
            add_log(f"🔥 [Bão Cytokine 3D] Viêm thần kinh cực cao (>80%) gây chai lỳ, xơ hóa và tăng ngưỡng kích hoạt {degraded} tế bào (+0.02)!")

    if upgrades.get("dentate_gyrus", 0) >= 1:
        if st.session_state.stats["memory"] >= 30.0 and chems["serotonin"] > 60.0:
            if random.random() < 0.25:
                empty_cells = []
                for x in range(GRID_SIZE):
                    for y in range(GRID_SIZE):
                        for z in range(GRID_SIZE):
                            if grid[x][y][z]["type"] == "Empty":
                                empty_cells.append((x, y, z))
                if empty_cells:
                    sp_x, sp_y, sp_z = random.choice(empty_cells)
                    st.session_state.stats["memory"] -= 15.0
                    grid[sp_x][sp_y][sp_z] = {
                        "type": "Interneuron",
                        "charge": 0.0,
                        "threshold": 0.5,
                        "fire_rate": 0.0,
                        "last_fired": -1,
                        "direction": "All",
                        "weight": 1.0,
                        "amyloid_plaque": False
                    }
                    add_log(f"🌱 [Hải Mã Neurogenesis 3D] Thùy răng (Dentate Gyrus) tự động sản sinh tế bào liên kết mới tại [{sp_x+1},{sp_y+1},{sp_z+1}]! (-15 MB Memory)")

    # 3D Cellular Mitosis and Mutation Simulator
    if st.session_state.get("cellular_evolution", True):
        # We only divide if total energy > 50% and nutrients > 50%
        if chems["energy"] > 50.0 and chems.get("neuro_nutrients", 80.0) > 50.0:
            # Look for active cells to divide
            active_dividers = []
            for x in range(GRID_SIZE):
                for y in range(GRID_SIZE):
                    for z in range(GRID_SIZE):
                        cell = grid[x][y][z]
                        if cell["type"] != "Empty":
                            # Check for adjacent empty spots
                            for dx, dy, dz in [(0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)]:
                                nx, ny, nz = x + dx, y + dy, z + dz
                                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and 0 <= nz < GRID_SIZE:
                                    if grid[nx][ny][nz]["type"] == "Empty":
                                        active_dividers.append((x, y, z, nx, ny, nz))

            # 10% chance of division per tick
            if active_dividers and random.random() < 0.10:
                px, py, pz, cx, cy, cz = random.choice(active_dividers)
                parent = grid[px][py][pz]

                # Copy parent state
                child_type = parent["type"]
                child_direction = parent.get("direction", "All")
                child_weight = parent.get("weight", 1.0)

                # 30% mutation chance
                is_mutated = False
                if random.random() < 0.30:
                    child_type = random.choice(["Sensory", "Interneuron", "Motor"])
                    child_direction = random.choice(["All", "Up", "Right", "Down", "Left", "Front", "Back"])
                    child_weight = random.choice([1.0, 2.0, 3.0])
                    is_mutated = True

                grid[cx][cy][cz] = {
                    "type": child_type,
                    "charge": 0.0,
                    "threshold": 0.4 if child_type == "Sensory" else (0.6 if child_type == "Motor" else 0.5),
                    "fire_rate": 0.25 if child_type == "Sensory" else 0.0,
                    "last_fired": -1,
                    "direction": child_direction,
                    "weight": child_weight,
                    "amyloid_plaque": False
                }

                # Metabolic costs
                chems["energy"] = max(0.0, chems["energy"] - 10.0)
                chems["neuro_nutrients"] = max(0.0, chems.get("neuro_nutrients", 80.0) - 5.0)

                mutation_text = " (Đột biến!)" if is_mutated else ""
                add_log(f"🧬 [Mitosis] Tế bào {parent['type']} [{px+1},{py+1},{pz+1}] phân chia! Con: {child_type} [{cx+1},{cy+1},{cz+1}]{mutation_text} (-10% Năng lượng)")

    if mode == "Schizophrenia" and ticks % 8 == 0:
        non_empty = []
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                for z in range(GRID_SIZE):
                    if grid[x][y][z]["type"] != "Empty":
                        non_empty.append((x, y, z))
        if non_empty:
            hx, hy, hz = random.choice(non_empty)
            grid[hx][hy][hz]["charge"] = grid[hx][hy][hz]["threshold"]
            add_log(f"📢 [Ảo thanh Phân liệt] Kích phát ảo giác kích động điện cực đại tại [{hx+1},{hy+1},{hz+1}]!")

    if mode == "Parkinson" and chems["dopamine"] < 40.0:
        if random.random() < 0.30:
            motor_cells = []
            for x in range(GRID_SIZE):
                for y in range(GRID_SIZE):
                    for z in range(GRID_SIZE):
                        if grid[x][y][z]["type"] == "Motor":
                            motor_cells.append((x, y, z))
            if motor_cells:
                tr_x, tr_y, tr_z = random.choice(motor_cells)
                chems["energy"] = max(0.0, chems["energy"] - 5.0)
                grid[tr_x][tr_y][tr_z]["charge"] = 0.0
                add_log(f"🤝 [Parkinson Run Giật] Thiếu Dopamine trầm trọng gây run giật vô thức tại [{tr_x+1},{tr_y+1},{tr_z+1}]! Hao hụt -5 Glucose.")

    if mode == "ADHD":
        chems["dopamine"] = max(10.0, min(90.0, chems["dopamine"] + random.choice([-15.0, 15.0])))
        chems["acetylcholine"] = max(0.0, min(100.0, chems["acetylcholine"] + (50.0 - chems["acetylcholine"]) * 0.12))

    epilepsy_mult = 2.0 if (mode == "Epilepsy") else 1.0
    if chems.get("gaba", 30.0) >= 70.0:
        epilepsy_mult = 1.0

    fire_stress = signals_fired * 1.5 * epilepsy_mult
    if mode == "Mania":
        fire_stress *= 2.0

    if chems.get("gaba", 30.0) >= 70.0:
        fire_stress *= 0.6

    stress_clearance = 1.5 + (upgrades["cerebellum"] * 1.0)
    if mode == "Epilepsy" and "GABRA1" in genes:
        stress_clearance *= 0.7
    if "COMT-Met" in genes:
        stress_clearance *= 0.7

    if buffs.get("ssri", 0) > 0:
        fire_stress *= 0.5

    amygdala_level = upgrades.get("amygdala", 0)
    fire_stress *= max(0.1, 1.0 - amygdala_level * 0.15)

    chems["stress"] = max(0.0, min(100.0, chems["stress"] + fire_stress - stress_clearance))

    serotonin_dampening = chems["serotonin"] * 0.1
    effective_stress = max(0.0, chems["stress"] - serotonin_dampening)

    drd4_sanity_mult = 2.0 if ("DRD4" in genes and chems["dopamine"] < 30.0) else 1.0
    if "DRD2" in genes and chems["stress"] > 50.0:
        drd4_sanity_mult *= 1.5

    if effective_stress > 60.0:
        sanity_damage = (effective_stress - 60.0) * 0.35 * drd4_sanity_mult
        chems["sanity"] = max(0.0, chems["sanity"] - sanity_damage)
        add_log(f"⚡ Căng thẳng cực độ gây tổn hại myelin và nơ-ron! (-{sanity_damage:.1f} Tỉnh táo)")
    else:
        healing = 0.5 + (chems["serotonin"] * 0.02)
        if buffs.get("oxytocin", 0) > 0:
            healing *= 2.0
        if mode == "Schizophrenia":
            healing *= 0.7
        chems["sanity"] = max(0.0, min(100.0, chems["sanity"] + healing))

    if mode == "Mania":
        chems["dopamine"] = min(100.0, chems["dopamine"] + 1.5)
        chems["sanity"] = max(0.0, chems["sanity"] - 1.0)

    inflammation_gain = (signals_fired * 0.4) + (effective_stress > 50.0 and (effective_stress - 50.0) * 0.2 or 0.0)
    if upgrades.get("blood_brain_barrier", 0) >= 1:
        inflammation_gain *= 0.5
    chems["neuro_inflammation"] = max(0.0, min(100.0, chems["neuro_inflammation"] + inflammation_gain - 1.0))

    if chems["neuro_inflammation"] > 80.0:
        cyto_decay = (chems["neuro_inflammation"] - 80.0) * 0.4
        chems["sanity"] = max(0.0, chems["sanity"] - cyto_decay)

    norepi_gain = (signals_fired * 0.5) + (chems["stress"] * 0.1)
    norepi_clearance = 0.75 if mode == "Mania" else 1.5
    chems["norepinephrine"] = max(0.0, min(100.0, chems.get("norepinephrine", 10.0) + norepi_gain - norepi_clearance))

    panic_threshold = 100.0 if "ADRA2A" in genes else 90.0
    if chems.get("norepinephrine", 10.0) >= panic_threshold:
        damage = 9.0 if "ADRA2A" in genes else 15.0
        chems["sanity"] = max(0.0, chems["sanity"] - damage)

        non_empty = []
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                for z in range(GRID_SIZE):
                    if grid[x][y][z]["type"] != "Empty":
                        non_empty.append((x, y, z))
        if non_empty:
            for _ in range(min(3, len(non_empty))):
                pr, py, pz = random.choice(non_empty)
                grid[pr][py][pz]["charge"] = 0.0
        add_log(f"🚨 [HOẢNG LOẠN 3D] Norepinephrine ({chems['norepinephrine']:.1f}%) vượt ngưỡng {panic_threshold}%! Gây mất -{damage:.1f} Sanity và đóng băng 3 nơ-ron.")
        chems["norepinephrine"] = 50.0

    if chems["sanity"] <= 0.0:
        st.session_state.stats["burnout_count"] += 1
        st.session_state.stats["burnout_streak"] = 0
        chems["sanity"] = 25.0
        chems["stress"] = 10.0
        chems["norepinephrine"] = 20.0

        damaged = 0
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                for z in range(GRID_SIZE):
                    if grid[x][y][z]["type"] == "Interneuron" and random.random() < 0.4:
                        grid[x][y][z] = {"type": "Empty", "charge": 0.0, "threshold": 0.5, "fire_rate": 0.0, "last_fired": -1, "direction": "All", "weight": 1.0, "amyloid_plaque": False}
                        damaged += 1
        add_log(f"⚠️ [SỤP ĐỔ TÂM THẦN 3D] Trạng thái Tỉnh táo sụt giảm về không! Kích phát cơn hoảng loạn xóa sổ {damaged} liên kết nơ-ron liên kết.")
    else:
        if chems["sanity"] >= 80.0:
            st.session_state.stats["burnout_streak"] += 1
            if st.session_state.stats["burnout_streak"] > st.session_state.stats["max_streak"]:
                st.session_state.stats["max_streak"] = st.session_state.stats["burnout_streak"]

    if st.session_state.stats["iq"] > st.session_state.stats["high_score_iq"]:
        st.session_state.stats["high_score_iq"] = st.session_state.stats["iq"]
    if st.session_state.stats["memory"] > st.session_state.stats["max_memory"]:
        st.session_state.stats["max_memory"] = st.session_state.stats["memory"]

    check_mission_statuses()
    update_3d_indices(grid, chems, upgrades)
    record_history(ticks, chems)

def update_3d_indices(grid, chems, upgrades):
    total_neurons = 0
    active_charged = 0
    total_interneurons = 0
    modified_interneurons = 0

    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            for z in range(GRID_SIZE):
                cell = grid[x][y][z]
                if cell["type"] != "Empty":
                    total_neurons += 1
                    if cell["charge"] >= cell["threshold"] * 0.5:
                        active_charged += 1
                if cell["type"] == "Interneuron":
                    total_interneurons += 1
                    if cell["threshold"] != 0.5:
                        modified_interneurons += 1

    st.session_state.csi = (active_charged / max(1, total_neurons)) * 100.0
    st.session_state.pdi = (modified_interneurons / max(1, total_interneurons)) * 100.0

    bbb_mult = 1.2 if upgrades.get("blood_brain_barrier", 0) >= 1 else 0.8
    st.session_state.vpi = min(100.0, chems.get("neuro_nutrients", 80.0) * bbb_mult)

import plotly.graph_objects as go
import streamlit as st
import random

# Visual/Audio Feedback Synthesizer Element via Web Audio API
audio_html = ""
if "audio_trigger" in st.session_state and st.session_state.audio_trigger:
    trig = st.session_state.audio_trigger
    sound_cmds = []
    if trig.get("sensory", 0) > 0:
        sound_cmds.append("playBeep(440, 'triangle', 0.12);")
    if trig.get("motor", 0) > 0:
        sound_cmds.append("playBeep(880, 'sine', 0.18);")

    if sound_cmds:
        joined_js = " ".join(sound_cmds)
        audio_html = f"""
        <script>
            function playBeep(frequency, type, duration) {{
                try {{
                    var context = new (window.AudioContext || window.webkitAudioContext)();
                    var oscillator = context.createOscillator();
                    var gainNode = context.createGain();

                    oscillator.type = type;
                    oscillator.frequency.value = frequency;

                    gainNode.gain.setValueAtTime(0.06, context.currentTime);
                    gainNode.gain.exponentialRampToValueAtTime(0.00001, context.currentTime + duration);

                    oscillator.connect(gainNode);
                    gainNode.connect(context.destination);

                    oscillator.start();
                    oscillator.stop(context.currentTime + duration);
                }} catch(e) {{ console.error(e); }}
            }}
            setTimeout(function() {{
                {joined_js}
            }}, 100);
        </script>
        """
    st.session_state.audio_trigger = None

if audio_html:
    st.components.v1.html(audio_html, height=0, width=0)

# Initialize Session
init_game_state()

# Visualizing 3D Interactive WebGPU-Style Brain Utility
def render_3d_brain(grid, selected_cell):
    x_nodes = []
    y_nodes = []
    z_nodes = []

    # Nested layers lists to simulate multi-pass WebGPU bloom shaders
    colors_core = []
    sizes_core = []

    colors_glow = []
    sizes_glow = []

    colors_halo = []
    sizes_halo = []

    texts = []

    sel_x, sel_y, sel_z = selected_cell

    # WebGPU vibrant emissive palettes
    color_map = {
        "Empty": "#1A1C1E",        # deep dim gray
        "Sensory": "#00F0FF",      # electric cyan
        "Interneuron": "#39FF14",  # high-energy green
        "Motor": "#FF007F"         # hot magenta
    }

    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            for z in range(GRID_SIZE):
                cell = grid[x][y][z]
                ctype = cell["type"]
                charge = cell["charge"]
                threshold = cell["threshold"]
                is_firing = cell["type"] != "Empty" and charge >= threshold

                x_nodes.append(x)
                y_nodes.append(y)
                z_nodes.append(z)

                # Default styling
                base_color = color_map.get(ctype, "#1A1C1E")

                # Active firing state flashes white-hot
                if is_firing:
                    core_c = "#FFFFFF"
                    glow_c = "#FFCC00"
                    halo_c = "rgba(255, 170, 0, 0.3)"

                    core_s = 14
                    glow_s = 28
                    halo_s = 48
                else:
                    core_c = base_color
                    glow_c = base_color
                    # Convert hex colors to semi-transparent rgba strings for outer bloom layers
                    if ctype == "Sensory":
                        halo_c = "rgba(0, 240, 255, 0.25)"
                        glow_c = "rgba(0, 240, 255, 0.45)"
                    elif ctype == "Interneuron":
                        halo_c = "rgba(57, 255, 20, 0.25)"
                        glow_c = "rgba(57, 255, 20, 0.45)"
                    elif ctype == "Motor":
                        halo_c = "rgba(255, 0, 127, 0.25)"
                        glow_c = "rgba(255, 0, 127, 0.45)"
                    else:
                        halo_c = "rgba(40, 40, 40, 0.1)"
                        glow_c = "rgba(40, 40, 40, 0.2)"

                    core_s = 9 if ctype != "Empty" else 4
                    glow_s = 18 if ctype != "Empty" else 0
                    halo_s = 28 if ctype != "Empty" else 0

                # Highlight selected node by styling
                if (x, y, z) == (sel_x, sel_y, sel_z):
                    core_s += 6
                    glow_s += 12
                    halo_s += 18
                    if not is_firing:
                        core_c = "#FFFFFF"

                colors_core.append(core_c)
                sizes_core.append(core_s)

                colors_glow.append(glow_c)
                sizes_glow.append(glow_s)

                colors_halo.append(halo_c)
                sizes_halo.append(halo_s)

                texts.append(f"Node [{x+1}, {y+1}, {z+1}]<br>Loại: {ctype}<br>Điện lượng: {charge:.2f}/{threshold:.2f}")

    # Trace 1: Emissive Core Layer
    trace_core = go.Scatter3d(
        x=x_nodes, y=y_nodes, z=z_nodes,
        mode="markers",
        marker=dict(
            size=sizes_core,
            color=colors_core,
            opacity=1.0,
            line=dict(color='#000000', width=1)
        ),
        text=texts,
        hoverinfo="text",
        name="Neural Core"
    )

    # Trace 2: Mid Volumetric Glow Layer
    trace_glow = go.Scatter3d(
        x=x_nodes, y=y_nodes, z=z_nodes,
        mode="markers",
        marker=dict(
            size=sizes_glow,
            color=colors_glow,
            opacity=0.45,
        ),
        hoverinfo="none",
        showlegend=False,
        name="Inner Glow"
    )

    # Trace 3: Wide Atmospheric Bloom Layer
    trace_halo = go.Scatter3d(
        x=x_nodes, y=y_nodes, z=z_nodes,
        mode="markers",
        marker=dict(
            size=sizes_halo,
            color=colors_halo,
            opacity=0.15,
        ),
        hoverinfo="none",
        showlegend=False,
        name="Outer Bloom"
    )

    # Synapse Lines (Axons)
    edge_x = []
    edge_y = []
    edge_z = []
    edge_colors = []

    dir_deltas = {
        "Up": [(0, 1, 0)],
        "Down": [(0, -1, 0)],
        "Left": [(-1, 0, 0)],
        "Right": [(1, 0, 0)],
        "Front": [(0, 0, 1)],
        "Back": [(0, 0, -1)],
        "All": [
            (0, 1, 0), (0, -1, 0),
            (-1, 0, 0), (1, 0, 0),
            (0, 0, 1), (0, 0, -1)
        ]
    }

    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            for z in range(GRID_SIZE):
                cell = grid[x][y][z]
                if cell["type"] == "Empty":
                    continue
                allowed_deltas = dir_deltas.get(cell.get("direction", "All"), dir_deltas["All"])
                for dx, dy, dz in allowed_deltas:
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and 0 <= nz < GRID_SIZE:
                        if grid[nx][ny][nz]["type"] != "Empty":
                            edge_x.extend([x, nx, None])
                            edge_y.extend([y, ny, None])
                            edge_z.extend([z, nz, None])

    # Plotly 3D synap lines
    trace_edges = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode="lines",
        line=dict(color="rgba(0, 240, 255, 0.4)", width=2),
        hoverinfo="none",
        name="Synap Channels"
    )

    fig = go.Figure(data=[trace_halo, trace_glow, trace_edges, trace_core])
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Trục X (Left-Right)", backgroundcolor="black", gridcolor="#22252A", showbackground=True),
            yaxis=dict(title="Trục Y (Down-Up)", backgroundcolor="black", gridcolor="#22252A", showbackground=True),
            zaxis=dict(title="Trục Z (Back-Front)", backgroundcolor="black", gridcolor="#22252A", showbackground=True),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1)
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor="black",
        plot_bgcolor="black"
    )
    return fig

# WebGPU Sci-fi Holographic HUD Style Injection
st.markdown("""
<style>
    /* Dark Sci-fi / Cyberpunk Biotech HUD */
    .stApp {
        background-color: #030508 !important;
        color: #00F0FF !important;
        font-family: 'Courier New', Courier, monospace !important;
    }

    /* Neon glow container cards */
    div.stButton > button {
        background: linear-gradient(135deg, #020b12 0%, #081a26 100%) !important;
        border: 1px solid #00F0FF !important;
        color: #00F0FF !important;
        box-shadow: 0px 0px 8px rgba(0, 240, 255, 0.3) !important;
        font-weight: bold !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        border-radius: 4px !important;
        transition: all 0.3s ease !important;
    }

    div.stButton > button:hover {
        background: #00F0FF !important;
        color: #000000 !important;
        box-shadow: 0px 0px 20px #00F0FF !important;
        transform: translateY(-2px) !important;
    }

    /* Headers with neon underlines */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        text-shadow: 0px 0px 10px rgba(0, 240, 255, 0.7) !important;
        border-bottom: 1px solid rgba(0, 240, 255, 0.3) !important;
        padding-bottom: 5px !important;
    }

    /* Telemetry Panel board styles */
    .telemetry-board {
        background-color: rgba(2, 8, 14, 0.85);
        border: 1px dashed #00F0FF;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0px;
        box-shadow: inset 0px 0px 15px rgba(0, 240, 255, 0.15), 0px 0px 10px rgba(0, 240, 255, 0.1);
    }

    /* Code serialize & logs inputs styled as tech terminals */
    textarea, input {
        background-color: #010408 !important;
        border: 1px solid #FF007F !important;
        color: #FF007F !important;
        font-family: 'Courier New', Courier, monospace !important;
        box-shadow: 0px 0px 6px rgba(255, 0, 127, 0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

# Title & Info
st.title("⚡ WEBGPU NEURO-EMULATOR ENGINE 3D")
st.write("Hệ thống giả lập mạng nơ-ron sinh học 3D thời gian thực, biên dịch trực tiếp bằng công nghệ hiển thị **WebGPU Emulation Mode**.")

with st.expander("🆕 [WEBGPU DIAGNOSTICS] Nhật Ký Cập Nhật Phiên Bản 6.0.0", expanded=False):
    st.markdown("""
    **🚀 Phiên bản 6.0.0 (Bản nâng cấp mô phỏng không gian 3D WebGPU tối hậu):**
    *   **Hiển thị WebGPU Bloom Shader lặp (Three-Pass Bloom):** Node nơ-ron hỏa phát được phủ 3 lớp lõi phát quang, quầng sáng nhiệt tích và quầng khuếch tán không khí mô phỏng chính xác công nghệ đồ họa thế hệ mới.
    *   **Lưới Nơ-ron Không Gian 3D (4x4x4 - 64 Nodes):** Nâng cấp toàn diện từ lưới 2D tĩnh sang mạng lưới 3D tương tác. Cho phép xoay, thu phóng 3 chiều thông qua Plotly linh hoạt.
    *   **Ba Chỉ Số Gameplay Mới (CSI, PDI, VPI):**
        *   *Cognitive Sync Index (CSI):* Đo lường sự đồng bộ xung điện toàn bộ não bộ 3D.
        *   *Plasticity Density Index (PDI):* Đo lường mật độ thích nghi liên kết (sự thay đổi ngưỡng điện thế của Interneuron).
        *   *Vascular Perfusion Index (VPI):* Đo lường hiệu suất tuần hoàn và nồng độ dưỡng chất trong huyết quản não.
    *   **Hàng Rào Máu Não (Blood-Brain Barrier - BBB) & Dinh Dưỡng Thần Kinh (Neuro-Nutrients):** Bổ sung chỉ số Dinh dưỡng Thần kinh tiêu hao 1.5% mỗi tick. Nếu dinh dưỡng < 30%, hiệu năng sinh năng lượng giảm 50% và chặn cơ chế Hebbian LTP. Bồi bổ Tiền chất dinh dưỡng (Diet Precursors) hồi phục +25% Dinh dưỡng. Nâng cấp Hàng rào máu não (BBB Upgrade) giúp giảm tốc độ hao hụt dinh dưỡng còn 1.0% và giảm 50% lượng Độc tố/Viêm nhiễm phát sinh.
    *   **Bệnh lý học Hưng Cảm (Mania Mode):** Trạng thái bệnh lý thứ 7 mô tả cơn cuồng sảng kích động. Tự động tăng Dopamine cực nhanh (+1.5%/tick), hao hụt Sanity mạnh (-1.0%/tick), giảm 50% tốc độ tự triệt tiêu Norepinephrine (0.75%) và nhân đôi Stress phát sinh từ các xung kích hỏa.
    *   **Liệu pháp Kích thích Dây Thần Kinh Phế Vị (VNS):** Khả năng lâm sàng chủ động mới. Tiêu hao 30 MB bộ nhớ, lập tiếp đặt Stress về 0.0%, hồi phục +20% Sanity và sạc đầy GABA lên 90% (Cooldown 40s).
    """)

tab1, tab2 = st.tabs(["🧠 Game Mô Phỏng Não Bộ 3D", "🤖 Trợ Lý AI VBot1 (Llama & Gemini)"])


webgpu_html_content = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BONSAI BRAIN SIMULATOR 3D</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: #050505;
            color: #E5E7EB;
            font-family: 'Inter', system-ui, sans-serif;
            padding: 20px;
            overflow-x: hidden;
        }

        h1 {
            font-size: 2.25rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            color: #FFFFFF;
            text-transform: uppercase;
        }

        .subtitle {
            font-size: 1.1rem;
            font-weight: 600;
            color: #9CA3AF;
            letter-spacing: 0.5px;
            margin-top: 4px;
            margin-bottom: 12px;
        }

        .description {
            font-size: 0.95rem;
            color: #9CA3AF;
            line-height: 1.5;
            margin-bottom: 20px;
        }

        /* Diagnostics status bar */
        .diagnostics-bar {
            background-color: #0F1113;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 24px;
            font-family: monospace;
        }

        .diag-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }

        .diag-title {
            font-weight: bold;
            color: #FFFFFF;
            font-size: 0.95rem;
            letter-spacing: 0.5px;
        }

        .diag-status {
            color: #39FF14;
            font-weight: bold;
            letter-spacing: 1px;
        }

        .diag-desc {
            font-size: 0.85rem;
            color: #9CA3AF;
            margin-bottom: 8px;
        }

        .progress-track {
            background-color: #1F2327;
            height: 8px;
            border-radius: 4px;
            overflow: hidden;
            width: 100%;
        }

        .progress-fill {
            background-color: #39FF14;
            height: 100%;
            width: 100%;
            box-shadow: 0px 0px 10px #39FF14;
        }

        /* Main grid container */
        .main-layout {
            display: grid;
            grid-template-columns: 1.3fr 1fr;
            gap: 24px;
        }

        @media(max-width: 1024px) {
            .main-layout {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background-color: #0A0B0C;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }

        .card-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #FFFFFF;
            margin-bottom: 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            padding-bottom: 8px;
        }

        /* Telemetry index boards */
        .telemetry-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-bottom: 16px;
        }

        .telemetry-item {
            background-color: #0F1113;
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 6px;
            padding: 12px;
            text-align: center;
        }

        .telemetry-label {
            font-size: 0.75rem;
            color: #9CA3AF;
            margin-bottom: 4px;
            text-transform: uppercase;
        }

        .telemetry-value {
            font-size: 1.4rem;
            font-weight: 700;
            color: #FFFFFF;
            text-shadow: 0px 0px 6px rgba(255, 255, 255, 0.2);
        }

        /* 3D Canvas element */
        .canvas-container {
            position: relative;
            background-color: #000000;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 6px;
            height: 400px;
            width: 100%;
            cursor: grab;
            overflow: hidden;
        }

        .canvas-container:active {
            cursor: grabbing;
        }

        canvas {
            display: block;
            width: 100%;
            height: 100%;
        }

        /* Controller and Forms styling */
        .form-group {
            margin-bottom: 12px;
        }

        .form-label {
            font-size: 0.85rem;
            color: #9CA3AF;
            margin-bottom: 4px;
            display: block;
        }

        select, input[type="text"], input[type="range"] {
            width: 100%;
            background-color: #0F1113;
            border: 1px solid rgba(255, 255, 255, 0.12);
            color: #E5E7EB;
            padding: 8px 12px;
            border-radius: 4px;
            font-family: inherit;
        }

        .btn {
            background-color: #121314;
            border: 1px solid rgba(255, 255, 255, 0.15);
            color: #FFFFFF;
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
            transition: all 0.2s ease;
            text-align: center;
        }

        .btn:hover {
            background-color: #FFFFFF;
            color: #000000;
            border-color: #FFFFFF;
            box-shadow: 0px 0px 15px rgba(255, 255, 255, 0.2);
        }

        .btn:disabled {
            opacity: 0.3;
            cursor: not-allowed;
            background-color: #121314 !important;
            color: #FFFFFF !important;
        }

        .btn-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
        }

        /* Code Inspector styles */
        .code-area {
            background-color: #050505;
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #39FF14;
            font-family: monospace;
            padding: 12px;
            border-radius: 6px;
            overflow-x: auto;
            max-height: 250px;
            font-size: 0.85rem;
            line-height: 1.4;
        }

        /* Log panel styling */
        .log-area {
            background-color: #050505;
            border: 1px solid rgba(255, 255, 255, 0.08);
            font-family: monospace;
            padding: 12px;
            border-radius: 6px;
            height: 140px;
            overflow-y: auto;
            font-size: 0.8rem;
            color: #A3A3A3;
            line-height: 1.4;
        }
    </style>
</head>
<body>

    <h1>🧠 BONSAI BRAIN SIMULATOR 3D</h1>
    <div class="subtitle">64 Nodes. 3D Synaptic Kernels. In your browser.</div>
    <div class="description">
        Bonsai Brain Simulator 3D by Prism Neuro is a 3D biological brain simulation. Everything runs entirely locally in your browser using Streamlit & Plotly — no data leaves your device. Agentic WebGPU kernel optimization simulated on 4x4x4 grid nodes.
    </div>

    <!-- Diagnostics bar -->
    <div class="diagnostics-bar">
        <div class="diag-header">
            <span class="diag-title">⚡ 3D NEURAL ENGINE DIAGNOSTICS</span>
            <span class="diag-status">● 100% ACTIVE</span>
        </div>
        <div class="diag-desc">
            VRAM ALLOCATED: 4.00 / 4.00 MB | COMPUTE SHADERS: WGSL 3.0 Compiled | DEVICE: Local Browser WebGPU Emulated Pipeline
        </div>
        <div class="progress-track">
            <div class="progress-fill"></div>
        </div>
    </div>

    <!-- Main columns -->
    <div class="main-layout">
        <!-- Left Side: Interactive 3D Visualizer & Telemetries -->
        <div>
            <!-- Telemetries -->
            <div class="telemetry-grid">
                <div class="telemetry-item">
                    <div class="telemetry-label">CSI (Cognitive Sync)</div>
                    <div class="telemetry-value" id="csi-val">0.0%</div>
                </div>
                <div class="telemetry-item">
                    <div class="telemetry-label">PDI (Plasticity Density)</div>
                    <div class="telemetry-value" id="pdi-val">0.0%</div>
                </div>
                <div class="telemetry-item">
                    <div class="telemetry-label">VPI (Vascular Perfusion)</div>
                    <div class="telemetry-value" id="vpi-val">80.0%</div>
                </div>
            </div>

            <!-- Interactive 3D Canvas -->
            <div class="card">
                <div class="card-title">⚡ WebGPU-Shader Emulation Pipeline (3D View)</div>
                <div class="canvas-container" id="canvas-container">
                    <canvas id="brain-canvas"></canvas>
                </div>
                <div style="font-size: 0.8rem; color: #9CA3AF; margin-top: 8px; text-align: center;">
                    Dùng chuột kéo để xoay mô hình 3D. Nhấp chuột vào một node để lựa chọn nó.
                </div>
            </div>

            <!-- Charts -->
            <div class="card">
                <div class="card-title">📈 EEG Telemetry & Neuromodulator Chart</div>
                <div style="height: 180px; width: 100%;">
                    <canvas id="chart-canvas" style="width: 100%; height: 100%;"></canvas>
                </div>
            </div>
        </div>

        <!-- Right Side: Node Editor & Upgrades & Shaders -->
        <div>
            <!-- Node Selector & Config -->
            <div class="card">
                <div class="card-title" id="editor-title">🛠️ Node Editor: Node [1, 1, 1]</div>
                <div class="form-group">
                    <div class="btn-grid">
                        <button class="btn" onclick="setNodeType('Sensory')">⚡ Sensory</button>
                        <button class="btn" onclick="setNodeType('Interneuron')">🧠 Interneuron</button>
                        <button class="btn" onclick="setNodeType('Motor')">💪 Motor</button>
                        <button class="btn" onclick="setNodeType('Empty')">❌ Gỡ bỏ</button>
                    </div>
                </div>
                <div class="form-group" style="margin-top: 15px;">
                    <button class="btn" onclick="injectCharge()">🔌 Kích xung điện cực (+1.0 Charge)</button>
                </div>
                <div class="form-group">
                    <label class="form-label">Hướng truyền tải Axon:</label>
                    <select id="axon-dir" onchange="changeDirection()">
                        <option value="All">🌐 Sáu hướng (All)</option>
                        <option value="Up">Up (Phía Y+)</option>
                        <option value="Right">Right (Phía X+)</option>
                        <option value="Down">Down (Phía Y-)</option>
                        <option value="Left">Left (Phía X-)</option>
                        <option value="Front">Front (Phía Z+)</option>
                        <option value="Back">Back (Phía Z-)</option>
                    </select>
                </div>
                <div class="form-group">
                    <label class="form-label">Trọng số Synaptic: <span id="weight-lbl">1.0</span></label>
                    <input type="range" id="axon-weight" min="1" max="3" step="1" value="1" oninput="changeWeight()">
                </div>
            </div>

            <!-- Active Clinical Therapies -->
            <div class="card">
                <div class="card-title">🧪 Active Clinical Therapies</div>
                <div class="btn-grid" style="grid-template-columns: repeat(3, 1fr); gap: 6px;">
                    <button class="btn" style="font-size: 0.75rem;" onclick="triggerTherapy('Doping')">⚡ Doping</button>
                    <button class="btn" style="font-size: 0.75rem;" onclick="triggerTherapy('SSRI')">💊 SSRI</button>
                    <button class="btn" style="font-size: 0.75rem;" onclick="triggerTherapy('Focus')">🧠 Focus</button>
                    <button class="btn" style="font-size: 0.75rem;" onclick="triggerTherapy('rTMS')">🏥 rTMS</button>
                    <button class="btn" style="font-size: 0.75rem;" onclick="triggerTherapy('Opto')">🔦 Opto</button>
                    <button class="btn" style="font-size: 0.75rem;" onclick="triggerTherapy('VNS')">❤️ VNS</button>
                </div>
            </div>

            <!-- Shader Inspector -->
            <div class="card">
                <div class="card-title">🎛️ WGSL Shaders Code Inspector</div>
                <div class="form-group">
                    <select id="shader-select" onchange="showShader()">
                        <option value="propagation">synaptic_charge_propagation.wgsl</option>
                        <option value="plasticity">hebbian_plasticity_attention.wgsl</option>
                        <option value="gaba">gaba_normalization.wgsl</option>
                        <option value="vns">vagus_nerve_clamp.wgsl</option>
                    </select>
                </div>
                <pre class="code-area" id="shader-code"></pre>
            </div>

            <!-- Logs -->
            <div class="card">
                <div class="card-title">📋 Brain Activity Log</div>
                <div class="log-area" id="log-box"></div>
            </div>
        </div>
    </div>

    <!-- JS Logic -->
    <script>
        // Core 3D Grid State
        const GRID_SIZE = 4;
        let grid = [];
        let selectedCell = { x: 0, y: 0, z: 0 };

        let csi = 0.0;
        let pdi = 0.0;
        let vpi = 80.0;

        let iq = 0.0;
        let memory = 100.0;
        let energy = 100.0;
        let stress = 10.0;
        let sanity = 100.0;

        let logs = ["Khởi tạo mạng nơ-ron 3D WebGPU thành công."];
        let history = { ticks: [], csi: [], pdi: [], vpi: [] };
        let currentTick = 0;

        function initGrid() {
            grid = [];
            for (let x = 0; x < GRID_SIZE; x++) {
                let plane = [];
                for (let y = 0; y < GRID_SIZE; y++) {
                    let row = [];
                    for (let z = 0; z < GRID_SIZE; z++) {
                        row.push({
                            type: "Empty",
                            charge: 0.0,
                            threshold: 0.5,
                            direction: "All",
                            weight: 1.0
                        });
                    }
                    plane.push(row);
                }
                grid.push(plane);
            }

            // Starter Nodes
            grid[0][0][0] = { type: "Sensory", charge: 0.0, threshold: 0.4, direction: "All", weight: 1.0 };
            grid[1][1][1] = { type: "Interneuron", charge: 0.0, threshold: 0.5, direction: "All", weight: 1.0 };
            grid[3][3][3] = { type: "Motor", charge: 0.0, threshold: 0.6, direction: "All", weight: 1.0 };
        }

        // WebGPU shaders definitions
        const shaders = {
            propagation: `// synaptic_charge_propagation.wgsl
// Low-level WebGPU compute shader doing 3D synaptic charge propagation

@group(0) @binding(0) var<storage, read> input_charge: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_charge: array<f32>;
@group(0) @binding(2) var<uniform> signal_efficiency: f32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= 64u) { return; }

    let current_charge = input_charge[index];
    if (current_charge >= 0.5) {
        output_charge[index] = 0.0; // Reset parent core
        let neighbors_count = 6u;
        let charge_transfer = (current_charge * signal_efficiency) / f32(neighbors_count);

        for (var i = 0u; i < neighbors_count; i = i + 1u) {
            let neighbor_idx = get_neighbor_index(index, i);
            if (neighbor_idx < 64u) {
                atomicAdd(&output_charge[neighbor_idx], charge_transfer);
            }
        }
    }
}`,
            plasticity: `// hebbian_plasticity_attention.wgsl
// Low-level WebGPU compute shader adjusting threshold adaptation

@group(0) @binding(0) var<storage, read> active_charges: array<f32>;
@group(0) @binding(1) var<storage, read_write> thresholds: array<f32>;
@group(0) @binding(2) var<uniform> learning_rate: f32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= 64u) { return; }

    let charge = active_charges[index];
    if (charge > 0.3) {
        thresholds[index] = max(0.2, thresholds[index] - learning_rate);
    }
}`,
            gaba: `// gaba_normalization.wgsl
// Normalization pass shader performing stress attenuation using GABA

@group(0) @binding(0) var<storage, read> input_stress: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_stress: array<f32>;
@group(0) @binding(2) var<uniform> gaba_level: f32;

@compute @workgroup_size(1)
fn main() {
    let baseline_stress = input_stress[0];
    if (gaba_level >= 70.0) {
        output_stress[0] = baseline_stress * 0.60;
    } else {
        output_stress[0] = baseline_stress;
    }
}`,
            vns: `// vagus_nerve_clamp.wgsl
// GPU device clamp code executing Clinical Vagus Nerve Stimulation (VNS)

@group(0) @binding(0) var<storage, read_write> stress: array<f32>;
@group(0) @binding(1) var<storage, read_write> sanity: array<f32>;
@group(0) @binding(2) var<storage, read_write> gaba: array<f32>;

@compute @workgroup_size(1)
fn main() {
    stress[0] = 0.0;
    sanity[0] = min(100.0, sanity[0] + 20.0);
    gaba[0] = 90.0;
}`
        };

        function showShader() {
            let select = document.getElementById("shader-select");
            let key = select.value;
            document.getElementById("shader-code").textContent = shaders[key];
        }

        // Add activity log
        function addLog(msg) {
            logs.push(msg);
            if (logs.length > 30) logs.shift();

            let logBox = document.getElementById("log-box");
            logBox.innerHTML = logs.slice().reverse().map(l => `<div>● ${l}</div>`).join("");
        }

        // Node Editor Controls
        function updateEditor() {
            let cell = grid[selectedCell.x][selectedCell.y][selectedCell.z];
            document.getElementById("editor-title").textContent = `🛠️ Node Editor: Node [${selectedCell.x+1}, ${selectedCell.y+1}, ${selectedCell.z+1}]`;
            document.getElementById("axon-dir").value = cell.direction;
            document.getElementById("axon-weight").value = cell.weight;
            document.getElementById("weight-lbl").textContent = cell.weight.toFixed(1);
        }

        function setNodeType(type) {
            let cell = grid[selectedCell.x][selectedCell.y][selectedCell.z];
            cell.type = type;
            cell.charge = 0.0;
            cell.threshold = type === "Sensory" ? 0.4 : (type === "Motor" ? 0.6 : 0.5);
            addLog(`Đã đặt node [${selectedCell.x+1}, ${selectedCell.y+1}, ${selectedCell.z+1}] thành ${type}.`);
            updateEditor();
        }

        function injectCharge() {
            let cell = grid[selectedCell.x][selectedCell.y][selectedCell.z];
            if (cell.type !== "Empty") {
                cell.charge = 1.0;
                addLog(`🔌 [AP Clamping] Kích xung điện lượng cực đại tại node [${selectedCell.x+1}, ${selectedCell.y+1}, ${selectedCell.z+1}]!`);
            }
        }

        function changeDirection() {
            let cell = grid[selectedCell.x][selectedCell.y][selectedCell.z];
            cell.direction = document.getElementById("axon-dir").value;
            addLog(`Định hướng sợi trục node [${selectedCell.x+1}, ${selectedCell.y+1}, ${selectedCell.z+1}] thành ${cell.direction}.`);
        }

        function changeWeight() {
            let cell = grid[selectedCell.x][selectedCell.y][selectedCell.z];
            cell.weight = parseFloat(document.getElementById("axon-weight").value);
            document.getElementById("weight-lbl").textContent = cell.weight.toFixed(1);
        }

        function triggerTherapy(name) {
            if (name === "VNS") {
                stress = 0.0;
                sanity = Math.min(100.0, sanity + 20.0);
                addLog(`❤️ LÂM SÀNG: Kích thích dây thần kinh phế vị VNS! Hạ stress về không, sạc đầy GABA.`);
            } else if (name === "Doping") {
                addLog("⚡ LÂM SÀNG: Tiêm Dopamine cưỡng chế! Hệ thống hưng phấn cực đại.");
            } else if (name === "SSRI") {
                addLog("💊 LÂM SÀNG: Sử dụng hoạt chất SSRI chống phân rã Serotonin!");
            } else if (name === "Focus") {
                addLog("🧠 LÂM SÀNG: Kích hoạt Deep Focus tăng tốc độ phản xạ nhận thức.");
            } else if (name === "rTMS") {
                addLog("🏥 LÂM SÀNG: Thực hiện liệu pháp rTMS vỏ não giải trừ xơ hóa.");
            } else if (name === "Opto") {
                addLog("🔦 LÂM SÀNG: Phóng tia laser quang di truyền kích hoạt thế năng.");
            }
        }

        // 3D Rendering (HTML5 Canvas Projection)
        const canvas = document.getElementById("brain-canvas");
        const ctx = canvas.getContext("2d");

        let angleX = 0.5;
        let angleY = 0.6;
        let scale = 50;

        let isDragging = false;
        let prevMousePos = { x: 0, y: 0 };

        function resizeCanvas() {
            let container = document.getElementById("canvas-container");
            canvas.width = container.clientWidth;
            canvas.height = container.clientHeight;
        }

        window.addEventListener("resize", resizeCanvas);
        resizeCanvas();

        // Mouse Drag Orbit Control
        canvas.addEventListener("mousedown", (e) => {
            isDragging = true;
            prevMousePos = { x: e.clientX, y: e.clientY };
        });

        window.addEventListener("mouseup", () => {
            isDragging = false;
        });

        canvas.addEventListener("mousemove", (e) => {
            if (!isDragging) return;
            let dx = e.clientX - prevMousePos.x;
            let dy = e.clientY - prevMousePos.y;

            angleY += dx * 0.01;
            angleX += dy * 0.01;

            prevMousePos = { x: e.clientX, y: e.clientY };
        });

        // Click detection in projected 3D space to select nodes
        let nodesProjected = [];

        canvas.addEventListener("click", (e) => {
            let rect = canvas.getBoundingClientRect();
            let mouseX = e.clientX - rect.left;
            let mouseY = e.clientY - rect.top;

            let closestNode = null;
            let minDist = 15;

            nodesProjected.forEach(n => {
                let dx = mouseX - n.px;
                let dy = mouseY - n.py;
                let dist = Math.sqrt(dx*dx + dy*dy);
                if (dist < minDist) {
                    minDist = dist;
                    closestNode = n;
                }
            });

            if (closestNode) {
                selectedCell = { x: closestNode.x, y: closestNode.y, z: closestNode.z };
                updateEditor();
                addLog(`Đã chọn node [${selectedCell.x+1}, ${selectedCell.y+1}, ${selectedCell.z+1}].`);
            }
        });

        // Projections
        function project3D(x, y, z) {
            // Center around (0,0,0)
            let cx = x - (GRID_SIZE - 1) / 2;
            let cy = y - (GRID_SIZE - 1) / 2;
            let cz = z - (GRID_SIZE - 1) / 2;

            // Rotation around Y axis
            let x1 = cx * Math.cos(angleY) - cz * Math.sin(angleY);
            let z1 = cx * Math.sin(angleY) + cz * Math.cos(angleY);

            // Rotation around X axis
            let y2 = cy * Math.cos(angleX) - z1 * Math.sin(angleX);
            let z2 = cy * Math.sin(angleX) + z1 * Math.cos(angleX);

            // Perspective Projection
            let perspective = 200 / (200 + z2);
            let px = canvas.width / 2 + x1 * scale * perspective;
            let py = canvas.height / 2 + y2 * scale * perspective;

            return { px, py, zDepth: z2 };
        }

        // Main animation & draw loop
        function drawLoop() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Calculate projections
            nodesProjected = [];
            for (let x = 0; x < GRID_SIZE; x++) {
                for (let y = 0; y < GRID_SIZE; y++) {
                    for (let z = 0; z < GRID_SIZE; z++) {
                        let proj = project3D(x, y, z);
                        nodesProjected.push({
                            x, y, z,
                            px: proj.px,
                            py: proj.py,
                            zDepth: proj.zDepth,
                            cell: grid[x][y][z]
                        });
                    }
                }
            }

            // Sort by depth (Back-to-Front painter's algorithm)
            nodesProjected.sort((a, b) => b.zDepth - a.zDepth);

            // Draw axons (synapse connections)
            ctx.lineWidth = 1;
            nodesProjected.forEach(n => {
                if (n.cell.type === "Empty") return;

                // Adjacent offsets
                let targets = [];
                if (n.cell.direction === "All") {
                    targets = [
                        {dx: 1, dy: 0, dz: 0}, {dx: -1, dy: 0, dz: 0},
                        {dx: 0, dy: 1, dz: 0}, {dx: 0, dy: -1, dz: 0},
                        {dx: 0, dy: 0, dz: 1}, {dx: 0, dy: 0, dz: -1}
                    ];
                } else if (n.cell.direction === "Up") targets = [{dx: 0, dy: 1, dz: 0}];
                else if (n.cell.direction === "Right") targets = [{dx: 1, dy: 0, dz: 0}];
                else if (n.cell.direction === "Down") targets = [{dx: 0, dy: -1, dz: 0}];
                else if (n.cell.direction === "Left") targets = [{dx: -1, dy: 0, dz: 0}];
                else if (n.cell.direction === "Front") targets = [{dx: 0, dy: 0, dz: 1}];
                else if (n.cell.direction === "Back") targets = [{dx: 0, dy: 0, dz: -1}];

                targets.forEach(t => {
                    let nx = n.x + t.dx;
                    let ny = n.y + t.dy;
                    let nz = n.z + t.dz;

                    if (nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE && nz >= 0 && nz < GRID_SIZE) {
                        if (grid[nx][ny][nz].type !== "Empty") {
                            let nproj = project3D(nx, ny, nz);
                            ctx.beginPath();
                            ctx.moveTo(n.px, n.py);
                            ctx.lineTo(nproj.px, nproj.py);

                            // Pulse color if firing
                            if (n.cell.charge >= n.cell.threshold) {
                                ctx.strokeStyle = "#FFFFFF";
                                ctx.lineWidth = 2;
                            } else {
                                ctx.strokeStyle = "rgba(0, 240, 255, 0.25)";
                                ctx.lineWidth = 1;
                            }
                            ctx.stroke();
                        }
                    }
                });
            });

            // Draw nodes
            nodesProjected.forEach(n => {
                let cell = n.cell;
                let radius = 6;
                let color = "#3A3D40"; // Empty

                if (cell.type === "Sensory") color = "#00F0FF";
                else if (cell.type === "Interneuron") color = "#39FF14";
                else if (cell.type === "Motor") color = "#FF007F";

                let isSelected = (n.x === selectedCell.x && n.y === selectedCell.y && n.z === selectedCell.z);
                let isFiring = cell.type !== "Empty" && cell.charge >= cell.threshold;

                // Emulate WebGPU multi-pass bloom glow
                if (cell.type !== "Empty") {
                    // Outer Bloom Layer with color shifting based on evolutionary stage of the brain
                    let halo_color = "rgba(0, 240, 255, 0.12)";
                    let inner_color = "rgba(0, 240, 255, 0.35)";

                    if (iq >= 10000.0) {
                        halo_color = "rgba(255, 215, 0, 0.18)"; // Glowing Golden Halo for advanced evolution!
                        inner_color = "rgba(255, 215, 0, 0.45)";
                    } else if (cell.type === "Sensory") {
                        halo_color = "rgba(0, 240, 255, 0.12)";
                        inner_color = "rgba(0, 240, 255, 0.35)";
                    } else if (cell.type === "Interneuron") {
                        halo_color = "rgba(57, 255, 20, 0.12)";
                        inner_color = "rgba(57, 255, 20, 0.35)";
                    } else if (cell.type === "Motor") {
                        halo_color = "rgba(255, 0, 127, 0.12)";
                        inner_color = "rgba(255, 0, 127, 0.35)";
                    }

                    ctx.beginPath();
                    ctx.arc(n.px, n.py, radius * 3, 0, 2 * Math.PI);
                    ctx.fillStyle = isFiring ? "rgba(255, 170, 0, 0.15)" : halo_color;
                    ctx.fill();

                    ctx.beginPath();
                    ctx.arc(n.px, n.py, radius * 1.8, 0, 2 * Math.PI);
                    ctx.fillStyle = isFiring ? "rgba(255, 204, 0, 0.35)" : inner_color;
                    ctx.fill();
                }

                // Draw specialized biological cell structures
                if (cell.type === "Sensory") {
                    ctx.strokeStyle = "rgba(0, 240, 255, 0.7)";
                    ctx.lineWidth = 1;
                    for (let i = 0; i < 6; i++) {
                        let angle = (i * Math.PI) / 3;
                        ctx.beginPath();
                        ctx.moveTo(n.px, n.py);
                        ctx.lineTo(n.px + Math.cos(angle) * 12, n.py + Math.sin(angle) * 12);
                        ctx.stroke();
                    }
                } else if (cell.type === "Interneuron") {
                    ctx.strokeStyle = "rgba(57, 255, 20, 0.6)";
                    ctx.lineWidth = 1;
                    for (let i = 0; i < 3; i++) {
                        let angle = (i * 2 * Math.PI) / 3;
                        let endX = n.px + Math.cos(angle) * 10;
                        let endY = n.py + Math.sin(angle) * 10;
                        ctx.beginPath();
                        ctx.moveTo(n.px, n.py);
                        ctx.lineTo(endX, endY);
                        ctx.stroke();

                        ctx.beginPath();
                        ctx.moveTo(endX, endY);
                        ctx.lineTo(endX + Math.cos(angle + 0.5) * 5, endY + Math.sin(angle + 0.5) * 5);
                        ctx.stroke();
                    }
                } else if (cell.type === "Motor") {
                    ctx.strokeStyle = "rgba(255, 0, 127, 0.7)";
                    ctx.lineWidth = 1.5;
                    let angle = Math.PI / 4;
                    let axEndX = n.px + Math.cos(angle) * 14;
                    let axEndY = n.py + Math.sin(angle) * 14;
                    ctx.beginPath();
                    ctx.moveTo(n.px, n.py);
                    ctx.lineTo(axEndX, axEndY);
                    ctx.stroke();

                    ctx.beginPath();
                    ctx.moveTo(axEndX - Math.sin(angle) * 4, axEndY + Math.cos(angle) * 4);
                    ctx.lineTo(axEndX + Math.sin(angle) * 4, axEndY - Math.cos(angle) * 4);
                    ctx.stroke();
                }

                // Draw action potential electric sparks when firing
                if (isFiring) {
                    ctx.strokeStyle = "#FFFFFF";
                    ctx.lineWidth = 1.5;
                    for (let i = 0; i < 4; i++) {
                        let angle = Math.random() * 2 * Math.PI;
                        ctx.beginPath();
                        ctx.moveTo(n.px, n.py);
                        ctx.lineTo(n.px + Math.cos(angle) * 16, n.py + Math.sin(angle) * 16);
                        ctx.stroke();
                    }
                }

                // Core Layer
                ctx.beginPath();
                ctx.arc(n.px, n.py, isSelected ? radius + 3 : radius, 0, 2 * Math.PI);
                ctx.fillStyle = isFiring ? "#FFFFFF" : color;
                ctx.fill();

                // Highlight boundary for selection
                if (isSelected) {
                    ctx.strokeStyle = "#FFFFFF";
                    ctx.lineWidth = 1.5;
                    ctx.stroke();
                }
            });

            requestAnimationFrame(drawLoop);
        }

        // Biological simulation cycle logic (calculates CSI, PDI, VPI)
        function simTick() {
            currentTick++;

            let totalNeurons = 0;
            let activeCharged = 0;
            let totalInterneurons = 0;
            let modifiedInterneurons = 0;

            // 1. Sensory Nodes auto charge
            for (let x = 0; x < GRID_SIZE; x++) {
                for (let y = 0; y < GRID_SIZE; y++) {
                    for (let z = 0; z < GRID_SIZE; z++) {
                        let cell = grid[x][y][z];
                        if (cell.type === "Sensory") {
                            cell.charge = Math.min(1.0, cell.charge + 0.15);
                        }
                    }
                }
            }

            // 2. Propagation pass
            let nextCharges = JSON.parse(JSON.stringify(grid)).map(p => p.map(r => r.map(c => c.charge)));

            for (let x = 0; x < GRID_SIZE; x++) {
                for (let y = 0; y < GRID_SIZE; y++) {
                    for (let z = 0; z < GRID_SIZE; z++) {
                        let cell = grid[x][y][z];
                        if (cell.type !== "Empty") {
                            totalNeurons++;
                            if (cell.charge >= cell.threshold) {
                                activeCharged++;

                                // Propagation
                                let targets = [];
                                if (cell.direction === "All") {
                                    targets = [
                                        {dx: 1, dy: 0, dz: 0}, {dx: -1, dy: 0, dz: 0},
                                        {dx: 0, dy: 1, dz: 0}, {dx: 0, dy: -1, dz: 0},
                                        {dx: 0, dy: 0, dz: 1}, {dx: 0, dy: 0, dz: -1}
                                    ];
                                } else if (cell.direction === "Up") targets = [{dx: 0, dy: 1, dz: 0}];
                                else if (cell.direction === "Right") targets = [{dx: 1, dy: 0, dz: 0}];
                                else if (cell.direction === "Down") targets = [{dx: 0, dy: -1, dz: 0}];
                                else if (cell.direction === "Left") targets = [{dx: -1, dy: 0, dz: 0}];
                                else if (cell.direction === "Front") targets = [{dx: 0, dy: 0, dz: 1}];
                                else if (cell.direction === "Back") targets = [{dx: 0, dy: 0, dz: -1}];

                                let neighbors = [];
                                targets.forEach(t => {
                                    let nx = x + t.dx;
                                    let ny = y + t.dy;
                                    let nz = z + t.dz;
                                    if (nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE && nz >= 0 && nz < GRID_SIZE) {
                                        if (grid[nx][ny][nz].type !== "Empty") {
                                            neighbors.push({x: nx, y: ny, z: nz});
                                        }
                                    }
                                });

                                if (neighbors.length > 0) {
                                    let transfer = (cell.charge * 0.45 * cell.weight) / neighbors.length;
                                    neighbors.forEach(n => {
                                        nextCharges[n.x][n.y][n.z] = Math.min(1.0, nextCharges[n.x][n.y][n.z] + transfer);
                                    });
                                }

                                nextCharges[x][y][z] = 0.0; // reset
                            } else if (cell.charge >= cell.threshold * 0.5) {
                                activeCharged++;
                            }
                        }

                        if (cell.type === "Interneuron") {
                            totalInterneurons++;
                            if (cell.threshold !== 0.5) {
                                modifiedInterneurons++;
                            }
                        }
                    }
                }
            }

            // Apply next charges
            for (let x = 0; x < GRID_SIZE; x++) {
                for (let y = 0; y < GRID_SIZE; y++) {
                    for (let z = 0; z < GRID_SIZE; z++) {
                        grid[x][y][z].charge = nextCharges[x][y][z];
                    }
                }
            }

            // Calculations CSI, PDI, VPI
            csi = (activeCharged / Math.max(1, totalNeurons)) * 100.0;
            pdi = (modifiedInterneurons / Math.max(1, totalInterneurons)) * 100.0;
            vpi = 80.0 + Math.sin(currentTick * 0.2) * 5.0; // Simulated flow

            document.getElementById("csi-val").textContent = `${csi.toFixed(1)}%`;
            document.getElementById("pdi-val").textContent = `${pdi.toFixed(1)}%`;
            document.getElementById("vpi-val").textContent = `${vpi.toFixed(1)}%`;

            // Record history and redraw chart
            history.ticks.push(currentTick);
            history.csi.push(csi);
            history.pdi.push(pdi);
            history.vpi.push(vpi);

            if (history.ticks.length > 40) {
                history.ticks.shift();
                history.csi.shift();
                history.pdi.shift();
                history.vpi.shift();
            }

            drawChart();
        }

        // Clean Canvas Line Chart for EEG
        const chartCanvas = document.getElementById("chart-canvas");
        const chartCtx = chartCanvas.getContext("2d");

        function drawChart() {
            chartCanvas.width = chartCanvas.parentElement.clientWidth;
            chartCanvas.height = chartCanvas.parentElement.clientHeight;

            chartCtx.clearRect(0, 0, chartCanvas.width, chartCanvas.height);
            let len = history.ticks.length;
            if (len < 2) return;

            // Draw grid lines
            chartCtx.strokeStyle = "rgba(255, 255, 255, 0.05)";
            chartCtx.lineWidth = 1;
            for (let i = 0; i < 5; i++) {
                let y = (chartCanvas.height / 4) * i;
                chartCtx.beginPath();
                chartCtx.moveTo(0, y);
                chartCtx.lineTo(chartCanvas.width, y);
                chartCtx.stroke();
            }

            // Draw lines for CSI, PDI, VPI
            drawChartLine(history.csi, "#00F0FF"); // Cyan
            drawChartLine(history.pdi, "#39FF14"); // Green
            drawChartLine(history.vpi, "#FF007F"); // Magenta
        }

        function drawChartLine(data, color) {
            chartCtx.strokeStyle = color;
            chartCtx.lineWidth = 2.0;
            chartCtx.beginPath();

            let len = data.length;
            for (let i = 0; i < len; i++) {
                let x = (chartCanvas.width / (len - 1)) * i;
                let y = chartCanvas.height - (data[i] / 100.0) * chartCanvas.height;

                if (i === 0) chartCtx.moveTo(x, y);
                else chartCtx.lineTo(x, y);
            }
            chartCtx.stroke();
        }

        // Initialize and Start loops
        initGrid();
        updateEditor();
        showShader();
        addLog("Hệ thống WebGPU Shading Pipeline sẵn sàng.");

        // Loops
        requestAnimationFrame(drawLoop);
        setInterval(simTick, 1000);
    </script>
</body>
</html>
"""
with tab1:
    st.components.v1.html(webgpu_html_content, height=1100, scrolling=True)

with tab2:
    st.header("🤖 Trợ Lý AI VBot1")
    st.write("Trò chuyện trực tiếp với Llama 3 hoặc tải lên tệp tài liệu PDF để tóm tắt nội dung bằng Gemini 1.5 Flash.")

    chat_input = st.text_input("Nhập câu hỏi của bạn cho Llama 3:", key="web_chat_input")
    if st.button("Gửi câu hỏi", key="web_chat_submit") and chat_input:
        if not hf_client:
            st.warning("Llama 3 hiện không khả dụng (thiếu HF_TOKEN).")
        else:
            with st.spinner("Llama 3 đang suy nghĩ..."):
                try:
                    messages = [{"role": "user", "content": chat_input}]
                    completion = hf_client.chat_completion(
                        model="meta-llama/Meta-Llama-3-8B-Instruct",
                        messages=messages,
                        max_tokens=500
                    )
                    st.write("**VBot1 (Llama 3):**")
                    st.write(completion.choices[0].message.content)
                except Exception as e:
                    st.error(f"Lỗi phản hồi: {e}")

    st.write("---")
    st.subheader("📄 Tóm tắt tài liệu PDF (Gemini 1.5 Flash)")
    uploaded_file = st.file_uploader("Tải lên tệp PDF của bạn:", type=["pdf"])
    if uploaded_file is not None:
        if st.button("Trích xuất và tóm tắt", key="web_pdf_submit"):
            with st.spinner("Đang xử lý tài liệu..."):
                file_bytes = uploaded_file.read()
                text = extract_pdf_text(file_bytes)
                if not text:
                    st.warning("Không tìm thấy văn bản trong tệp PDF.")
                else:
                    st.write("**Nội dung tóm tắt (Gemini 1.5 Flash):**")
                    summary = summarize_with_gemini(text)
                    st.write(summary)

# Background Ticking Thread
if "bot_thread" not in st.session_state:
    st.session_state.bot_thread = True
    if TELEGRAM_TOKEN:
        t = threading.Thread(target=run_bot, daemon=True)
        t.start()
        add_log("🤖 SYSTEM: Đã khởi động luồng nền cho Trợ lý Telegram Bot.")

# Realtime browser refresh if playing
if st.session_state.playing:
    time.sleep(1.0 / st.session_state.tick_speed)
    run_simulation_tick()
    st.rerun()
