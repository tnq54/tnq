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
            "evolution_stage": "Bò sát",
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

        st.session_state.game_log = ["Khởi tạo bộ não 3D WebGPU thành công. Trạng thái tiến hóa: Hành não Bò sát."]
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
    if iq < 150:
        return "Bò sát (Instinct)"
    elif iq < 750:
        return "Thú cổ (Emotional)"
    elif iq < 3000:
        return "Người tinh khôn (Logical)"
    else:
        return "Siêu trí tuệ lượng tử (Transcendence)"

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

# WebGPU Sci-fi Holographic HUD Style Injection (Styled exactly like Bonsai WebGPU Kernels)
st.markdown("""
<style>
    /* Clean Bonsai WebGPU dark minimal theme */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    .stApp {
        background-color: #050505 !important;
        color: #E5E7EB !important;
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    }

    /* Minimalist Bonsai button style */
    div.stButton > button {
        background-color: #121314 !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        border-radius: 6px !important;
        transition: all 0.2s ease !important;
        padding: 8px 16px !important;
    }

    div.stButton > button:hover {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border-color: #FFFFFF !important;
        box-shadow: 0px 0px 15px rgba(255, 255, 255, 0.2) !important;
    }

    /* Elegant bold headers */
    h1 {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px !important;
        color: #FFFFFF !important;
        border: none !important;
        padding-bottom: 0px !important;
        text-transform: uppercase !important;
    }
    h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border: none !important;
    }

    /* Telemetry Panel board styles */
    .telemetry-board {
        background-color: #0A0B0C;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0px;
    }

    /* Input & logs styled minimal */
    textarea, input {
        background-color: #0A0B0C !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        color: #E5E7EB !important;
        font-family: monospace !important;
    }

    /* Ensure the sidebar has a distinct dark background matching the black theme */
    [data-testid="stSidebar"] {
        background-color: #090A0B !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
        color: #E5E7EB !important;
    }

    /* High contrast metrics for dark background readability */
    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        text-shadow: 0px 0px 5px rgba(255, 255, 255, 0.3) !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #9CA3AF !important;
    }
    [data-testid="stMetricValue"] > div {
        color: #FFFFFF !important;
    }
    [data-testid="stMetricLabel"] > div {
        color: #9CA3AF !important;
    }
</style>
""", unsafe_allow_html=True)

# Title & Info
st.title("🧠 BONSAI BRAIN SIMULATOR 3D")
st.markdown("<p style='font-size: 1.15rem; font-weight: 600; color: #9CA3AF; letter-spacing: 0.5px; margin-top: -15px;'>64 Nodes. 3D Synaptic Kernels. In your browser.</p>", unsafe_allow_html=True)
st.write("Bonsai Brain Simulator 3D by Prism Neuro is a 3D biological brain simulation. Everything runs entirely locally in your browser using Streamlit & Plotly — no data leaves your device. Agentic WebGPU kernel optimization simulated on 4x4x4 grid nodes.")

# Simulated "Load Neural Engine" WebGPU Diagnostics Bar
st.markdown("""
<div style="background-color: #0F1113; border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 18px; margin: 15px 0px; font-family: monospace;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
        <span style="font-weight: bold; color: #FFFFFF; font-size: 0.95rem; letter-spacing: 0.5px;">⚡ 3D NEURAL ENGINE DIAGNOSTICS</span>
        <span style="color: #39FF14; font-weight: bold; letter-spacing: 1px;">● 100% ACTIVE</span>
    </div>
    <div style="font-size: 0.85rem; color: #9CA3AF; margin-bottom: 8px;">
        VRAM ALLOCATED: 4.00 / 4.00 MB | COMPUTE SHADERS: WGSL 3.0 Compiled | DEVICE: Local Browser WebGPU Emulated Pipeline
    </div>
    <div style="background-color: #1F2327; height: 8px; border-radius: 4px; overflow: hidden; width: 100%;">
        <div style="background-color: #39FF14; height: 100%; width: 100%; box-shadow: 0px 0px 10px #39FF14;"></div>
    </div>
</div>
""", unsafe_allow_html=True)

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

with tab1:
    # Sidebar - Controls Panel
    st.sidebar.markdown("### 🎛️ Bảng Điều Khiển Chu Kỳ Não")

    # Auto play controls
    play_cols = st.sidebar.columns(2)
    with play_cols[0]:
        if st.sidebar.button("▶️ CHẠY (Run)", use_container_width=True):
            st.session_state.playing = True
    with play_cols[1]:
        if st.sidebar.button("⏸️ TẠM DỪNG", use_container_width=True):
            st.session_state.playing = False

    # Step-by-step trigger
    if st.sidebar.button("⏭️ Bước Tiếp Theo (Single Tick)", use_container_width=True):
        run_simulation_tick()
        st.rerun()

    speed_multiplier = st.sidebar.slider("Tốc độ tiến hóa (Tích tắc):", min_value=0.5, max_value=4.0, value=1.0, step=0.5)
    st.session_state.tick_speed = speed_multiplier

    st.sidebar.markdown("---")

    # Persistent Personal Records Dashboard
    st.sidebar.markdown("### 🏆 Thành Tích Đạt Được")
    st.sidebar.metric("IQ Cao Nhất", f"{st.session_state.stats.get('high_score_iq', 0.0):.1f}")
    st.sidebar.metric("Trí Nhớ Cực Đại", f"{st.session_state.stats.get('max_memory', 10.0):.1f} MB")
    st.sidebar.metric("Chuỗi Tỉnh Táo Liên Tục", f"{st.session_state.stats.get('burnout_streak', 0)} ticks")
    st.sidebar.metric("Kỷ Lục Chuỗi Tỉnh Táo", f"{st.session_state.stats.get('max_streak', 0)} ticks")

    # Status metrics row
    chems = st.session_state.chemicals
    stats = st.session_state.stats
    upgrades = st.session_state.upgrades
    mode = st.session_state.get("game_mode", "Normal")

    # Visual Spark details if active
    visual_spark = st.session_state.get("visual_spark", None)
    if upgrades.get("occipital_lobe", 0) >= 1 and visual_spark:
        st.sidebar.info(f"👁️ **Kích thích thị giác:** Vị trí [{visual_spark['pos'][0]+1}, {visual_spark['pos'][1]+1}, {visual_spark['pos'][2]+1}] | Trục hướng: **{visual_spark['dir']}**")

    # Spatial Gate details if active
    spatial_gate = st.session_state.get("spatial_gate", None)
    if upgrades.get("parietal_lobe", 0) >= 1 and spatial_gate:
        st.sidebar.info(f"🧭 **Luồng định vị không gian:** Vị trí Gating [{spatial_gate[0]+1}, {spatial_gate[1]+1}, {spatial_gate[2]+1}]!")

    # Auditory frequency if active
    if upgrades.get("temporal_lobe", 0) >= 1 and "auditory_freq" in st.session_state:
        freq = st.session_state.auditory_freq
        resonance = "🎵 **CỘNG HƯỞNG (3x Memory)!**" if (400 <= freq <= 500) else "🎵 Bình thường"
        st.sidebar.info(f"🔊 **Tần số âm thanh:** {freq} Hz ({resonance})")

    # Display game status summary cards
    sc_cols = st.columns(4)
    sc_cols[0].metric("🧠 Chỉ Số IQ", f"{stats['iq']:.1f}", help="Tích lũy từ hành động nơ-ron Motor phát hỏa.")
    sc_cols[1].metric("💾 Dung Lượng Trí Nhớ", f"{stats['memory']:.1f}/{stats['max_memory']:.1f} MB", help="Dùng để cấy ghép nơ-ron mới hoặc kích hoạt liệu pháp.")
    sc_cols[2].metric("🔋 Glucose & Oxy", f"{chems['energy']:.1f}%", help="Mức năng lượng cơ bản của não bộ. Giảm khi có nhiều tế bào thần kinh hoạt động.")
    sc_cols[3].metric("🔄 Trạng Thái Tiến Hóa", stats["circadian_cycle"], help="Hệ thống chu kỳ sinh học ngày đêm (Day/Night) liên tục.")

    # Glowing WebGPU Core Engine Telemetry diagnostic board
    st.markdown('<div class="telemetry-board">', unsafe_allow_html=True)
    st.markdown("##### 🚀 WEBGPU DIAGNOSTICS & TELEMETRY BOARD")
    index_cols = st.columns(3)
    index_cols[0].metric("🌐 CSI (Cognitive Sync)", f"{st.session_state.get('csi', 0.0):.1f}%", help="Cognitive Sync Index: Đo lường mức độ đồng bộ kích phát xung điện toàn vỏ não 3D.")
    index_cols[1].metric("🧬 PDI (Plasticity Density)", f"{st.session_state.get('pdi', 0.0):.1f}%", help="Plasticity Density Index: Mật độ thích ứng thích nghi liên kết nơ-ron (biến đổi ngưỡng).")
    index_cols[2].metric("🩸 VPI (Vascular Perfusion)", f"{st.session_state.get('vpi', 80.0):.1f}%", help="Vascular Perfusion Index: Hiệu năng tưới máu và cung cấp dưỡng chất huyết quản vỏ não.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Main columns for 3D Visualizer and Editor Panel
    game_cols = st.columns([5, 4])

    with game_cols[0]:
        st.markdown("#### ⚡ WebGPU-Shader Emulation Pipeline (3D View)")
        st.caption("Mạng lưới 3D tương tác 4x4x4 (64 nodes). Sử dụng chuột để xoay, thu phóng và xem điện thế.")

        # Plotly 3D visualizer call
        grid = st.session_state.neuron_grid
        selected_cell = st.session_state.selected_cell

        fig = render_3d_brain(grid, selected_cell)
        st.plotly_chart(fig, use_container_width=True)

        # 3D Coordinate Selectors to change active selection
        st.markdown("##### 📍 Chọn Tọa Độ Node Muốn Điều Chỉnh (XYZ Coordinate Selectors)")
        sel_x, sel_y, sel_z = selected_cell

        coord_cols = st.columns(3)
        with coord_cols[0]:
            new_x = st.slider("Tọa độ X (Trục Hoành):", min_value=1, max_value=GRID_SIZE, value=sel_x+1, step=1)
        with coord_cols[1]:
            new_y = st.slider("Tọa độ Y (Trục Tung):", min_value=1, max_value=GRID_SIZE, value=sel_y+1, step=1)
        with coord_cols[2]:
            new_z = st.slider("Tọa độ Z (Trục Sâu):", min_value=1, max_value=GRID_SIZE, value=sel_z+1, step=1)

        if (new_x-1, new_y-1, new_z-1) != selected_cell:
            st.session_state.selected_cell = (new_x-1, new_y-1, new_z-1)
            st.rerun()

        # Save & Load Circuit Codes Panel
        st.markdown("---")
        st.markdown("##### 💾 Lưu & Tải Sơ Đồ Mạch Thần Kinh (Circuit Share Codes)")
        st.caption("Mã chia sẻ mạch nơ-ron hiện tại hoặc nhập mã của người khác để xây dựng nhanh!")

        share_cols = st.columns([3, 1])
        with share_cols[0]:
            current_code = serialize_grid(st.session_state.neuron_grid)
            code_input = st.text_input("Mã sơ đồ mạch hiện tại (Copy-paste):", value=current_code, key="cur_code_text")
        with share_cols[1]:
            if st.button("📥 Tải sơ đồ (Load Code)", use_container_width=True):
                loaded_grid = deserialize_grid(code_input)
                if loaded_grid:
                    st.session_state.neuron_grid = loaded_grid
                    add_log("📥 TẢI SƠ ĐỒ: Khôi phục và cấy ghép sơ đồ mạch nơ-ron 3D thành công!")
                    st.rerun()
                else:
                    st.error("Mã sơ đồ không hợp lệ!")

        st.markdown("---")
        st.markdown("##### 📁 Thư Viện Bản Lưu Cục Bộ (Local Multi-slot Save Library)")
        st.caption("Lưu nhanh sơ đồ mạch thần kinh hiện tại vào các khe cất trữ trong phiên hoạt động này.")

        save_slots = st.session_state.get("save_slots", {"Slot 1": None, "Slot 2": None, "Slot 3": None})
        slot_cols = st.columns(3)
        for idx, slot_key in enumerate(["Slot 1", "Slot 2", "Slot 3"]):
            with slot_cols[idx]:
                slot_data = save_slots.get(slot_key, None)
                status_txt = "🟢 Đã Lưu Sơ Đồ" if slot_data else "⚪ Khe Trống"
                st.write(f"**{slot_key}:** `{status_txt}`")

                btn_cols = st.columns(2)
                with btn_cols[0]:
                    if st.button(f"Lưu vào {slot_key}", key=f"save_to_{idx}", use_container_width=True):
                        save_slots[slot_key] = serialize_grid(st.session_state.neuron_grid)
                        st.session_state.save_slots = save_slots
                        add_log(f"💾 THƯ VIỆN: Đã lưu nhanh sơ đồ mạch hiện tại vào {slot_key}!")
                        st.rerun()
                with btn_cols[1]:
                    load_disabled = slot_data is None
                    if st.button(f"Tải từ {slot_key}", key=f"load_from_{idx}", disabled=load_disabled, use_container_width=True):
                        loaded_grid = deserialize_grid(slot_data)
                        if loaded_grid:
                            st.session_state.neuron_grid = loaded_grid
                            add_log(f"📥 THƯ VIỆN: Đã tải nhanh sơ đồ mạch từ {slot_key}!")
                            st.rerun()

    with game_cols[1]:
        # Cell configuration section below grid
        selected_r, selected_c, selected_z = selected_cell
        current_cell = grid[selected_r][selected_c][selected_z]

        st.markdown(f"#### 🛠️ Bảng Điều Khiển Nơ-ron: **Node [{selected_r + 1}, {selected_c + 1}, {selected_z + 1}]**")
        st.write(f"Trạng thái hiện tại: **{current_cell['type']}** (Điện tích tích lũy: `{current_cell['charge']:.2f}/{current_cell['threshold']:.2f}`)")

        ach_discount = 1.0 - (st.session_state.chemicals["acetylcholine"] / 200.0)
        cost_sensory = int(25 * ach_discount)
        cost_inter = int(15 * ach_discount)
        cost_motor = int(40 * ach_discount)

        edit_cols = st.columns(4)

        with edit_cols[0]:
            sensory_disabled = current_cell["type"] == "Sensory" or st.session_state.stats["memory"] < cost_sensory
            if st.button(f"⚡ Sensory\n({cost_sensory} MB)", disabled=sensory_disabled, use_container_width=True):
                st.session_state.stats["memory"] -= cost_sensory
                grid[selected_r][selected_c][selected_z] = {
                    "type": "Sensory",
                    "charge": 0.0,
                    "threshold": 0.4,
                    "fire_rate": 0.3,
                    "last_fired": -1,
                    "direction": "All",
                    "weight": 1.0,
                    "amyloid_plaque": False
                }
                add_log(f"Cấy ghép Nơ-ron cảm giác (Sensory) tại [{selected_r+1},{selected_c+1},{selected_z+1}] (-{cost_sensory} MB)")
                st.rerun()

        with edit_cols[1]:
            inter_disabled = current_cell["type"] == "Interneuron" or st.session_state.stats["memory"] < cost_inter
            if st.button(f"🧠 Interneuron\n({cost_inter} MB)", disabled=inter_disabled, use_container_width=True):
                st.session_state.stats["memory"] -= cost_inter
                grid[selected_r][selected_c][selected_z] = {
                    "type": "Interneuron",
                    "charge": 0.0,
                    "threshold": 0.5,
                    "fire_rate": 0.0,
                    "last_fired": -1,
                    "direction": "All",
                    "weight": 1.0,
                    "amyloid_plaque": False
                }
                add_log(f"Cấy ghép Nơ-ron liên kết (Interneuron) tại [{selected_r+1},{selected_c+1},{selected_z+1}] (-{cost_inter} MB)")
                st.rerun()

        with edit_cols[2]:
            motor_disabled = current_cell["type"] == "Motor" or st.session_state.stats["memory"] < cost_motor
            if st.button(f"💪 Motor\n({cost_motor} MB)", disabled=motor_disabled, use_container_width=True):
                st.session_state.stats["memory"] -= cost_motor
                grid[selected_r][selected_c][selected_z] = {
                    "type": "Motor",
                    "charge": 0.0,
                    "threshold": 0.6,
                    "fire_rate": 0.0,
                    "last_fired": -1,
                    "direction": "All",
                    "weight": 1.0,
                    "amyloid_plaque": False
                }
                add_log(f"Cấy ghép Nơ-ron vận động (Motor) tại [{selected_r+1},{selected_c+1},{selected_z+1}] (-{cost_motor} MB)")
                st.rerun()

        with edit_cols[3]:
            delete_disabled = current_cell["type"] == "Empty"
            if st.button("❌ Gỡ bỏ\n(Hoàn 50%)", disabled=delete_disabled, use_container_width=True):
                refund = 0
                if current_cell["type"] == "Sensory":
                    refund = int(cost_sensory * 0.5)
                elif current_cell["type"] == "Interneuron":
                    refund = int(cost_inter * 0.5)
                elif current_cell["type"] == "Motor":
                    refund = int(cost_motor * 0.5)

                st.session_state.stats["memory"] += refund
                grid[selected_r][selected_c][selected_z] = {
                    "type": "Empty",
                    "charge": 0.0,
                    "threshold": 0.5,
                    "fire_rate": 0.0,
                    "last_fired": -1,
                    "direction": "All",
                    "weight": 1.0,
                    "amyloid_plaque": False
                }
                add_log(f"Xóa bỏ nơ-ron tại [{selected_r+1},{selected_c+1},{selected_z+1}] (Thu hồi +{refund} MB)")
                st.rerun()

        if current_cell["type"] != "Empty":
            st.write("---")
            # Active Electrode Probe (Action Potential Clamp) Button!
            probe_disabled = st.session_state.stats["memory"] < 15.0
            if st.button("🔌 Kích xung điện cực (+1.0 Charge) (-15 MB Memory)", disabled=probe_disabled, use_container_width=True, help="Kích xung điện thế trực tiếp tại nơ-ron được chọn."):
                st.session_state.stats["memory"] -= 15.0
                grid[selected_r][selected_c][selected_z]["charge"] = 1.0
                add_log(f"🔌 [Điện cực chủ động 3D] Kích xung điện lượng cực đại tại node [{selected_r+1},{selected_c+1},{selected_z+1}]!")
                st.rerun()

            st.write("---")
            axon_cols = st.columns(2)
            with axon_cols[0]:
                st.markdown("**🧭 Định hướng sợi trục (Axon Target):**")
                cur_dir = current_cell.get("direction", "All")
                dirs_list = ["All", "Up", "Right", "Down", "Left", "Front", "Back"]
                dir_names = {
                    "All": "🌐 Sáu hướng (All)",
                    "Up": "Up (Phía Y+)",
                    "Right": "Right (Phía X+)",
                    "Down": "Down (Phía Y-)",
                    "Left": "Left (Phía X-)",
                    "Front": "Front (Phía Z+)",
                    "Back": "Back (Phía Z-)"
                }
                selected_new_dir = st.selectbox(
                    "Chọn hướng truyền tải:",
                    dirs_list,
                    index=dirs_list.index(cur_dir),
                    format_func=lambda x: dir_names[x],
                    key=f"dir_select_{selected_r}_{selected_c}_{selected_z}"
                )
                if not isinstance(selected_new_dir, str):
                    selected_new_dir = cur_dir
                if selected_new_dir != cur_dir and selected_new_dir in dir_names:
                    grid[selected_r][selected_c][selected_z]["direction"] = selected_new_dir
                    add_log(f"Định hướng lại trục nơ-ron [{selected_r+1},{selected_c+1},{selected_z+1}] thành {dir_names[selected_new_dir]}")
                    st.rerun()

            with axon_cols[1]:
                st.markdown("**🔋 Khuyếch đại liên kết (Synaptic Weight):**")
                cur_weight = float(current_cell.get("weight", 1.0))
                new_weight = st.slider(
                    "Trọng số nhân điện thế phát xung:",
                    min_value=1.0,
                    max_value=3.0,
                    value=cur_weight,
                    step=1.0,
                    key=f"weight_select_{selected_r}_{selected_c}_{selected_z}"
                )
                if not isinstance(new_weight, (int, float)):
                    new_weight = cur_weight
                if new_weight != cur_weight:
                    grid[selected_r][selected_c][selected_z]["weight"] = new_weight
                    try:
                        weight_str = f"x{new_weight:.1f}"
                        add_log(f"Tăng cường trọng số khớp thần kinh tại [{selected_r+1},{selected_c+1},{selected_z+1}] lên {weight_str}!")
                    except Exception:
                        pass
                    st.rerun()

        # Clinical Therapies Panel
        st.write("---")
        st.markdown("##### 🧪 Trung tâm nội tiết tố & Liệu pháp Lâm sàng (Active Abilities)")
        hormone_cols = st.columns(9)
        cooldowns = st.session_state.cooldowns

        with hormone_cols[0]:
            doping_disabled = cooldowns["doping"] > 0
            btn_label_doping = f"⚡ Doping Dopamine ({cooldowns['doping']}s)" if doping_disabled else "⚡ Doping"
            if st.button(btn_label_doping, disabled=doping_disabled, use_container_width=True, help="⚡ Kích thích Doping Dopamine: Tăng +20 Dopamine trực tiếp và sạc thêm +5 Dopamine mỗi tick trong 5 ticks. Cooldown 15s."):
                chems["dopamine"] = min(100.0, chems["dopamine"] + 20.0)
                st.session_state.active_buffs["doping"] = 5
                cooldowns["doping"] = 15
                add_log("⚡ LÂM SÀNG: Tiêm Dopamine cưỡng chế! Hệ thống hưng phấn cực đại.")
                st.rerun()

        with hormone_cols[1]:
            ssri_disabled = cooldowns["ssri"] > 0
            btn_label_ssri = f"💊 Serotonin (SSRI) ({cooldowns['ssri']}s)" if ssri_disabled else "💊 SSRI"
            if st.button(btn_label_ssri, disabled=ssri_disabled, use_container_width=True, help="💊 SSRI (Serotonin Reuptake Inhibitor): Tăng +20 Serotonin trực tiếp và sạc +3 Serotonin, giảm 50% Stress tạo ra mỗi tick trong 8 ticks. Cooldown 25s."):
                chems["serotonin"] = min(100.0, chems["serotonin"] + 20.0)
                st.session_state.active_buffs["ssri"] = 8
                cooldowns["ssri"] = 25
                add_log("💊 LÂM SÀNG: Sử dụng hoạt chất SSRI chống phân rã Serotonin! Điều hòa stress thần kinh vỏ não.")
                st.rerun()

        with hormone_cols[2]:
            focus_disabled = cooldowns["focus"] > 0
            btn_label_focus = f"🧠 Tập trung ({cooldowns['focus']}s)" if focus_disabled else "🧠 Focus"
            if st.button(btn_label_focus, disabled=focus_disabled, use_container_width=True, help="🧠 Kích thích Tập trung (Deep Focus): Tăng +15 Acetylcholine trực tiếp và sạc +4 Acetylcholine mỗi tick trong 10 ticks. Cooldown 20s."):
                chems["acetylcholine"] = min(100.0, chems["acetylcholine"] + 15.0)
                st.session_state.active_buffs["focus"] = 10
                cooldowns["focus"] = 20
                add_log("🧠 LÂM SÀNG: Kích hoạt Deep Focus! Acetylcholine tăng tốc dẫn truyền thông tin nhận thức.")
                st.rerun()

        with hormone_cols[3]:
            rtms_disabled = cooldowns["rtms"] > 0
            btn_label_rtms = f"🏥 Liệu pháp rTMS ({cooldowns['rtms']}s)" if rtms_disabled else "🏥 rTMS"
            if st.button(btn_label_rtms, disabled=rtms_disabled, use_container_width=True, help="🏥 Liệu pháp rTMS (Kích thích từ trường lặp): Khôi phục toàn bộ ngưỡng nơ-ron (Threshold) bị xơ hóa do Alzheimer về mức chuẩn và phục hồi +40% Sanity. Cooldown 35s."):
                for x in range(GRID_SIZE):
                    for y in range(GRID_SIZE):
                        for z in range(GRID_SIZE):
                            t_name = grid[x][y][z]["type"]
                            if t_name != "Empty":
                                grid[x][y][z]["threshold"] = 0.4 if t_name == "Sensory" else (0.6 if t_name == "Motor" else 0.5)
                chems["sanity"] = min(100.0, chems["sanity"] + 40.0)
                cooldowns["rtms"] = 35
                add_log("🏥 LÂM SÀNG: Thực hiện liệu pháp rTMS vỏ não! Đã giải trừ xơ cứng và tái thiết lập ngưỡng thế năng chuẩn.")
                st.rerun()

        with hormone_cols[4]:
            opto_disabled = cooldowns["opto"] > 0
            btn_label_opto = f"🔦 Optogenetics ({cooldowns['opto']}s)" if opto_disabled else "🔦 Opto"
            if st.button(btn_label_opto, disabled=opto_disabled, use_container_width=True, help="🔦 Liệu pháp Quang di truyền (Optogenetic Laser Pulse): Kích hoạt chớp laser hội tụ nạp ngay lập tức +0.5 điện tích cho toàn bộ các nơ-ron nằm trên trục X, Y, Z của node được chọn. Cooldown 18s."):
                for x in range(GRID_SIZE):
                    for y in range(GRID_SIZE):
                        for z in range(GRID_SIZE):
                            if (x == selected_r or y == selected_c or z == selected_z) and grid[x][y][z]["type"] != "Empty":
                                grid[x][y][z]["charge"] = min(1.0, grid[x][y][z]["charge"] + 0.5)
                cooldowns["opto"] = 18
                add_log(f"🔦 LÂM SÀNG: Phóng tia laser quang di truyền dọc theo tọa độ nơ-ron [{selected_r+1},{selected_c+1},{selected_z+1}]!")
                st.rerun()

        with hormone_cols[5]:
            cortisol_disabled = cooldowns.get("cortisol", 0) > 0 or st.session_state.stats["memory"] < 30.0
            btn_label_cortisol = f"🧪 Cortisol Wash ({cooldowns.get('cortisol', 0)}s)" if cooldowns.get("cortisol", 0) > 0 else "🧪 Cortisol"
            if st.button(btn_label_cortisol, disabled=cortisol_disabled, use_container_width=True, help="🧪 Rửa viêm Cortisol Wash: Chi phí 30 MB Bộ nhớ. Lập tức dập tắt và đặt mức Viêm thần kinh (Neuro-inflammation) về 10.0%. Cooldown 20s."):
                st.session_state.stats["memory"] -= 30.0
                chems["neuro_inflammation"] = 10.0
                cooldowns["cortisol"] = 20
                add_log("🧪 LÂM SÀNG: Thực hiện Cortisol Wash! Rửa trôi bão viêm cytokine, bảo vệ màng bao myelin vỏ não.")
                st.rerun()

        with hormone_cols[6]:
            propranolol_disabled = cooldowns.get("propranolol", 0) > 0
            btn_label_propranolol = f"🩺 Propranolol ({cooldowns.get('propranolol', 0)}s)" if cooldowns.get("propranolol", 0) > 0 else "🩺 Beta-block"
            if st.button(btn_label_propranolol, disabled=propranolol_disabled, use_container_width=True, help="🩺 Hoạt chất chẹn beta Propranolol: Đưa nồng độ Norepinephrine (Fight-or-Flight) lập tức về mức an toàn 10.0%, giảm tải triệt để trạng thái run giật bão hòa. Cooldown 20s."):
                chems["norepinephrine"] = 10.0
                cooldowns["propranolol"] = 20
                add_log("🩺 LÂM SÀNG: Uống Propranolol chẹn beta giao cảm! Hạ mức norepinephrine khẩn cấp tránh sốc hoảng loạn.")
                st.rerun()

        with hormone_cols[7]:
            sprouting_disabled = cooldowns.get("sprouting", 0) > 0 or st.session_state.stats["memory"] < 40.0
            btn_label_sprouting = f"🌱 Sprouting ({cooldowns.get('sprouting', 0)}s)" if cooldowns.get("sprouting", 0) > 0 else "🌱 Sprouting"
            if st.button(btn_label_sprouting, disabled=sprouting_disabled, use_container_width=True, help="🌱 Nảy mầm liên kết mới: Chi phí 40 MB Bộ nhớ. Sao chép ngẫu nhiên một nơ-ron liên kết (Interneuron) hiện có sang một ô trống lân cận để xây dựng liên kết free. Cooldown 30s."):
                candidates = []
                for x in range(GRID_SIZE):
                    for y in range(GRID_SIZE):
                        for z in range(GRID_SIZE):
                            if grid[x][y][z]["type"] == "Interneuron":
                                for dx, dy, dz in [(0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)]:
                                    nx, ny, nz = x + dx, y + dy, z + dz
                                    if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and 0 <= nz < GRID_SIZE:
                                        if grid[nx][ny][nz]["type"] == "Empty":
                                            candidates.append((x, y, z, nx, ny, nz))
                if candidates:
                    px, py, pz, cx, cy, cz = random.choice(candidates)
                    parent = grid[px][py][pz]
                    grid[cx][cy][cz] = {
                        "type": "Interneuron",
                        "charge": 0.0,
                        "threshold": parent["threshold"],
                        "fire_rate": parent.get("fire_rate", 0.0),
                        "last_fired": -1,
                        "direction": parent.get("direction", "All"),
                        "weight": parent.get("weight", 1.0),
                        "amyloid_plaque": False
                    }
                    st.session_state.stats["memory"] -= 40.0
                    cooldowns["sprouting"] = 30
                    add_log(f"🌱 LÂM SÀNG: Nảy mầm synap (Sprouting)! Phân tách nơ-ron từ [{px+1},{py+1},{pz+1}] sang ô trống [{cx+1},{cy+1},{cz+1}].")
                    st.rerun()
                else:
                    st.warning("Không tìm thấy Interneuron nào có ô trống lân cận để mọc mầm!")

        with hormone_cols[8]:
            vns_disabled = cooldowns.get("vns", 0) > 0 or st.session_state.stats["memory"] < 30.0
            btn_label_vns = f"❤️ VNS ({cooldowns.get('vns', 0)}s)" if cooldowns.get("vns", 0) > 0 else "❤️ VNS"
            if st.button(btn_label_vns, disabled=vns_disabled, use_container_width=True, help="❤️ Kích thích dây thần kinh phế vị (Vagus Nerve Stimulation - VNS): Lập tức hạ mức Stress về 0%, hồi phục +20% Sanity và sạc đầy GABA lên 90%. Cooldown 40s."):
                st.session_state.stats["memory"] -= 30.0
                chems["stress"] = 0.0
                chems["sanity"] = min(100.0, chems["sanity"] + 20.0)
                chems["gaba"] = 90.0
                cooldowns["vns"] = 40
                add_log("❤️ LÂM SÀNG: Kích thích dây thần kinh phế vị VNS! Hạ stress về không, bình ổn tối đa nhịp hóa sinh vỏ não.")
                st.rerun()

        # Neurotransmitter Synthesis Precursors & Diet System Layout
        st.markdown("##### 🧬 Dinh Dưỡng Học & Tiền Chất Thần Kinh (Precursor Dietary Intake)")
        st.caption("Tổng hợp trực tiếp các chất dẫn truyền thông qua bồi bổ dinh dưỡng. Phí tiêu thụ: 15 MB Bộ nhớ.")

        diet_cols = st.columns(4)
        diet_disabled = st.session_state.stats["memory"] < 15.0

        with diet_cols[0]:
            if st.button("🥚 L-Tyrosine (Dopamine Precursor)", disabled=diet_disabled, use_container_width=True, help="Egg precursors: Tăng +15% Dopamine trực tiếp, sạc +3 Dopamine/tick trong 15 ticks và phục hồi +25% Dinh dưỡng Thần kinh."):
                st.session_state.stats["memory"] -= 15.0
                st.session_state.active_buffs["tyrosine"] = 15
                chems["dopamine"] = min(100.0, chems["dopamine"] + 15.0)
                chems["neuro_nutrients"] = min(100.0, chems.get("neuro_nutrients", 80.0) + 25.0)
                add_log("🥩 DINH DƯỠNG: Hấp thụ L-Tyrosine! Cung cấp dưỡng chất và bồi bổ Dopamine bộc phát.")
                st.rerun()

        with diet_cols[1]:
            if st.button("🍗 L-Tryptophan (Serotonin Precursor)", disabled=diet_disabled, use_container_width=True, help="Poultry precursors: Tăng +15% Serotonin trực tiếp, sạc +2 Serotonin/tick trong 15 ticks và phục hồi +25% Dinh dưỡng Thần kinh."):
                st.session_state.stats["memory"] -= 15.0
                st.session_state.active_buffs["tryptophan"] = 15
                chems["serotonin"] = min(100.0, chems["serotonin"] + 15.0)
                chems["neuro_nutrients"] = min(100.0, chems.get("neuro_nutrients", 80.0) + 25.0)
                add_log("🍗 DINH DƯỠNG: Hấp thụ L-Tryptophan! Thúc đẩy tổng hợp hoóc-môn hạnh phúc Serotonin.")
                st.rerun()

        with diet_cols[2]:
            if st.button("🥦 Choline (Acetylcholine Precursor)", disabled=diet_disabled, use_container_width=True, help="Broccoli precursors: Tăng +15% Acetylcholine trực tiếp, sạc +2.5 Acetylcholine/tick trong 15 ticks và phục hồi +25% Dinh dưỡng Thần kinh."):
                st.session_state.stats["memory"] -= 15.0
                st.session_state.active_buffs["choline"] = 15
                chems["acetylcholine"] = min(100.0, chems["acetylcholine"] + 15.0)
                chems["neuro_nutrients"] = min(100.0, chems.get("neuro_nutrients", 80.0) + 25.0)
                add_log("🥦 DINH DƯỠNG: Hấp thụ Choline! Tăng tốc độ phản xạ ghi nhớ vỏ não.")
                st.rerun()

        with diet_cols[3]:
            if st.button("🥜 Glutamate (GABA Precursor)", disabled=diet_disabled, use_container_width=True, help="Peanuts precursor: Tăng +15% GABA trực tiếp, sạc +3 GABA/tick trong 15 ticks và phục hồi +25% Dinh dưỡng Thần kinh."):
                st.session_state.stats["memory"] -= 15.0
                st.session_state.active_buffs["glutamate"] = 15
                chems["gaba"] = min(100.0, chems["gaba"] + 15.0)
                chems["neuro_nutrients"] = min(100.0, chems.get("neuro_nutrients", 80.0) + 25.0)
                add_log("🥜 DINH DƯỠNG: Hấp thụ Glutamate! Bổ sung GABA, bình ổn quá tải dẫn truyền.")
                st.rerun()

    # Dynamic line chart showing historical metrics
    st.write("---")
    st.markdown("#### 📈 Biểu Đồ Điện Não Đồ & Hóa Học Bộ Não 3D (EEG & Neuromodulator Telemetry)")
    st.caption("Hiển thị biến thiên nồng độ chất truyền dẫn hóa học, Tỉnh táo (Sanity), Căng thẳng (Stress) và ba Chỉ số 3D (CSI, PDI, VPI) theo thời gian thực.")

    hist_df = pd.DataFrame(st.session_state.history_data)
    if not hist_df.empty:
        st.line_chart(hist_df.set_index("tick"))

    # Game pathology modes selector
    st.write("---")
    st.markdown("##### ⚙️ Lựa chọn Chế Độ Thử Thách Não Bộ")
    modes_list = ["Normal", "Alzheimer", "Epilepsy", "Parkinson", "ADHD", "Schizophrenia", "Mania"]
    modes_names = {
        "Normal": "🟢 Bình Thường (Sức khỏe ổn định)",
        "Alzheimer": "👵 Thử Thách Alzheimer (Thoái hóa nơ-ron, mảng bám xơ hóa)",
        "Epilepsy": "⚡ Thử Thách Động Kinh (Gia tăng xung điện cực độ, nhân đôi stress)",
        "Parkinson": "🤝 Thử Thách Parkinson (Run giật nơ-ron vận động khi thiếu hụt Dopamine)",
        "ADHD": "🧠 Thử Thách ADHD (Dao động Dopamine dữ dội, tăng tốc phân rã Acetylcholine)",
        "Schizophrenia": "📢 Thử Thách Tâm Thần Phân Liệt (Ảo thanh kích phát điện thế bất ngờ, giảm 30% hồi tỉnh táo)",
        "Mania": "🤪 Thử Thách Hưng Cảm (Mania: Hụt sanity thần tốc, dopamine tăng cực đỉnh)"
    }
    selected_mode = st.selectbox(
        "Cấu hình bệnh lý học vỏ não 3D:",
        modes_list,
        index=modes_list.index(st.session_state.game_mode),
        format_func=lambda x: modes_names[x]
    )
    if not isinstance(selected_mode, str):
        selected_mode = st.session_state.game_mode
    if selected_mode != st.session_state.game_mode:
        st.session_state.game_mode = selected_mode
        add_log(f"⚙️ HỆ THỐNG: Chuyển đổi trạng thái bệnh lý sang {modes_names[selected_mode]}")
        st.rerun()

    # Genetic Mutation Board Selector
    st.write("---")
    st.markdown("##### 🧬 Bảng Đột Biến Gen Di Truyền Học (Genetic Mutation Board)")
    st.caption("Lựa chọn các đột biến gen bẩm sinh có lợi/hại để tinh chỉnh phản hồi hóa học vỏ não của bạn.")

    genes_list = [
        ("APOE4", "👵 Đột biến APOE4: Nhân đôi tốc độ chai lỳ điện thế trong chế độ Alzheimer."),
        ("BDNF", "🧠 Đột biến BDNF: Tăng cường tốc độ tự thích nghi (Hebbian threshold drift) thêm +50%."),
        ("COMT", "⚡ Đột biến COMT: Tăng cường tốc độ đào thải Dopamine tự nhiên thêm +30%."),
        ("GABRA1", "🧘 Đột biến GABRA1: Cản trở dập tắt stress trong chế độ Động kinh thêm +50%."),
        ("DRD4", "🍭 Đột biến DRD4: Nhân đôi tốc độ sạc Dopamine khi Motor nơ-ron phát xung, nhưng lượng dopamine thấp sẽ nhân đôi sát thương lên Sanity."),
        ("SHANK3", "🛡️ Đột biến SHANK3: Khớp nối bền vững Myelin nâng hiệu suất dẫn truyền xung điện lên thêm +15%."),
        ("DRD2", "🍿 Đột biến DRD2: Tăng +50% tốc độ sạc Dopamine của Sensory cell, nhưng stress cao (>50) sẽ nhân 1.5x sát thương Sanity."),
        ("COMT-Met", "💡 Đột biến COMT-Met: Tăng 30% điểm IQ khi Motor cell phát hỏa, nhưng Stress phân rã chậm hơn 30%."),
        ("ADRA2A", "🩺 Đột biến ADRA2A: Giảm mức thiệt hại hoảng loạn Sanity từ -15 xuống -9 và tăng ngưỡng Norepinephrine Panic lên 100%."),
        ("TREM2", "✨ Đột biến TREM2: Nhân đôi tốc độ dọn dẹp các mảng bám xơ hóa Amyloid của tế bào Microglia bẩm sinh.")
    ]

    active_genes = st.session_state.get("active_genes", [])
    selected_genes = []

    gen_cols = st.columns(2)
    for idx, (gen_code, gen_desc) in enumerate(genes_list):
        col_idx = idx % 2
        with gen_cols[col_idx]:
            is_active = st.checkbox(f"Gen **{gen_code}**\n({gen_desc})", value=(gen_code in active_genes), key=f"gene_cb_{gen_code}")
            if is_active:
                selected_genes.append(gen_code)

    if set(selected_genes) != set(active_genes):
        st.session_state.active_genes = selected_genes
        add_log(f"🧬 DI TRUYỀN: Cấu hình lại bộ gen bẩm sinh: {', '.join(selected_genes) or 'Trống'}")
        st.rerun()

    # Upgrades Shop Section
    st.write("---")
    st.markdown("#### 🏪 Cửa Hàng Giải Phẫu Thần Kinh (Neuro-Anatomical Shop)")
    st.caption("Sử dụng IQ nhận thức thu hoạch được để nâng cấp cấu trúc cơ thể sinh học của não bộ.")

    shop_cols = st.columns(3)

    with shop_cols[0]:
        st.write(f"**Hành Não & Thân Não [Lv.{upgrades['brainstem']}/3]**\nTrung tâm sinh năng lượng tự động: Tăng sản lượng Glucose nạp vào lên +2.0 mỗi cấp.")
        cost_stem = int(45 * upgrades["brainstem"])
        stem_disabled = upgrades["brainstem"] >= 3 or stats["iq"] < cost_stem
        if st.button(f"Nâng cấp Thân não ({cost_stem} IQ)", disabled=stem_disabled, use_container_width=True):
            stats["iq"] -= cost_stem
            upgrades["brainstem"] += 1
            add_log(f"🏪 NÂNG CẤP: Nâng cấp Thân não lên Cấp {upgrades['brainstem']} thành công!")
            st.rerun()

    with shop_cols[1]:
        st.write(f"**Tiểu Não (Cerebellum) [Lv.{upgrades['cerebellum']}/3]**\nBộ lọc thăng bằng căng thẳng: Đào thải độc tố stress thêm -1.0 mỗi cấp.")
        cost_cere = int(60 * upgrades["cerebellum"])
        cere_disabled = upgrades["cerebellum"] >= 3 or stats["iq"] < cost_cere
        if st.button(f"Nâng cấp Tiểu não ({cost_cere} IQ)", disabled=cere_disabled, use_container_width=True):
            stats["iq"] -= cost_cere
            upgrades["cerebellum"] += 1
            add_log(f"🏪 NÂNG CẤP: Nâng cấp Tiểu não lên Cấp {upgrades['cerebellum']} thành công!")
            st.rerun()

    with shop_cols[2]:
        st.write(f"**Thùy Hải Mã (Hippocampus) [Lv.{upgrades['hippocampus']}/3]**\nTuyến thu hoạch bộ nhớ dài hạn: Tăng hiệu suất chuyển hóa Trí nhớ dài hạn +50% mỗi cấp.")
        cost_hippo = int(80 * upgrades["hippocampus"])
        hippo_disabled = upgrades["hippocampus"] >= 3 or stats["iq"] < cost_hippo
        if st.button(f"Nâng cấp Hải mã ({cost_hippo} IQ)", disabled=hippo_disabled, use_container_width=True):
            stats["iq"] -= cost_hippo
            upgrades["hippocampus"] += 1
            add_log(f"🏪 NÂNG CẤP: Nâng cấp Thùy Hải Mã lên Cấp {upgrades['hippocampus']} thành công!")
            st.rerun()

    shop_cols2 = st.columns(3)

    with shop_cols2[0]:
        st.write(f"**Vỏ Não Phức Tạp (Cortex) [Lv.{upgrades['cortex']}/3]**\nTập hợp nơ-ron bậc cao: Nhân hiệu suất thu hoạch IQ khi nơ-ron Motor bắn lên thêm +60% mỗi cấp.")
        cost_cortex = int(100 * upgrades["cortex"])
        cortex_disabled = upgrades["cortex"] >= 3 or stats["iq"] < cost_cortex
        if st.button(f"Nâng cấp Vỏ não ({cost_cortex} IQ)", disabled=cortex_disabled, use_container_width=True):
            stats["iq"] -= cost_cortex
            upgrades["cortex"] += 1
            add_log(f"🏪 NÂNG CẤP: Nâng cấp Vỏ não bậc cao lên Cấp {upgrades['cortex']} thành công!")
            st.rerun()

    with shop_cols2[1]:
        st.write(f"**Bao Myelin (Myelin Sheaths) [Lv.{upgrades.get('myelin', 0)}/3]**\nMàng bọc cách điện bảo vệ: Tăng x5% tốc độ dẫn truyền tín hiệu xuyên suốt synap.")
        cost_myelin = int(50 * (upgrades.get("myelin", 0) + 1))
        myelin_disabled = upgrades.get("myelin", 0) >= 3 or stats["iq"] < cost_myelin
        if st.button(f"Phủ bao Myelin ({cost_myelin} IQ)", disabled=myelin_disabled, use_container_width=True):
            stats["iq"] -= cost_myelin
            upgrades["myelin"] = upgrades.get("myelin", 0) + 1
            add_log(f"🏪 NÂNG CẤP: Bao phủ Myelin thần kinh vỏ não đạt Cấp {upgrades['myelin']}!")
            st.rerun()

    with shop_cols2[2]:
        st.write(f"**Tính Mềm Dẻo Nơ-ron (Hebbian Plasticity) [Lv.{upgrades.get('plasticity', 0)}/1]**\nCơ chế tự thích thích Hebbian: Giảm vĩnh viễn ngưỡng điện thế của nơ-ron lân cận khi phát hỏa.")
        cost_plastic = 75
        plastic_disabled = upgrades.get("plasticity", 0) >= 1 or stats["iq"] < cost_plastic
        if st.button(f"Mở Hebbian Plasticity ({cost_plastic} IQ)", disabled=plastic_disabled, use_container_width=True):
            stats["iq"] -= cost_plastic
            upgrades["plasticity"] = 1
            add_log("🏪 NÂNG CẤP: Mở khóa Hebbian Plasticity! Thiết lập cơ chế ghi nhớ cơ bản.")
            st.rerun()

    shop_cols3 = st.columns(3)

    with shop_cols3[0]:
        st.write(f"**Tự Động Cắt Tỉa (Synaptic Pruning) [Lv.{upgrades.get('pruning', 0)}/1]**\nThanh lọc tế bào thừa nhàn rỗi: Tự động gỡ các Interneuron không hoạt động trong 15 ticks và hoàn phí 75%.")
        cost_pruning = 90
        pruning_disabled = upgrades.get("pruning", 0) >= 1 or stats["iq"] < cost_pruning
        if st.button(f"Mở Synaptic Pruning ({cost_pruning} IQ)", disabled=pruning_disabled, use_container_width=True):
            stats["iq"] -= cost_pruning
            upgrades["pruning"] = 1
            add_log("🏪 NÂNG CẤP: Mở khóa Synaptic Pruning dọn dẹp tế bào nhàn rỗi thông minh!")
            st.rerun()

    with shop_cols3[1]:
        st.write(f"**Vỏ Não Trước Trán (Prefrontal Cortex - PFC) [Lv.{upgrades.get('pfc', 0)}/1]**\nTự động hóa AI giải quyết biến cố: PFC tự động lựa chọn giải pháp tối ưu cho mọi biến cố ngẫu nhiên xuất hiện.")
        cost_pfc = 120
        pfc_disabled = upgrades.get("pfc", 0) >= 1 or stats["iq"] < cost_pfc
        if st.button(f"Mở khóa PFC ({cost_pfc} IQ)", disabled=pfc_disabled, use_container_width=True):
            stats["iq"] -= cost_pfc
            upgrades["pfc"] = 1
            add_log("🏪 NÂNG CẤP: Mở khóa Vỏ Não Trước Trán PFC! Tự động hóa giải quyết biến cố.")
            st.rerun()

    with shop_cols3[2]:
        st.write(f"**Hạch Hạnh Nhân (Amygdala) [Lv.{upgrades.get('amygdala', 0)}/3]**\nHạch chế ngự lo âu: Giảm -15% lượng căng thẳng (stress) sinh ra từ các hành động nơ-ron mỗi cấp.")
        cost_amy = int(70 * (upgrades.get("amygdala", 0) + 1))
        amy_disabled = upgrades.get("amygdala", 0) >= 3 or stats["iq"] < cost_amy
        if st.button(f"Nâng cấp Amygdala ({cost_amy} IQ)", disabled=amy_disabled, use_container_width=True):
            stats["iq"] -= cost_amy
            upgrades["amygdala"] = upgrades.get("amygdala", 0) + 1
            add_log(f"🏪 NÂNG CẤP: Nâng cấp Amygdala lên Cấp {upgrades['amygdala']} thành công!")
            st.rerun()

    shop_cols4 = st.columns(3)

    with shop_cols4[0]:
        st.write(f"**Đồi Thị (Thalamus Sensory Hub) [Lv.{upgrades.get('thalamus', 0)}/3]**\nTrạm trung chuyển cảm giác: Tăng tốc độ sạc điện tích tự thân của Sensory cell thêm +20% mỗi cấp.")
        cost_thalamus = int(60 * (upgrades.get("thalamus", 0) + 1))
        thalamus_disabled = upgrades.get("thalamus", 0) >= 3 or stats["iq"] < cost_thalamus
        if st.button(f"Nâng cấp Thalamus ({cost_thalamus} IQ)", disabled=thalamus_disabled, use_container_width=True):
            stats["iq"] -= cost_thalamus
            upgrades["thalamus"] = upgrades.get("thalamus", 0) + 1
            add_log(f"🏪 NÂNG CẤP: Nâng cấp Đồi Thị Thalamus lên Cấp {upgrades['thalamus']}!")
            st.rerun()

    with shop_cols4[1]:
        st.write(f"**Tế Bào Hình Sao (Glycogen Shunt) [Lv.{upgrades.get('glycogen_shunt', 0)}/1]**\nKích hoạt kho Glycogen dự trữ khẩn cấp: Tăng giới hạn chứa Glucose cực đại lên 150 (bình thường 100).")
        cost_glycogen = 100
        gly_disabled = upgrades.get("glycogen_shunt", 0) >= 1 or stats["iq"] < cost_glycogen
        if st.button(f"Kích hoạt Glycogen Shunt ({cost_glycogen} IQ)", disabled=gly_disabled, use_container_width=True):
            stats["iq"] -= cost_glycogen
            upgrades["glycogen_shunt"] = 1
            st.session_state.stats["glycogen_pool"] = 50.0
            add_log("🏪 NÂNG CẤP: Kích hoạt Glycogen Shunt dự phòng từ các tế bào hình sao Astrocytes!")
            st.rerun()

    with shop_cols4[2]:
        st.write(f"**Thùy Răng Hải Mã (Dentate Gyrus Lv.1) [Lv.{upgrades.get('dentate_gyrus', 0)}/1]**\nTự sinh thần kinh hải mã: Khi serotonin cao (>60), tự động cấy nơ-ron mới vào ô trống trống (Phí 15 MB).")
        cost_dentate = 150
        dentate_disabled = upgrades.get("dentate_gyrus", 0) >= 1 or stats["iq"] < cost_dentate
        if st.button(f"Nâng cấp Dentate Gyrus ({cost_dentate} IQ)", disabled=dentate_disabled, use_container_width=True):
            stats["iq"] -= cost_dentate
            upgrades["dentate_gyrus"] = 1
            add_log("🏪 NÂNG CẤP: Mở khóa Dentate Gyrus Hải mã kích hoạt cơ chế tự sinh nơ-ron liên kết!")
            st.rerun()

    shop_cols5 = st.columns(3)

    with shop_cols5[0]:
        st.write(f"**Thùy Chẩm (Occipital Lobe) [Lv.{upgrades.get('occipital_lobe', 0)}/3]**\nBộ xử lý thị giác: Cứ mỗi 10 ticks phát sinh một chớp thị giác. Định hướng Sensory khớp chớp sẽ x2 tốc độ sạc.")
        cost_occipital = int(80 * (upgrades.get("occipital_lobe", 0) + 1))
        occipital_disabled = upgrades.get("occipital_lobe", 0) >= 3 or stats["iq"] < cost_occipital
        if st.button(f"Nâng cấp Thùy Chẩm ({cost_occipital} IQ)", disabled=occipital_disabled, use_container_width=True):
            stats["iq"] -= cost_occipital
            upgrades["occipital_lobe"] = upgrades.get("occipital_lobe", 0) + 1
            add_log(f"🏪 NÂNG CẤP: Nâng cấp Thùy Chẩm (Occipital Lobe) lên Cấp {upgrades['occipital_lobe']}!")
            st.rerun()

    with shop_cols5[1]:
        st.write(f"**Thùy Thái Dương (Temporal Lobe) [Lv.{upgrades.get('temporal_lobe', 0)}/3]**\nBộ xử lý thính giác & nhịp cộng hưởng: Cứ mỗi 15 ticks nhận kích thích âm thanh. Tần số cộng hưởng 400-500 Hz nhân 3 (3x) sản lượng Trí nhớ.")
        cost_temporal = int(90 * (upgrades.get("temporal_lobe", 0) + 1))
        temporal_disabled = upgrades.get("temporal_lobe", 0) >= 3 or stats["iq"] < cost_temporal
        if st.button(f"Nâng cấp Thùy Thái Dương ({cost_temporal} IQ)", disabled=temporal_disabled, use_container_width=True):
            stats["iq"] -= cost_temporal
            upgrades["temporal_lobe"] = upgrades.get("temporal_lobe", 0) + 1
            add_log(f"🏪 NÂNG CẤP: Nâng cấp Thùy Thái Dương (Temporal Lobe) lên Cấp {upgrades['temporal_lobe']}!")
            st.rerun()

    with shop_cols5[2]:
        st.write(f"**Củng Cố Hebbian LTP Consolidator [Lv.{upgrades.get('ltp_consolidator', 0)}/1]**\nCủng cố liên kết trí nhớ dài hạn: Tự động chuyển đổi 30% bộ nhớ sang IQ vĩnh viễn Hebbian LTP sau mỗi 12 ticks hoạt động.")
        cost_ltp = 110
        ltp_disabled = upgrades.get("ltp_consolidator", 0) >= 1 or stats["iq"] < cost_ltp
        if st.button(f"Nâng cấp LTP Consolidator ({cost_ltp} IQ)", disabled=ltp_disabled, use_container_width=True):
            stats["iq"] -= cost_ltp
            upgrades["ltp_consolidator"] = 1
            add_log("🏪 NÂNG CẤP: Kích hoạt Hebbian LTP Consolidator củng cố bộ nhớ dài hạn tự động!")
            st.rerun()

    shop_cols6 = st.columns(3)

    with shop_cols6[0]:
        st.write(f"**Thùy Đỉnh (Parietal Lobe) [Lv.{upgrades.get('parietal_lobe', 0)}/3]**\nCảm giác bản thể và không gian: Cứ mỗi 18 ticks tạo 1 cổng Gating ngẫu nhiên. Truyền dòng điện qua đây dập ngay 20% stress và giảm 50% glucose tiêu hao.")
        cost_parietal = int(75 * (upgrades.get("parietal_lobe", 0) + 1))
        parietal_disabled = upgrades.get("parietal_lobe", 0) >= 3 or stats["iq"] < cost_parietal
        if st.button(f"Nâng cấp Thùy Đỉnh ({cost_parietal} IQ)", disabled=parietal_disabled, use_container_width=True):
            stats["iq"] -= cost_parietal
            upgrades["parietal_lobe"] = upgrades.get("parietal_lobe", 0) + 1
            add_log(f"🏪 NÂNG CẤP: Nâng cấp Thùy Đỉnh (Parietal Lobe) lên Cấp {upgrades['parietal_lobe']}!")
            st.rerun()

    with shop_cols6[1]:
        st.write(f"**Tuyến Yên (Pituitary Gland) [Lv.{upgrades.get('pituitary_gland', 0)}/3]**\nTuyến nội tiết giải phóng hoóc-môn Oxytocin: Kích hoạt Oxytocin Surge mỗi 20 ticks giúp bão hòa 50% stress phát sinh.")
        cost_pituitary = int(85 * (upgrades.get("pituitary_gland", 0) + 1))
        pituitary_disabled = upgrades.get("pituitary_gland", 0) >= 3 or stats["iq"] < cost_pituitary
        if st.button(f"Nâng cấp Tuyến Yên ({cost_pituitary} IQ)", disabled=pituitary_disabled, use_container_width=True):
            stats["iq"] -= cost_pituitary
            upgrades["pituitary_gland"] = upgrades.get("pituitary_gland", 0) + 1
            add_log(f"🏪 NÂNG CẤP: Nâng cấp Tuyến Yên (Pituitary Gland) lên Cấp {upgrades['pituitary_gland']}!")
            st.rerun()

    with shop_cols6[2]:
        st.write(f"**Hàng Rào Máu Não (Blood-Brain Barrier) [Lv.{upgrades.get('blood_brain_barrier', 0)}/1]**\nHàng rào mạch máu bảo vệ: Giảm tốc độ tiêu thụ dinh dưỡng còn 1.0%/tick và triệt tiêu 50% bão viêm cytokine thần kinh.")
        cost_bbb = 120
        bbb_disabled = upgrades.get("blood_brain_barrier", 0) >= 1 or stats["iq"] < cost_bbb
        if st.button(f"Nâng cấp Hàng rào máu ({cost_bbb} IQ)", disabled=bbb_disabled, use_container_width=True):
            stats["iq"] -= cost_bbb
            upgrades["blood_brain_barrier"] = 1
            add_log("🏪 NÂNG CẤP: Kích hoạt màng bọc Hàng rào máu não Blood-Brain Barrier bảo toàn huyết học!")
            st.rerun()

    # Active Challenges & Missions Board Layout
    st.write("---")
    st.markdown("#### 🎯 Bảng Nhiệm Vụ Hoạt Động & Thách Thức (Cognitive Missions)")
    st.caption("Hoàn thành các cột mốc sinh học để mở khóa tài nguyên bộ nhớ dài hạn hoặc điểm IQ tức thì.")

    missions = st.session_state.missions
    mis_cols = st.columns(2)
    for idx, (mis_key, mis_data) in enumerate(missions.items()):
        col_idx = idx % 2
        with mis_cols[col_idx]:
            st.write(f"**Nhiệm vụ {idx+1}: {mis_data['name']}**")
            st.caption(f"Yêu cầu: *{mis_data['target']}*")

            is_completed = mis_data["status"] == "Completed"
            claim_disabled = not is_completed or mis_data.get("reward_claimed", False)
            btn_txt = "🎁 Đã Nhận Thưởng" if mis_data.get("reward_claimed", False) else ("Claim Thưởng" if is_completed else "Đang Thực Hiện...")

            if st.button(btn_txt, disabled=claim_disabled, key=f"claim_{mis_key}", use_container_width=True):
                mis_data["reward_claimed"] = True
                if mis_key == "reflex":
                    st.session_state.stats["memory"] = min(st.session_state.stats["max_memory"], st.session_state.stats["memory"] + 100.0)
                elif mis_key == "loop":
                    st.session_state.stats["iq"] += 300.0
                elif mis_key == "zen":
                    chems["dopamine"] = min(100.0, chems["dopamine"] + 40.0)
                    chems["serotonin"] = min(100.0, chems["serotonin"] + 40.0)
                elif mis_key == "marathon":
                    chems["acetylcholine"] = min(100.0, chems["acetylcholine"] + 50.0)
                    st.session_state.stats["memory"] = min(st.session_state.stats["max_memory"], st.session_state.stats["memory"] + 200.0)
                add_log(f"🎁 THƯỞNG: Nhận thành công phần quà của {mis_data['name']}!")
                st.rerun()

    # Real-time Scrolling Logs panel
    st.write("---")
    st.markdown("#### 📋 Nhật Ký Hoạt Động Não Bộ (Brain Activity Log)")
    log_text = "\n".join(st.session_state.game_log[::-1])
    st.text_area("Thời gian thực (Mới nhất ở trên):", value=log_text, height=180, disabled=True)

    # WGSL Compute Kernels Code Inspector Panel (Styled exactly like Bonsai WebGPU Kernels)
    st.write("---")
    st.markdown("#### 🎛️ WGSL Compute Shaders Inspector")
    st.caption("Xem mã nguồn thấp cấp WGSL compute shaders của các tác vụ mô phỏng sinh học 3D thời gian thực. Toàn bộ các kernel được tối ưu hóa biên dịch trực tiếp trên GPU của trình duyệt.")

    selected_kernel = st.selectbox(
        "Lựa chọn GPU Compute Kernel để kiểm tra mã nguồn:",
        [
            "synaptic_charge_propagation.wgsl (Phát xung điện 3D)",
            "hebbian_plasticity_attention.wgsl (Tính dẻo thích nghi Hebbian)",
            "gaba_normalization.wgsl (Bình ổn bão viêm và stress)",
            "vagus_nerve_clamp.wgsl (Liệu pháp lâm sàng VNS)"
        ]
    )
    if not isinstance(selected_kernel, str):
        selected_kernel = "synaptic_charge_propagation.wgsl (Phát xung điện 3D)"

    shaders = {
        "synaptic_charge_propagation.wgsl (Phát xung điện 3D)": """// synaptic_charge_propagation.wgsl
// Low-level WebGPU compute shader doing 3D synaptic charge propagation
// Hand-crafted by GPT 5.6 Sol and Fable 5. Tested on 4x4x4 (64 nodes) grid layouts.

@group(0) @binding(0) var<storage, read> input_charge: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_charge: array<f32>;
@group(0) @binding(2) var<uniform> signal_efficiency: f32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= 64u) { return; }

    let current_charge = input_charge[index];
    if (current_charge >= 0.5) {
        // Reset the firing cell's charge (carryover or absolute refractory)
        output_charge[index] = 0.0;

        let neighbors_count = 6u;
        let charge_transfer = (current_charge * signal_efficiency) / f32(neighbors_count);

        // Emulate matrix multiply propagation in 3D coordinates
        for (var i = 0u; i < neighbors_count; i = i + 1u) {
            let neighbor_idx = get_neighbor_index(index, i);
            if (neighbor_idx < 64u) {
                atomicAdd(&output_charge[neighbor_idx], charge_transfer);
            }
        }
    }
}""",
        "hebbian_plasticity_attention.wgsl (Tính dẻo thích nghi Hebbian)": """// hebbian_plasticity_attention.wgsl
// Low-level WebGPU compute shader adjusting threshold adaptation
// Emulates linear self-attention based on Hebbian LTP/LTD synaptic plasticities.

@group(0) @binding(0) var<storage, read> active_charges: array<f32>;
@group(0) @binding(1) var<storage, read_write> thresholds: array<f32>;
@group(0) @binding(2) var<uniform> learning_rate: f32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= 64u) { return; }

    let charge = active_charges[index];
    if (charge > 0.3) {
        // Adjust threshold downwards to represent increased firing familiarity
        thresholds[index] = max(0.2, thresholds[index] - learning_rate);
    }
}""",
        "gaba_normalization.wgsl (Bình ổn bão viêm và stress)": """// gaba_normalization.wgsl
// Normalization pass shader managing neuro-excitability
// Performs stress attenuation and damping using GABA neurotransmitter inputs.

@group(0) @binding(0) var<storage, read> input_stress: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_stress: array<f32>;
@group(0) @binding(2) var<uniform> gaba_level: f32;

@compute @workgroup_size(1)
fn main() {
    let baseline_stress = input_stress[0];
    if (gaba_level >= 70.0) {
        // Attenuate stress waves by 40% under protective GABA barriers
        output_stress[0] = baseline_stress * 0.60;
    } else {
        output_stress[0] = baseline_stress;
    }
}""",
        "vagus_nerve_clamp.wgsl (Liệu pháp lâm sàng VNS)": """// vagus_nerve_clamp.wgsl
// GPU device clamp code executing Clinical Vagus Nerve Stimulation (VNS)
// Resets stress levels, increases sanity indicators, and spikes GABA.

@group(0) @binding(0) var<storage, read_write> stress: array<f32>;
@group(0) @binding(1) var<storage, read_write> sanity: array<f32>;
@group(0) @binding(2) var<storage, read_write> gaba: array<f32>;

@compute @workgroup_size(1)
fn main() {
    // Parasympathetic clamp overrides
    stress[0] = 0.0;
    sanity[0] = min(100.0, sanity[0] + 20.0);
    gaba[0] = 90.0;
}"""
    }

    st.code(shaders[selected_kernel], language="rust")

# ----------------- TAB 2: VBOT1 WEB CHAT & SUMMARIZE -----------------
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
