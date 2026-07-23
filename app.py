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

GRID_SIZE = 6

def serialize_grid(grid):
    cells = []
    dir_map = {"All": "A", "Up": "U", "Right": "R", "Down": "D", "Left": "L"}
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            cell = grid[r][c]
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
    if len(parts) != GRID_SIZE * GRID_SIZE:
        return None

    type_map = {"E": "Empty", "S": "Sensory", "I": "Interneuron", "M": "Motor"}
    dir_map = {"A": "All", "U": "Up", "R": "Right", "D": "Down", "L": "Left"}

    new_grid = []
    idx = 0
    for r in range(GRID_SIZE):
        row = []
        for c in range(GRID_SIZE):
            part = parts[idx]
            if len(part) < 2:
                return None
            t_char, d_char = part[0], part[1]
            weight_val = 1.0
            if len(part) >= 3:
                try:
                    weight_val = float(part[2])
                except ValueError:
                    weight_val = 1.0
            t_name = type_map.get(t_char, "Empty")
            d_name = dir_map.get(d_char, "All")

            row.append({
                "type": t_name,
                "charge": 0.0,
                "threshold": 0.4 if t_name == "Sensory" else (0.6 if t_name == "Motor" else 0.5),
                "fire_rate": 0.3 if t_name == "Sensory" else 0.0,
                "last_fired": -1,
                "direction": d_name,
                "weight": weight_val
            })
            idx += 1
        new_grid.append(row)
    return new_grid

def init_game_state():
    if "game_initialized" not in st.session_state:
        st.session_state.game_initialized = True

        # Grid Initialization: 6x6 Matrix of dicts
        grid = []
        for r in range(GRID_SIZE):
            row = []
            for c in range(GRID_SIZE):
                row.append({
                    "type": "Empty",
                    "charge": 0.0,
                    "threshold": 0.5,
                    "fire_rate": 0.25,
                    "last_fired": -1,
                    "direction": "All",
                    "weight": 1.0
                })
            grid.append(row)

        # Default starting network structure
        grid[0][0] = {"type": "Sensory", "charge": 0.0, "threshold": 0.4, "fire_rate": 0.35, "last_fired": -1, "direction": "All", "weight": 1.0}
        grid[2][2] = {"type": "Interneuron", "charge": 0.0, "threshold": 0.5, "fire_rate": 0.0, "last_fired": -1, "direction": "All", "weight": 1.0}
        grid[5][5] = {"type": "Motor", "charge": 0.0, "threshold": 0.6, "fire_rate": 0.0, "last_fired": -1, "direction": "All", "weight": 1.0}

        st.session_state.neuron_grid = grid

        # Chemistry metrics (added melatonin, neuro_inflammation, norepinephrine, gaba)
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
            "gaba": 30.0
        }

        # Advanced Game Modes (Normal, Alzheimer, Epilepsy, Parkinson, ADHD)
        st.session_state.game_mode = "Normal"

        # UPGRADE: Genetic Mutation Board Selection (supports PGC-1alpha & SLC6A4, DRD2, COMT-Met)
        st.session_state.active_genes = []

        # UPGRADE: Continuous Active Buffs tick counter
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
            "cortisol": 0, # clinical anti-inflammatory wash cooldown
            "propranolol": 0, # clinical beta-blocker cooldown
            "sprouting": 0 # clinical synaptic sprouting cooldown
        }

        # Audio Synthesizer Triggers
        st.session_state.audio_trigger = None

        # Challenges & Missions System
        st.session_state.missions = {
            "reflex": {"name": "⚡ Cung Phản Xạ Sinh Học", "target": "Đặt ít nhất 1 Sensory và 1 Motor trên lưới", "status": "In Progress", "reward_claimed": False, "desc": "+100 MB Trí nhớ"},
            "loop": {"name": "🧠 Vòng Lặp Phản Hồi Tự Trị", "target": "Đặt ít nhất 6 nơ-ron hoạt động trên lưới", "status": "In Progress", "reward_claimed": False, "desc": "+300 IQ"},
            "zen": {"name": "🧘 Thiền Tĩnh Tâm Trị Liệu", "target": "Căng thẳng dưới 5% và Tỉnh táo trên 95%", "status": "In Progress", "reward_claimed": False, "desc": "+40 Dopamine & +40 Serotonin"},
            "marathon": {"name": "🏆 Chạy Đua Nhận Thức Siêu Phàm", "target": "Tích lũy tối thiểu 500 điểm IQ nhận thức", "status": "In Progress", "reward_claimed": False, "desc": "+50 Acetylcholine & +200 MB Trí nhớ"}
        }

        # Progression and stats
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
            "dentate_gyrus": 0, # Dentate Gyrus Lv.0
            "occipital_lobe": 0,
            "temporal_lobe": 0,
            "ltp_consolidator": 0,
            "parietal_lobe": 0,
            "pituitary_gland": 0
        }

        # Local Circuit Save Slots Library
        st.session_state.save_slots = {
            "Slot 1": None,
            "Slot 2": None,
            "Slot 3": None
        }

        # Logs, playing status, speed, selected cell, events
        st.session_state.game_log = ["Khởi tạo bộ não thành công. Trạng thái tiến hóa: Hành não Bò sát."]
        st.session_state.playing = False
        st.session_state.tick_speed = 1.0
        st.session_state.selected_cell = (0, 0)
        st.session_state.current_event = None
        st.session_state.history_data = {
            "tick": [0],
            "sanity": [100.0],
            "energy": [100.0],
            "dopamine": [50.0],
            "stress": [10.0],
            "norepinephrine": [10.0],
            "gaba": [30.0]
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
                    "label": "Espresso kép (Double Shot)",
                    "effect": "Dopamine +30, Acetylcholine +25, Căng thẳng +25, Năng lượng +30",
                    "apply": lambda: apply_event_effects(30, 25, 25, 30, 0, 0, "Uống espresso siêu đậm! Đầu óc cực kỳ hưng phấn nhưng nhịp tim tăng nhanh.")
                },
                {
                    "label": "Trà xanh thanh ngọt",
                    "effect": "Acetylcholine +15, Serotonin +15, Căng thẳng -10",
                    "apply": lambda: apply_event_effects(0, 15, -10, 0, 15, 0, "Thưởng trà thanh tĩnh. Tăng độ tập trung ôn hòa.")
                },
                {
                    "label": "Thôi, uống nước lọc",
                    "effect": "Năng lượng +10, Căng thẳng -5",
                    "apply": lambda: apply_event_effects(0, 0, -5, 10, 0, 0, "Chọn nước lọc tinh khiết. Năng lượng hồi phục nhẹ nhàng.")
                }
            ]
        },
        {
            "title": "📚 Đêm Trước Kỳ Thi Học Thuật",
            "desc": "Kỳ thi Olympic Sinh học sắp diễn ra vào sáng mai. Bạn sẽ làm gì để phân bổ tài nguyên não?",
            "choices": [
                {
                    "label": "Cày cuốc xuyên đêm",
                    "effect": "IQ +120, Trí nhớ +50, Năng lượng -40, Căng thẳng +30, Tỉnh táo -20",
                    "apply": lambda: apply_event_effects(15, 20, 30, -40, 0, -20, "Nhồi nhét kiến thức thâu đêm! IQ và Bộ nhớ tăng vọt, nhưng cơ thể cực kỳ mệt mỏi.", 120.0, 50.0)
                },
                {
                    "label": "Ôn lướt nhẹ, ngủ đủ",
                    "effect": "IQ +40, Trí nhớ +15, Năng lượng +15, Căng thẳng -5",
                    "apply": lambda: apply_event_effects(5, 5, -5, 15, 5, 5, "Học tập điều độ kết hợp ngủ ngon. Não bộ hấp thu kiến thức rất tốt.", 40.0, 15.0)
                },
                {
                    "label": "Đi ngủ sớm thanh thản",
                    "effect": "Tỉnh táo +25, Năng lượng +40, Căng thẳng -15",
                    "apply": lambda: apply_event_effects(0, 0, -15, 40, 0, 25, "Ngủ một mạch 9 tiếng. Não bộ được dọn dẹp sạch độc tố và phục hồi hoàn toàn.")
                }
            ]
        },
        {
            "title": "📱 Cơn Nghiện TikTok / Reels",
            "desc": "Bạn cầm điện thoại lên định kiểm tra tin nhắn nhưng vô tình lọt vào vòng lặp video ngắn vô tận.",
            "choices": [
                {
                    "label": "Lướt thêm 1 tiếng",
                    "effect": "Dopamine +45, Acetylcholine -20, Năng lượng -15, Căng thẳng +10",
                    "apply": lambda: apply_event_effects(45, -20, 10, -15, 0, -5, "Hệ thống phần thưởng bị kích thích mạnh mẽ, nhưng mức độ tập trung sụt giảm nghiêm trọng.")
                },
                {
                    "label": "Tắt máy đọc sách giấy",
                    "effect": "Trí nhớ +10, Acetylcholine +25, Dopamine -10",
                    "apply": lambda: apply_event_effects(-10, 25, -5, 5, 10, 5, "Cách ly thiết bị điện tử giúp tái cấu trúc nơ-ron, tăng khả năng lưu trữ sâu.", 10.0, 10.0)
                }
            ]
        },
        {
            "title": "🧘 Thiền Định Sâu",
            "desc": "Bạn thực hành kỹ thuật thiền định Chánh niệm (Mindfulness) để làm lắng dịu các sóng não dao động.",
            "choices": [
                {
                    "label": "Thiền Vipassana 20 phút",
                    "effect": "Căng thẳng -35, Serotonin +30, Tỉnh táo +20, Năng lượng -10",
                    "apply": lambda: apply_event_effects(-10, 15, -35, -10, 30, 20, "Tập trung vào hơi thở giúp dập tắt căng thẳng dư thừa, tăng nồng độ Serotonin.")
                },
                {
                    "label": "Chỉ nằm thư giãn nhẹ",
                    "effect": "Căng thẳng -15, Năng lượng +15",
                    "apply": lambda: apply_event_effects(0, 0, -15, 15, 5, 5, "Để đầu óc trống rỗng trong chốc lát. Cơ thể tự cân bằng năng lượng.")
                }
            ]
        }
    ]
    st.session_state.current_event = random.choice(events)

def apply_event_effects(da, ach, stress, energy, se, sanity, log_msg, iq_gain=0.0, mem_gain=0.0):
    st.session_state.chemicals["dopamine"] = max(0.0, min(100.0, st.session_state.chemicals["dopamine"] + da))
    st.session_state.chemicals["acetylcholine"] = max(0.0, min(100.0, st.session_state.chemicals["acetylcholine"] + ach))
    st.session_state.chemicals["stress"] = max(0.0, min(100.0, st.session_state.chemicals["stress"] + stress))
    st.session_state.chemicals["energy"] = max(0.0, min(100.0, st.session_state.chemicals["energy"] + energy))
    st.session_state.chemicals["serotonin"] = max(0.0, min(100.0, st.session_state.chemicals["serotonin"] + se))
    st.session_state.chemicals["sanity"] = max(0.0, min(100.0, st.session_state.chemicals["sanity"] + sanity))

    st.session_state.stats["iq"] += iq_gain
    st.session_state.stats["memory"] += mem_gain

    add_log(f"⚡ [Sự kiện] {log_msg}")
    st.session_state.current_event = None

# Check mission completions dynamically
def check_mission_statuses():
    grid = st.session_state.neuron_grid
    chems = st.session_state.chemicals
    stats = st.session_state.stats
    missions = st.session_state.missions

    has_sensory = False
    has_motor = False
    active_count = 0
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if grid[r][c]["type"] == "Sensory":
                has_sensory = True
                active_count += 1
            elif grid[r][c]["type"] == "Motor":
                has_motor = True
                active_count += 1
            elif grid[r][c]["type"] != "Empty":
                active_count += 1

    if has_sensory and has_motor and missions["reflex"]["status"] == "In Progress":
        missions["reflex"]["status"] = "Completed"
        add_log("🏆 NHIỆM VỤ ĐẠT: 'Cung Phản Xạ Sinh Học' đã sẵn sàng nhận thưởng!")

    if active_count >= 6 and missions["loop"]["status"] == "In Progress":
        missions["loop"]["status"] = "Completed"
        add_log("🏆 NHIỆM VỤ ĐẠT: 'Vòng Lặp Phản Hồi Tự Trị' đã sẵn sàng nhận thưởng!")

    if chems["stress"] <= 5.0 and chems["sanity"] >= 95.0 and missions["zen"]["status"] == "In Progress":
        missions["zen"]["status"] = "Completed"
        add_log("🏆 NHIỆM VỤ ĐẠT: 'Thiền Tĩnh Tâm Trị Liệu' đã sẵn sàng nhận thưởng!")

    if stats["iq"] >= 500.0 and missions["marathon"]["status"] == "In Progress":
        missions["marathon"]["status"] = "Completed"
        add_log("🏆 NHIỆM VỤ ĐẠT: 'Chạy Đua Nhận Thức Siêu Phàm' đã sẵn sàng nhận thưởng!")

def add_log(msg):
    st.session_state.game_log.insert(0, f"[{st.session_state.stats['ticks']}] {msg}")
    if len(st.session_state.game_log) > 50:
        st.session_state.game_log.pop()

def run_simulation_tick():
    st.session_state.stats["ticks"] += 1
    ticks = st.session_state.stats["ticks"]

    grid = st.session_state.neuron_grid
    chems = st.session_state.chemicals
    upgrades = st.session_state.upgrades
    mode = st.session_state.get("game_mode", "Normal")
    genes = st.session_state.get("active_genes", [])

    # UPGRADE: Occipital Lobe Visual Sensory Spark Game
    if upgrades.get("occipital_lobe", 0) >= 1 and ticks % 10 == 0:
        sensory_cells = []
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if grid[r][c]["type"] == "Sensory":
                    sensory_cells.append((r, c))
        if sensory_cells:
            sr, sc = random.choice(sensory_cells)
            target_dir = random.choice(["Up", "Right", "Down", "Left"])
            st.session_state.visual_spark = {"pos": (sr, sc), "dir": target_dir}
            add_log(f"👁️ [Thùy Chẩm] Kích thích thị giác tại Sensory [{sr+1},{sc+1}]! Khớp hướng '{target_dir}' để kích hoạt 2x sạc nơ-ron.")

    # UPGRADE: Temporal Lobe Auditory Stimulation System
    if upgrades.get("temporal_lobe", 0) >= 1 and ticks % 15 == 0:
        auditory_freq = random.randint(200, 1000)
        st.session_state.auditory_freq = auditory_freq
        resonance = "Khớp cộng hưởng! 3x Memory" if (400 <= auditory_freq <= 500) else "Bình thường"
        add_log(f"🎵 [Thùy Thái Dương] Kích thích thính giác: Tần số {auditory_freq} Hz ({resonance})!")

    # UPGRADE: LTP Memory Consolidator (Long-Term Potentiation)
    if upgrades.get("ltp_consolidator", 0) >= 1 and ticks % 12 == 0:
        current_mem = st.session_state.stats["memory"]
        if current_mem > 10.0:
            consolidated_mem = current_mem * 0.30
            st.session_state.stats["memory"] -= consolidated_mem
            st.session_state.stats["iq"] += consolidated_mem
            add_log(f"💾 [LTP Consolidator] Tự động củng cố Hebbian LTP chuyển đổi 30% Trí nhớ sang IQ (+{consolidated_mem:.1f} IQ/MB)!")

    # UPGRADE: Parietal Lobe Spatial Prophecy Game
    if upgrades.get("parietal_lobe", 0) >= 1 and ticks % 18 == 0:
        pr = random.randint(0, GRID_SIZE-1)
        pc = random.randint(0, GRID_SIZE-1)
        st.session_state.spatial_gate = (pr, pc)
        add_log(f"🧭 [Thùy Đỉnh] Luồng định vị không gian: Điểm Gating xuất hiện tại nơ-ron [{pr+1},{pc+1}]! Truyền điện thế qua đây để kích hoạt giảm stress và năng lượng.")

    # UPGRADE: Pituitary Gland (Tuyến Yên) Oxytocin Surge Game
    if upgrades.get("pituitary_gland", 0) >= 1 and ticks % 20 == 0:
        st.session_state.active_buffs["oxytocin"] = 5
        add_log("🧠 [Tuyến Yên] Kích hoạt giải phóng Oxytocin Surge! Giảm 50% Stress phát sinh và nhân đôi tốc độ hồi phục Tỉnh táo trong 5 ticks.")

    # Increment burnout-free streak
    st.session_state.stats["burnout_streak"] += 1
    st.session_state.stats["max_streak"] = max(st.session_state.stats["max_streak"], st.session_state.stats["burnout_streak"])

    # UPGRADE: Neurodevelopmental ADHD Pathology Mode
    # ADHD wildly fluctuates Dopamine baseline levels randomly and accelerates Acetylcholine decay
    if mode == "ADHD":
        da_fluctuate = random.uniform(-15.0, 15.0)
        chems["dopamine"] = max(0.0, min(100.0, chems["dopamine"] + da_fluctuate))
        if da_fluctuate > 10.0:
            add_log(f"🧠 [ADHD] Dao động Dopamine bất thường đột ngột (+{da_fluctuate:.1f}%) gây tăng động tăng tập trung ngắn hạn!")
        elif da_fluctuate < -10.0:
            add_log(f"🧠 [ADHD] Dopamine sụt giảm đột ngột ({da_fluctuate:.1f}%) làm mất khả năng tập trung sâu!")

    # Day/Night Circadian Rhythm (30 Day, 10 Night)
    cycle_time = ticks % 40
    if cycle_time < 30:
        st.session_state.stats["circadian_cycle"] = "Day"
        chems["melatonin"] = max(0.0, chems["melatonin"] - 1.5)
    else:
        st.session_state.stats["circadian_cycle"] = "Night"
        chems["melatonin"] = min(100.0, chems["melatonin"] + 4.5)
        # Night melatonin slightly lowers baseline stress
        chems["stress"] = max(0.0, chems["stress"] - 1.0)

    # SLC6A4 gene increases SSRI active buff duration x1.5
    slc6a4_mult = 1.5 if "SLC6A4" in genes else 1.0

    # Decrement active neuromodulator buffs tick timers and apply ongoing biochemical bonuses
    buffs = st.session_state.get("active_buffs", {"doping": 0, "ssri": 0, "focus": 0, "tyrosine": 0, "tryptophan": 0, "choline": 0, "glutamate": 0, "somatosensory_gating": 0, "oxytocin": 0})
    for k in list(buffs.keys()):
        if buffs[k] > 0:
            buffs[k] -= 1
            if k == "doping":
                chems["dopamine"] = min(100.0, chems["dopamine"] + 5.0)
            elif k == "ssri":
                chems["serotonin"] = min(100.0, chems["serotonin"] + 3.0)
            elif k == "focus":
                chems["acetylcholine"] = min(100.0, chems["acetylcholine"] + 4.0)
            # Dietary Synthesis Precursor Tick Bonuses
            elif k == "tyrosine":
                chems["dopamine"] = min(100.0, chems["dopamine"] + 3.0)
            elif k == "tryptophan":
                chems["serotonin"] = min(100.0, chems["serotonin"] + 2.0)
            elif k == "choline":
                chems["acetylcholine"] = min(100.0, chems["acetylcholine"] + 2.5)
            elif k == "glutamate":
                chems["gaba"] = min(100.0, chems.get("gaba", 30.0) + 3.0)

    # Decrement active hormone abilities cooldowns
    cooldowns = st.session_state.get("cooldowns", {"doping": 0, "ssri": 0, "focus": 0, "rtms": 0, "opto": 0, "cortisol": 0})
    for k in cooldowns:
        if cooldowns[k] > 0:
            cooldowns[k] -= 1

    # Alzheimer Pathology Tick
    if mode == "Alzheimer" and ticks % 10 == 0:
        degraded = 0
        # UPGRADE: APOE4 Gene Mutation doubles threshold drift speed
        drift_rate = 0.08 if "APOE4" in genes else 0.04
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                cell = grid[r][c]
                if cell["type"] != "Empty" and cell["threshold"] < 0.9:
                    cell["threshold"] = min(0.9, cell["threshold"] + drift_rate)
                    degraded += 1
        if degraded > 0:
            add_log(f"🧠 [Alzheimer] Suy giảm nhận thức khiến {degraded} nơ-ron bị chai lỳ, tăng ngưỡng kích hoạt (+{drift_rate})!")

    # UPGRADE: Amyloid-Beta Plaque Accumulation in Alzheimer's Mode
    if mode == "Alzheimer" and random.random() < 0.25:
        # 25% chance per tick of placing a plaque on a random non-empty cell that doesn't have one yet
        non_plaque_cells = []
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                cell = grid[r][c]
                if cell["type"] != "Empty" and not cell.get("amyloid_plaque", False):
                    non_plaque_cells.append((r, c))
        if non_plaque_cells:
            pr, pc = random.choice(non_plaque_cells)
            grid[pr][pc]["amyloid_plaque"] = True
            add_log(f"🧬 [Amyloid-Beta] Mảng bám Amyloid tích tụ trên nơ-ron [{pr+1},{pc+1}], giảm 50% hiệu suất truyền dẫn!")

    # UPGRADE: Microglia Phagocytosis (Plaque clearing)
    if chems["serotonin"] > 60.0 and chems["acetylcholine"] > 60.0:
        clear_chance = 0.40 if "TREM2" in genes else 0.20
        if random.random() < clear_chance:
            plaque_cells = []
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    if grid[r][c].get("amyloid_plaque", False):
                        plaque_cells.append((r, c))
            if plaque_cells:
                cr, cc = random.choice(plaque_cells)
                grid[cr][cc]["amyloid_plaque"] = False
                add_log(f"🧹 [Thực bào Microglia] Tế bào thần kinh đệm dọn sạch mảng bám Amyloid tại [{cr+1},{cc+1}]!")

    # UPGRADE: Chronic Neuro-Inflammation threshold drift (Cytokine Storm)
    # Inflammation above 80% causes slow, chronic neural threshold drift
    if chems.get("neuro_inflammation", 0.0) > 80.0 and ticks % 8 == 0:
        degraded = 0
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                cell = grid[r][c]
                if cell["type"] != "Empty" and cell["threshold"] < 0.9:
                    cell["threshold"] = min(0.9, cell["threshold"] + 0.02)
                    degraded += 1
        if degraded > 0:
            add_log(f"🔥 [Bão Cytokine] Viêm thần kinh cực cao (>80%) gây chai lỳ, xơ hóa và tăng ngưỡng kích hoạt {degraded} tế bào (+0.02)!")

    # UPGRADE: Dentate Gyrus Neurogenesis Anatomy Upgrade
    # Every tick, if Dentate Gyrus is upgraded and serotonin is high, auto-implant Interneuron at random empty cells
    if upgrades.get("dentate_gyrus", 0) >= 1:
        if st.session_state.stats["memory"] >= 30.0 and chems["serotonin"] > 60.0:
            if random.random() < 0.25:
                empty_cells = []
                for r in range(GRID_SIZE):
                    for c in range(GRID_SIZE):
                        if grid[r][c]["type"] == "Empty":
                            empty_cells.append((r, c))
                if empty_cells:
                    sp_r, sp_c = random.choice(empty_cells)
                    st.session_state.stats["memory"] -= 15.0
                    grid[sp_r][sp_c] = {
                        "type": "Interneuron",
                        "charge": 0.0,
                        "threshold": 0.5,
                        "fire_rate": 0.0,
                        "last_fired": -1,
                        "direction": "All",
                        "weight": 1.0
                    }
                    add_log(f"🌱 [Hải Mã Neurogenesis] Thùy răng (Dentate Gyrus) tự động sản sinh tế bào liên kết mới tại [{sp_r+1},{sp_c+1}]! (-15 MB Memory)")

    # Schizophrenia Pathology Tick
    # Trigger Auditory Hallucinations setting a random cell charge directly to threshold
    if mode == "Schizophrenia" and ticks % 8 == 0:
        non_empty = []
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if grid[r][c]["type"] != "Empty":
                    non_empty.append((r, c))
        if non_empty:
            hr, hc = random.choice(non_empty)
            grid[hr][hc]["charge"] = grid[hr][hc]["threshold"]
            add_log(f"📢 [Tâm thần phân liệt] Ảo thanh kích hoạt đột ngột tại nơ-ron [{hr+1},{hc+1}]!")

    # Parkinson's Pathology Tick
    # Low Dopamine causes random Motor cells to misfire (tremors), draining energy but yielding 0 IQ/Memory.
    if mode == "Parkinson" and chems["dopamine"] < 40.0:
        if random.random() < 0.30:
            motor_cells = []
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    if grid[r][c]["type"] == "Motor":
                        motor_cells.append((r, c))
            if motor_cells:
                tr_r, tr_c = random.choice(motor_cells)
                grid[tr_r][tr_c]["charge"] = 0.0
                chems["energy"] = max(0.0, chems["energy"] - 5.0)
                add_log(f"🤝 [Parkinson] Mức Dopamine quá thấp (<40%) kích hoạt cơn run giật (tremor) tại Motor [{tr_r+1},{tr_c+1}], làm rò rỉ điện tích mà không sinh ra IQ/Memory! (-5 Energy)")

    # Synaptic Pruning (Forget idle connections)
    if upgrades.get("pruning", 0) == 1:
        pruned_count = 0
        ach_discount = 1.0 - (chems["acetylcholine"] / 200.0)
        cost_inter = int(15 * ach_discount)

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                cell = grid[r][c]
                if cell["type"] == "Interneuron":
                    last_fired = cell.get("last_fired", -1)
                    if last_fired != -1 and (ticks - last_fired) > 15:
                        refund = int(cost_inter * 0.75)
                        st.session_state.stats["memory"] = min(1000.0, st.session_state.stats["memory"] + refund)
                        grid[r][c] = {
                            "type": "Empty",
                            "charge": 0.0,
                            "threshold": 0.5,
                            "fire_rate": 0.0,
                            "last_fired": -1,
                            "direction": "All",
                            "weight": 1.0
                        }
                        pruned_count += 1
        if pruned_count > 0:
            add_log(f"✂️ [Cắt tỉa] Đã tự động cắt tỉa {pruned_count} liên kết nơ-ron nhàn rỗi (>15 ticks) và hoàn phí +75% MB.")

    # 1. Metabolism and Fuel Check
    energy_generation = 4.0 + upgrades["brainstem"] * 2.0

    neuron_count = 0
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if grid[r][c]["type"] != "Empty":
                neuron_count += 1

    metabolic_cost = 1.0 + (neuron_count * 0.4)
    # CHRNA7 genetic mutation increases Energy metabolic consumption by 15%
    if "CHRNA7" in genes:
        metabolic_cost *= 1.15

    # Norepinephrine drains energy faster (Fight-or-Flight cost)
    norepi_val = chems.get("norepinephrine", 10.0)
    metabolic_cost += (norepi_val / 100.0) * 3.0

    # Somatosensory gating active buff reduces metabolic cost by 50%
    if buffs.get("somatosensory_gating", 0) > 0:
        metabolic_cost *= 0.5

    max_energy = 100.0
    # Astrocytic Glycogen Shunt & PGC-1alpha genetics boost max energy storage capacity
    if upgrades.get("glycogen_shunt", 0) == 1:
        max_energy = 150.0
    if "PGC-1alpha" in genes:
        max_energy += 40.0

    # Auto-release emergency glycogen pool
    if chems["energy"] < 15.0 and st.session_state.stats["glycogen_pool"] > 0:
        chems["energy"] = min(max_energy, chems["energy"] + 30.0)
        st.session_state.stats["glycogen_pool"] = max(0.0, st.session_state.stats["glycogen_pool"] - 30.0)
        add_log("🔋 [Glycogen Shunt] Năng lượng dưới 15%! Tự động xuất kho Glycogen khẩn cấp từ tế bào hình sao (+30 Energy).")

    chems["energy"] = max(0.0, min(max_energy, chems["energy"] + energy_generation - metabolic_cost))

    if chems["energy"] <= 0.0:
        add_log("⚠️ Cảnh báo: Bộ não cạn kiệt Glucose và Oxy! Không thể truyền tín hiệu.")
        chems["stress"] = max(0.0, min(100.0, chems["stress"] + 5.0))
        record_history(ticks, chems)
        return

    # Sleep Circus Loop Processing
    if st.session_state.stats.get("sleep_state", False):
        # Sensory inputs, signal propagation, and motor output are skipped/disabled
        # Rapid sleep recovery heals biochemistry and flushes waste
        chems["energy"] = min(max_energy, chems["energy"] + 15.0)
        chems["sanity"] = min(100.0, chems["sanity"] + 8.0)
        chems["stress"] = max(0.0, chems["stress"] - 12.0)

        # Reset cell charges to 0.0 during deep sleep
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                grid[r][c]["charge"] = 0.0

        # Stabilize dopamine and serotonin extremely quickly back towards 50%
        chems["dopamine"] += (50.0 - chems["dopamine"]) * 0.35
        chems["serotonin"] += (50.0 - chems["serotonin"]) * 0.35

        # Sleep flushes neuro-inflammation quickly
        chems["neuro_inflammation"] = max(5.0, chems["neuro_inflammation"] - 6.0)

        # Sleep flushes norepinephrine quickly
        chems["norepinephrine"] = max(5.0, chems.get("norepinephrine", 10.0) - 8.0)

        if cycle_time == 0: # Day shifted, wake up automatically!
            st.session_state.stats["sleep_state"] = False
            add_log("🌞 [Circadian] Mặt trời lên! Bộ não tự động tỉnh giấc, khôi phục hệ thống kích thích.")

        record_history(ticks, chems)
        return

    # 2. Sensory Stimuli Fire Check
    sensory_fires = 0
    visual_boost_active = False
    visual_spark = st.session_state.get("visual_spark", None)

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            cell = grid[r][c]
            if cell["type"] == "Sensory":
                # UPGRADE: DRD2 genetic mutation boosts dopamine sensory fire rate multipliers by +50%
                drd2_mult = 1.5 if "DRD2" in genes else 1.0
                boost = 1.0 + (chems["dopamine"] / 100.0) * drd2_mult
                # Thalamus boosts sensory cell charge speed by +20% per level
                thalamus_level = upgrades.get("thalamus", 0)
                boost *= (1.0 + thalamus_level * 0.2)

                # Fight-or-flight Norepinephrine boost: up to +40% fire rate boost
                norepi_boost = 1.0 + (chems.get("norepinephrine", 10.0) / 100.0) * 0.4
                boost *= norepi_boost

                # Occipital Lobe visual alignment boost
                if visual_spark and visual_spark["pos"] == (r, c):
                    if cell.get("direction", "All") == visual_spark["dir"]:
                        boost *= 2.0
                        visual_boost_active = True
                        add_log(f"👁️ [Thùy Chẩm] Khớp hướng thành công tại [{r+1},{c+1}]! Sensory nhận gia tốc 2.0x.")

                cell["charge"] += cell["fire_rate"] * boost
                if cell["charge"] >= cell["threshold"]:
                    sensory_fires += 1

    st.session_state.visual_boost_active = visual_boost_active

    # 3. Signal Propagation Model with Output Weights
    next_charges = [[grid[r][c]["charge"] for c in range(GRID_SIZE)] for r in range(GRID_SIZE)]
    signals_fired = 0
    fired_cells = set()

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            cell = grid[r][c]
            if cell["type"] != "Empty" and cell["charge"] >= cell["threshold"]:
                fired_cells.add((r, c))
                cell["last_fired"] = ticks

                # Check Parietal Lobe Spatial Gate
                gate = st.session_state.get("spatial_gate", None)
                if upgrades.get("parietal_lobe", 0) >= 1 and gate == (r, c):
                    chems["stress"] = max(0.0, chems["stress"] - 20.0)
                    st.session_state.active_buffs["somatosensory_gating"] = 3
                    add_log(f"🧭 [Thùy Đỉnh] Luồng điện tích khớp vị trí Gating [{r+1},{c+1}]! Giải tỏa stress lập tức và kích hoạt Somatosensory Gating (giảm 50% tiêu hao năng lượng).")
                carry_over = 0.05 * upgrades["plasticity"] if cell["type"] == "Interneuron" else 0.0
                next_charges[r][c] = carry_over

                dir_deltas = {
                    "Up": [(-1, 0)],
                    "Down": [(1, 0)],
                    "Left": [(0, -1)],
                    "Right": [(0, 1)],
                    "All": [(-1, 0), (1, 0), (0, -1), (0, 1)]
                }
                allowed_deltas = dir_deltas.get(cell.get("direction", "All"), dir_deltas["All"])

                neighbors = []
                for dr, dc in allowed_deltas:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                        if grid[nr][nc]["type"] != "Empty":
                            neighbors.append((nr, nc))

                if neighbors:
                    # SHANK3 (Synaptic Scaffolding) boosts Myelin transmission efficiency by +15%
                    shank3_bonus = 0.15 if "SHANK3" in genes else 0.0
                    signal_efficiency = 0.35 + (upgrades["myelin"] * 0.05) + shank3_bonus

                    # Epilepsy Pathology increases charge speed
                    if mode == "Epilepsy":
                        signal_efficiency *= 1.35

                    # Synaptic Output Weight scale
                    cell_weight = cell.get("weight", 1.0)
                    # Amyloid-Beta plaques reduce effective Synaptic weight by 50%
                    if cell.get("amyloid_plaque", False):
                        cell_weight *= 0.5
                    transfer_charge = (cell["charge"] * signal_efficiency * cell_weight) / len(neighbors)

                    for nr, nc in neighbors:
                        next_charges[nr][nc] = min(1.0, next_charges[nr][nc] + transfer_charge)
                        if upgrades["plasticity"] > 0 and grid[nr][nc]["charge"] > 0.3:
                            # BDNF Gene Mutation increases plasticity learning speed x1.5
                            learn_rate = 0.015 if "BDNF" in genes else 0.01
                            grid[nr][nc]["threshold"] = max(0.2, grid[nr][nc]["threshold"] - learn_rate)

                signals_fired += 1

    # Apply calculated charges
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            grid[r][c]["charge"] = next_charges[r][c]

    # 4. Motor Output Processing
    motor_yield_iq = 0.0
    motor_yield_mem = 0.0
    motor_fired_count = 0

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            cell = grid[r][c]
            if cell["type"] == "Motor" and (r, c) in fired_cells:
                motor_fired_count += 1
                iq_multiplier = 1.0 + (upgrades["cortex"] * 0.6)
                focus_bonus = 1.0 + (chems["acetylcholine"] / 100.0)

                # UPGRADE: COMT-Met gene mutation increases IQ gains from Motor fires by +30%
                comtmet_mult = 1.3 if "COMT-Met" in genes else 1.0
                cell_iq = 5.0 * iq_multiplier * focus_bonus * comtmet_mult

                # Occipital Lobe visual target stimulation bonus: +100% IQ
                if st.session_state.get("visual_boost_active", False):
                    cell_iq *= 2.0

                motor_yield_iq += cell_iq

                mem_multiplier = 1.0 + (upgrades["hippocampus"] * 0.4)
                cell_mem = 2.0 * mem_multiplier

                # Temporal Lobe auditory resonance: 3x Memory!
                freq_val = st.session_state.get("auditory_freq", 0)
                if upgrades.get("temporal_lobe", 0) >= 1 and 400 <= freq_val <= 500:
                    cell_mem *= 3.0

                motor_yield_mem += cell_mem

                # DRD4 Mutation doubles dopamine reward from Motor fires
                doping_multiplier = 2.0 if "DRD4" in genes else 1.0
                chems["dopamine"] = min(100.0, chems["dopamine"] + (8.0 * doping_multiplier))
                chems["acetylcholine"] = max(0.0, chems["acetylcholine"] - 4.0)

    # Apply motor accomplishments
    if motor_fired_count > 0:
        st.session_state.stats["iq"] += motor_yield_iq
        st.session_state.stats["memory"] = min(1000.0, st.session_state.stats["memory"] + motor_yield_mem)
        # Update session records
        st.session_state.stats["high_score_iq"] = max(st.session_state.stats["high_score_iq"], st.session_state.stats["iq"])
        st.session_state.stats["max_memory"] = max(st.session_state.stats["max_memory"], st.session_state.stats["memory"])

        add_log(f"🎯 Hành động Motor kích hoạt! Trùng hợp phát xung thành công (+{motor_yield_iq:.1f} IQ, +{motor_yield_mem:.1f} Trí nhớ)")

    st.session_state.audio_trigger = {
        "sensory": sensory_fires,
        "motor": motor_fired_count
    }

    # 5. Chemistry & Health Delta Calculations
    # Epilepsy doubles active propagation stress
    # GABRA1 Gene Mutation reduces Epilepsy stress by 35%
    epilepsy_stress_mult = 1.3 if "GABRA1" in genes else 2.0
    fire_stress_mult = epilepsy_stress_mult if mode == "Epilepsy" else 1.0

    # High GABA (>70) completely suppresses epilepsy hyper-excitability stress multiplication
    if chems.get("gaba", 30.0) > 70.0:
        fire_stress_mult = 1.0

    fire_stress = signals_fired * 1.5 * fire_stress_mult

    # High GABA reduces overall stress generation by 40%
    if chems.get("gaba", 30.0) > 70.0:
        fire_stress *= 0.6

    # Oxytocin active buff reduces stress generation by 50%
    if buffs.get("oxytocin", 0) > 0:
        fire_stress *= 0.5

    # MAOA genetic mutation increases stress generation on firing by +40%
    if "MAOA" in genes:
        fire_stress *= 1.4

    stress_clearance = 1.5 + (upgrades["cerebellum"] * 1.0)
    # SLC6A4 gene slightly dampens normal stress clearance by 30%
    if "SLC6A4" in genes:
        stress_clearance *= 0.7
    # UPGRADE: COMT-Met genetic mutation decays stress 30% slower
    if "COMT-Met" in genes:
        stress_clearance *= 0.7

    # Active SSRI Buff reduces stress generation by 50%
    if buffs.get("ssri", 0) > 0:
        fire_stress *= 0.5

    # Amygdala anatomy reduces active stress generation by -15% per level
    amygdala_level = upgrades.get("amygdala", 0)
    fire_stress *= max(0.1, 1.0 - amygdala_level * 0.15)

    chems["stress"] = max(0.0, min(100.0, chems["stress"] + fire_stress - stress_clearance))

    serotonin_dampening = chems["serotonin"] * 0.1
    effective_stress = max(0.0, chems["stress"] - serotonin_dampening)

    # DRD4 mutation causes low dopamine (<30.0) to double stress damage on sanity
    drd4_sanity_mult = 2.0 if ("DRD4" in genes and chems["dopamine"] < 30.0) else 1.0
    # UPGRADE: DRD2 genetic mutation causes high stress to inflict 1.5x sanity damage
    if "DRD2" in genes and chems["stress"] > 50.0:
        drd4_sanity_mult *= 1.5

    if effective_stress > 60.0:
        sanity_damage = (effective_stress - 60.0) * 0.35 * drd4_sanity_mult
        chems["sanity"] = max(0.0, min(100.0, chems["sanity"] - sanity_damage))
        if sanity_damage > 1.0:
            add_log(f"⚡ Căng thẳng cực độ gây tổn hại myelin và nơ-ron! (-{sanity_damage:.1f} Tỉnh táo)")
    else:
        healing = 0.5 + (chems["serotonin"] * 0.02)
        if buffs.get("oxytocin", 0) > 0:
            healing *= 2.0
        if mode == "Schizophrenia":
            healing *= 0.7
        chems["sanity"] = max(0.0, min(100.0, chems["sanity"] + healing))

    # UPGRADE: Neuro-Inflammation & Microglia Immune Delta Engine
    # Neuro-inflammation rises on signals fired and high stress
    inflammation_gain = (signals_fired * 0.4) + (effective_stress > 50.0 and (effective_stress - 50.0) * 0.2 or 0.0)
    # Natural immune clearance of 1.0% per tick
    chems["neuro_inflammation"] = max(0.0, min(100.0, chems["neuro_inflammation"] + inflammation_gain - 1.0))

    # Active Cytokine Storm Sanity decay
    if chems["neuro_inflammation"] > 80.0:
        cyto_decay = (chems["neuro_inflammation"] - 80.0) * 0.4
        chems["sanity"] = max(0.0, chems["sanity"] - cyto_decay)

    # UPGRADE: Norepinephrine Fight-or-Flight & Panic Attack Delta Engine
    # Norepinephrine rises with signals fired and high stress
    norepi_gain = (signals_fired * 0.5) + (chems["stress"] * 0.1)
    # Natural clearance of 1.5% per tick
    chems["norepinephrine"] = max(0.0, min(100.0, chems.get("norepinephrine", 10.0) + norepi_gain - 1.5))

    # Panic Attack Trigger Check (threshold is 90% normally, or 100% if ADRA2A gene is active)
    panic_threshold = 100.0 if "ADRA2A" in genes else 90.0
    if chems.get("norepinephrine", 10.0) >= panic_threshold:
        damage = 9.0 if "ADRA2A" in genes else 15.0
        chems["sanity"] = max(0.0, chems["sanity"] - damage)

        # Paralyze 3 random cells: find non-empty neurons and set their charge to 0
        non_empty = []
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if grid[r][c]["type"] != "Empty":
                    non_empty.append((r, c))
        if non_empty:
            paralyzed_cells = random.sample(non_empty, min(3, len(non_empty)))
            for pr, pc in paralyzed_cells:
                grid[pr][pc]["charge"] = 0.0

        add_log(f"🚨 [HOẢNG LOẠN] Norepinephrine ({chems['norepinephrine']:.1f}%) vượt ngưỡng {panic_threshold}%! Gây mất -{damage:.1f} Sanity và đóng băng 3 nơ-ron.")
        # Post-panic crash
        chems["norepinephrine"] = 50.0

    # COMT Gene Mutation decays Dopamine 40% slower and Stress 20% slower
    # MAOA Gene Mutation decays Dopamine and Serotonin 30% slower
    da_decay_rate = 0.048 if "COMT" in genes else 0.08
    if "MAOA" in genes:
        da_decay_rate *= 0.7

    se_decay_rate = 0.056 if "MAOA" in genes else 0.08
    stress_decay_factor = 0.064 if "COMT" in genes else 0.08

    chems["dopamine"] += (50.0 - chems["dopamine"]) * da_decay_rate
    chems["serotonin"] += (50.0 - chems["serotonin"]) * se_decay_rate

    # CHRNA7 boosts Acetylcholine generation / baseline stabilization by +25%
    ach_decay_rate = 0.08
    # UPGRADE: ADHD Mode accelerates Acetylcholine decay rate by 50%
    if mode == "ADHD":
        ach_decay_rate *= 1.5

    ach_delta = (50.0 - chems["acetylcholine"]) * ach_decay_rate
    if "CHRNA7" in genes:
        ach_delta *= 1.25
    chems["acetylcholine"] += ach_delta

    # GABA natural decay and stress response delta
    gaba_decay = 1.0
    gaba_gain = 0.0
    if chems["stress"] > 40.0:
        gaba_gain = (chems["stress"] - 40.0) * 0.05
    chems["gaba"] = max(0.0, min(100.0, chems.get("gaba", 30.0) + gaba_gain - gaba_decay))

    # Apply Temporal Lobe sound effects
    freq_sound = st.session_state.get("auditory_freq", None)
    if upgrades.get("temporal_lobe", 0) >= 1 and freq_sound:
        if freq_sound > 600:
            chems["dopamine"] = min(100.0, chems["dopamine"] + 2.0)
            chems["gaba"] = min(100.0, chems.get("gaba", 30.0) + 2.0)
        else:
            chems["acetylcholine"] = min(100.0, chems["acetylcholine"] + 2.0)

    # Burnout Check: Sanity is 0
    if chems["sanity"] <= 0.0:
        st.session_state.stats["burnout_count"] += 1
        st.session_state.stats["burnout_streak"] = 0 # reset streak
        st.session_state.playing = False
        chems["sanity"] = 25.0
        chems["stress"] = 10.0
        chems["energy"] = 50.0
        chems["dopamine"] = 20.0
        chems["serotonin"] = 30.0
        chems["neuro_inflammation"] = 30.0 # moderate post-burnout inflammation
        chems["norepinephrine"] = 20.0 # moderate post-burnout norepinephrine

        # Reset ongoing buffs on burnout
        st.session_state.active_buffs = {"doping": 0, "ssri": 0, "focus": 0, "tyrosine": 0, "tryptophan": 0, "choline": 0}

        degraded = 0
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if grid[r][c]["type"] == "Interneuron" and random.random() < 0.4:
                    grid[r][c] = {"type": "Empty", "charge": 0.0, "threshold": 0.5, "fire_rate": 0.0, "last_fired": -1, "direction": "All", "weight": 1.0}
                    degraded += 1

        add_log(f"💥 ĐỘT QUY/SUY NHƯỢC KINH NIÊN! Não bộ quá tải và tự thiết lập lại. {degraded} nơ-ron liên kết bị phá hủy.")

    # Update Evolution Stages
    old_stage = st.session_state.stats["evolution_stage"]
    new_stage = get_evolution_stage(st.session_state.stats["iq"])
    if old_stage != new_stage:
        st.session_state.stats["evolution_stage"] = new_stage
        add_log(f"🎉 TIẾN HÓA: Não bộ đã bước vào kỷ nguyên '{new_stage}'!")

    # Check challenges
    check_mission_statuses()

    # Trigger periodic cognitive events (12% chance)
    if st.session_state.current_event is None and random.random() < 0.12 and ticks % 4 == 0:
        trigger_random_event()

    # UPGRADE: Prefrontal Cortex PFC AI Decision Auto-chooser
    if upgrades.get("pfc", 0) == 1 and st.session_state.current_event is not None:
        ev = st.session_state.current_event
        best_idx = 0
        if "Espresso" in ev["title"]:
            best_idx = 1 # tea
        elif "Kỳ Thi" in ev["title"]:
            best_idx = 1 # normal study
        elif "TikTok" in ev["title"]:
            best_idx = 1 # book reading
        elif "Thiền" in ev["title"]:
            best_idx = 0 # full vipassana
        else:
            best_idx = min(1, len(ev["choices"]) - 1)

        choice = ev["choices"][best_idx]
        choice["apply"]()
        add_log(f"🧠 [PFC Tự Quyết] Thùy trán trước đã tự động quyết định tối ưu: '{choice['label']}'")
        st.session_state.current_event = None

    # Clear visual boost at the end of the tick
    st.session_state.visual_boost_active = False

    # Record history for plot
    record_history(ticks, chems)

def record_history(ticks, chems):
    hist = st.session_state.history_data
    hist["tick"].append(ticks)
    hist["sanity"].append(chems["sanity"])
    hist["energy"].append(chems["energy"])
    hist["dopamine"].append(chems["dopamine"])
    hist["stress"].append(chems["stress"])
    hist["norepinephrine"].append(chems.get("norepinephrine", 10.0))
    hist["gaba"].append(chems.get("gaba", 30.0))

    if len(hist["tick"]) > 40:
        for key in hist:
            hist[key] = hist[key][-40:]

# Start the Telegram background bot thread if not already running (only when run as the main script)
if __name__ == "__main__":
    if "bot_thread" not in st.session_state:
        st.session_state.bot_thread = True
        thread = threading.Thread(target=run_bot, daemon=True)
        thread.start()

# ----------------- STREAMLIT INTERFACE RENDERING -----------------

st.set_page_config(
    page_title="🧠 Brain Simulator & VBot1 System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom css for dark science/cyberpunk theme and grids
st.markdown("""
<style>
    .reportview-container {
        background-color: #0d0f12;
    }
    .metric-box {
        background-color: #171b21;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #232d38;
        text-align: center;
    }
    .grid-cell {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 60px;
        border-radius: 6px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .cell-Empty { background-color: #161b22; border: 1px solid #30363d; color: #8b949e; }
    .cell-Sensory { background-color: #3e2e00; border: 2px solid #f1e05a; color: #f1e05a; }
    .cell-Interneuron { background-color: #0f2c59; border: 2px solid #58a6ff; color: #58a6ff; }
    .cell-Motor { background-color: #1b3a1b; border: 2px solid #3fb950; color: #3fb950; }
    .cell-firing { box-shadow: 0 0 15px #ff7b72; border: 2px solid #ff7b72 !important; background-color: #491d1d !important; }
    .buff-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.85em;
        font-weight: bold;
        margin-right: 5px;
    }
    .badge-doping { background-color: #ffaa00; color: black; }
    .badge-ssri { background-color: #00aaff; color: white; }
    .badge-focus { background-color: #8800ff; color: white; }
    .badge-tyrosine { background-color: #e5c158; color: black; }
    .badge-tryptophan { background-color: #55cbf9; color: black; }
    .badge-choline { background-color: #b77ef4; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Siêu Hệ Thống VBot1 & Game Mô Phỏng Não Bộ")
st.write("Dự án tích hợp: Game mô phỏng tiến hóa nơ-ron sinh học kết hợp Trợ lý AI Telegram Llama 3 & Gemini 1.5. **(Version 4.0.0 - The Ultimate Neurological Integration Update)**")

with st.expander("🆕 [CHANGELOG] Nhật Ký Cập Nhật Phiên Bản 4.0.0 - Hệ Nội Tiết & Thử Thách Tâm Thần", expanded=False):
    st.markdown("""
    **🚀 Phiên bản 4.0.0 (Bản nâng cấp tối thượng hệ nội tiết và bệnh lý học):**
    *   **Tuyến Yên (Pituitary Gland) & Giải phóng Oxytocin:** Mở rộng cấu trúc giải phẫu Tuyến Yên điều hòa hormone. Tự động kích hoạt **Oxytocin Surge** mỗi 20 ticks trong 5 ticks liên tục giúp triệt tiêu 50% Stress phát sinh và nhân đôi tốc độ hồi phục Tỉnh táo (Sanity).
    *   **Bệnh lý học Tâm Thần Phân Liệt (Schizophrenia Mode):** Thử thách thứ 6 mô phỏng ảo giác thần kinh. Cứ mỗi 8 ticks, ảo thanh đột ngột kích phát điện thế của một nơ-ron ngẫu nhiên lên mức tối đa gây hỗn loạn liên kết truyền dẫn, đồng thời giảm 30% tốc độ tự chữa lành Sanity của não bộ.

    **🚀 Phiên bản 3.2.0:**
    *   **Thùy Đỉnh (Parietal Lobe) & Định vị không gian:** Nhận điểm định vị Gating ngẫu nhiên mỗi 18 ticks. Truyền điện thế qua ô Gating này để kích hoạt **Somatosensory Gating** dập tắt ngay lập tức 20% Stress và cắt giảm 50% tiêu hao Năng lượng (Fuel) trong 3 ticks kế tiếp.
    *   **Nảy mầm liên kết Synaptic Sprouting:** Thêm khả năng chủ động mới "🌱 Sprouting" (cooldown 30s, chi phí 40 MB). Sao chép ngẫu nhiên cấu hình một Interneuron sang một ô trống lân cận để xây dựng liên kết free.

    **🚀 Phiên bản 3.1.0:**
    *   **Thùy Thái Dương (Temporal Lobe) & Thính giác:** Nhận kích thích âm thanh (Auditory Stimulus) ngẫu nhiên mỗi 15 ticks. Tần số cao (>600 Hz) kích thích tăng Dopamine/GABA, tần số thấp (<=600 Hz) tăng Acetylcholine.
    *   **Cộng hưởng Thính giác - Vận động:** Tần số cộng hưởng (400 Hz đến 500 Hz) sẽ kích hoạt trạng thái cộng hưởng thính giác-vận động, nhân ba (3.0x) sản lượng Trí nhớ (Memory MB) từ các hành động Motor phát xung thành công.
    *   **Bộ củng cố Trí nhớ dài hạn Hebbian LTP (LTP Consolidator):** Nâng cấp giải phẫu tự động củng cố 30% bộ nhớ sang IQ vĩnh viễn Hebbian LTP sau mỗi 12 ticks hoạt động.

    **🚀 Phiên bản 3.0.0:**
    *   **Hệ thống dẫn truyền ức chế GABA:** Bổ sung hóa chất mới GABA giúp giải tỏa stress và dập tắt hoàn toàn trạng thái quá kích của bệnh lý Động kinh. Bổ sung tiền chất **Glutamate Precursor** vào chế độ ăn uống.
    *   **Thùy Chẩm (Occipital Lobe) & Visual Spark:** Cơ chế phản hồi kích thích thị giác ngẫu nhiên mỗi 10 ticks. Định hướng trục nơ-ron cảm giác khớp với hướng kích thích thị giác sẽ khuyếch đại tốc độ sạc nơ-ron x2 và nhân đôi (+100%) điểm IQ từ các hành động Motor.
    *   **Thư viện Bản Lưu Cục Bộ (Save Slots Library):** Tích hợp 3 khe lưu trữ sơ đồ mạch thần kinh tức thì để cất trữ và khôi phục nhanh chóng sơ đồ nơ-ron trong phiên chơi.

    *Cám ơn bạn đã luôn đồng hành cùng Trình tiến hóa não bộ sinh học!*
    """)

tab1, tab2 = st.tabs(["🧠 Game Mô Phỏng Não Bộ", "🤖 Trợ Lý AI VBot1 (Llama & Gemini)"])

# Initialize game session state
init_game_state()

# ----------------- TAB 1: BRAIN GAME -----------------
with tab1:
    st.subheader("Trình Mô Phỏng Mạng Lưới Nơ-ron và Tiến Hóa Hóa Học")

    # Personal records display
    st.markdown("##### 🏆 Bảng Kỷ Lục Nhận Thức Cá Nhân (Personal Records)")
    record_cols = st.columns(5)
    with record_cols[0]:
        st.write(f"📈 **IQ Cao Nhất:** `{st.session_state.stats['high_score_iq']:.1f} pts`")
    with record_cols[1]:
        st.write(f"💾 **Trí nhớ Lớn Nhất:** `{st.session_state.stats['max_memory']:.1f} MB`")
    with record_cols[2]:
        st.write(f"🔥 **Chuỗi Sống Khỏe:** `{st.session_state.stats['burnout_streak']} ticks`")
    with record_cols[3]:
        st.write(f"👑 **Kỷ Lục Chuỗi:** `{st.session_state.stats['max_streak']} ticks`")
    with record_cols[4]:
        cycle_emoji = "🌞 Ngày (Day)" if st.session_state.stats["circadian_cycle"] == "Day" else "🌙 Đêm (Night)"
        st.write(f"⏰ **Chu kỳ sinh học:** `{cycle_emoji}`")

    st.markdown("---")

    # Game pathology modes selector (Normal, Alzheimer, Epilepsy, Parkinson, ADHD)
    st.markdown("##### ⚙️ Lựa chọn Chế Độ Thử Thách Não Bộ")
    modes_list = ["Normal", "Alzheimer", "Epilepsy", "Parkinson", "ADHD", "Schizophrenia"]
    modes_names = {
        "Normal": "🟢 Bình Thường (Sức khỏe ổn định)",
        "Alzheimer": "👵 Thử Thách Alzheimer (Thoái hóa nơ-ron, chai lỳ điện thế)",
        "Epilepsy": "⚡ Thử Thách Động Kinh (Gia tăng xung điện cực độ, nhân đôi stress)",
        "Parkinson": "🤝 Thử Thách Parkinson (Run giật nơ-ron vận động khi thiếu hụt Dopamine)",
        "ADHD": "🧠 Thử Thách ADHD (Dao động Dopamine dữ dội, tăng tốc phân rã Acetylcholine)",
        "Schizophrenia": "📢 Thử Thách Tâm Thần Phân Liệt (Ảo thanh kích phát điện thế bất ngờ, giảm 30% hồi tỉnh táo)"
    }
    selected_mode = st.selectbox(
        "Cấu hình bệnh lý học vỏ não:",
        modes_list,
        index=modes_list.index(st.session_state.game_mode),
        format_func=lambda x: modes_names[x]
    )
    if not isinstance(selected_mode, str):
        selected_mode = st.session_state.game_mode

    if selected_mode != st.session_state.game_mode:
        st.session_state.game_mode = selected_mode
        add_log(f"⚠️ CẤU HÌNH: Chuyển cấu hình vỏ não sang chế độ: {modes_names[selected_mode]}")
        st.rerun()

    # Continuous Active Buff badges indicators
    st.markdown("##### 🧪 Trạng thái hoạt hóa hóa học (Active Neuromodulator Buffs)")
    active_buffs = st.session_state.get("active_buffs", {"doping": 0, "ssri": 0, "focus": 0, "tyrosine": 0, "tryptophan": 0, "choline": 0, "glutamate": 0, "oxytocin": 0})
    buffs_html = []
    if active_buffs.get("doping", 0) > 0:
        buffs_html.append(f"<span class='buff-badge badge-doping'>⚡ Hyper-Dopamine ({active_buffs['doping']} ticks)</span>")
    if active_buffs.get("ssri", 0) > 0:
        buffs_html.append(f"<span class='buff-badge badge-ssri'>💊 SSRI Serenity ({active_buffs['ssri']} ticks)</span>")
    if active_buffs.get("focus", 0) > 0:
        buffs_html.append(f"<span class='buff-badge badge-focus'>🧠 Deep Focus ({active_buffs['focus']} ticks)</span>")
    if active_buffs.get("tyrosine", 0) > 0:
        buffs_html.append(f"<span class='buff-badge badge-tyrosine'>🥩 L-Tyrosine Synthesis ({active_buffs['tyrosine']} ticks)</span>")
    if active_buffs.get("tryptophan", 0) > 0:
        buffs_html.append(f"<span class='buff-badge badge-tryptophan'>🍌 L-Tryptophan Synthesis ({active_buffs['tryptophan']} ticks)</span>")
    if active_buffs.get("choline", 0) > 0:
        buffs_html.append(f"<span class='buff-badge badge-choline'>🥚 Choline Synthesis ({active_buffs['choline']} ticks)</span>")
    if active_buffs.get("glutamate", 0) > 0:
        buffs_html.append(f"<span class='buff-badge badge-choline'>🥦 Glutamate Synthesis ({active_buffs['glutamate']} ticks)</span>")
    if active_buffs.get("oxytocin", 0) > 0:
        buffs_html.append(f"<span class='buff-badge badge-ssri'>💕 Oxytocin Surge ({active_buffs['oxytocin']} ticks)</span>")

    if buffs_html:
        st.markdown(" ".join(buffs_html), unsafe_allow_html=True)
    else:
        st.info("Không có hoạt chất bổ trợ nào đang hoạt động liên tục.")

    # Row 1: Metrics display
    cols = st.columns(6)
    with cols[0]:
        st.metric("Trạng thái Tiến Hóa", st.session_state.stats["evolution_stage"])
    with cols[1]:
        st.metric("Chỉ số IQ (Nhận thức)", f"{st.session_state.stats['iq']:.1f} pts")
    with cols[2]:
        st.metric("Bộ nhớ lưu trữ", f"{st.session_state.stats['memory']:.1f} MB")
    with cols[3]:
        st.metric("Tỉnh táo (Sanity)", f"{st.session_state.chemicals['sanity']:.1f}%")
    with cols[4]:
        # Displays max capacity according to upgrades & genes
        max_cap = 100.0
        if st.session_state.upgrades.get("glycogen_shunt", 0) == 1:
            max_cap = 150.0
        if "PGC-1alpha" in st.session_state.active_genes:
            max_cap += 40.0
        st.metric("Mức năng lượng (Fuel)", f"{st.session_state.chemicals['energy']:.1f}% / {max_cap:.0f}%")
    with cols[5]:
        st.metric("Kho Glycogen tế bào hình sao", f"{st.session_state.stats.get('glycogen_pool', 0.0):.1f} units")

    # Progress bars for detailed chemistry (added Melatonin, Inflammation, Norepinephrine, GABA)
    chem_cols = st.columns(8)
    with chem_cols[0]:
        val = st.session_state.chemicals["dopamine"]
        st.progress(val / 100.0, text=f"Dopamine (Động lực): {val:.1f}%")
    with chem_cols[1]:
        val = st.session_state.chemicals["serotonin"]
        st.progress(val / 100.0, text=f"Serotonin (Ổn định): {val:.1f}%")
    with chem_cols[2]:
        val = st.session_state.chemicals["acetylcholine"]
        st.progress(val / 100.0, text=f"Acetylcholine (Tập trung): {val:.1f}%")
    with chem_cols[3]:
        val = st.session_state.chemicals["stress"]
        st.progress(val / 100.0, text=f"Căng thẳng (Stress): {val:.1f}%")
    with chem_cols[4]:
        val = st.session_state.chemicals.get("melatonin", 10.0)
        st.progress(val / 100.0, text=f"Melatonin (Gây ngủ): {val:.1f}%")
    with chem_cols[5]:
        val = st.session_state.chemicals.get("neuro_inflammation", 10.0)
        micro_state = "🔥 Reactive (Bão)" if val > 80.0 else ("⚠️ Reactive" if val >= 50.0 else "Normal")
        st.progress(val / 100.0, text=f"Viêm thần kinh: {val:.1f}% ({micro_state})")
    with chem_cols[6]:
        val = st.session_state.chemicals.get("norepinephrine", 10.0)
        st.progress(val / 100.0, text=f"Norepinephrine (Fight-or-Flight): {val:.1f}%")
    with chem_cols[7]:
        val = st.session_state.chemicals.get("gaba", 30.0)
        st.progress(val / 100.0, text=f"GABA (Ức chế dịu não): {val:.1f}%")

    # Live EEG Brainwave Telemetry monitoring
    st.markdown("##### 📊 Sóng não lâm sàng (EEG Brainwave Telemetry)")
    eeg_cols = st.columns(5)

    active_count = sum(1 for r in range(GRID_SIZE) for c in range(GRID_SIZE) if st.session_state.neuron_grid[r][c]["type"] != "Empty")

    gamma = min(100.0, max(5.0, (st.session_state.chemicals["acetylcholine"] * 0.6) + (active_count * 2.0)))
    beta = min(100.0, max(5.0, (st.session_state.chemicals["stress"] * 0.8) + (st.session_state.chemicals["dopamine"] * 0.3)))
    alpha = min(100.0, max(5.0, (st.session_state.chemicals["serotonin"] * 0.8) - (st.session_state.chemicals["stress"] * 0.3)))
    theta = min(100.0, max(5.0, (st.session_state.upgrades["hippocampus"] * 8.0) + (50.0 - st.session_state.chemicals["dopamine"] * 0.2)))
    delta = min(100.0, max(5.0, (100.0 - st.session_state.chemicals["sanity"]) * 0.7 + (100.0 - st.session_state.chemicals["energy"]) * 0.4))

    with eeg_cols[0]:
        st.progress(gamma / 100.0, text=f"Sóng Gamma (Nhận thức sâu): {gamma:.1f} Hz")
    with eeg_cols[1]:
        st.progress(beta / 100.0, text=f"Sóng Beta (Cảnh giác/Lo âu): {beta:.1f} Hz")
    with eeg_cols[2]:
        st.progress(alpha / 100.0, text=f"Sóng Alpha (Thư giãn): {alpha:.1f} Hz")
    with eeg_cols[3]:
        st.progress(theta / 100.0, text=f"Sóng Theta (Ghi nhớ/Thiền): {theta:.1f} Hz")
    with eeg_cols[4]:
        st.progress(delta / 100.0, text=f"Sóng Delta (Hồi phục/Ngủ): {delta:.1f} Hz")

    # Hormone active abilities layout
    st.markdown("##### 🧪 Trung tâm nội tiết tố & Liệu pháp Lâm sàng (Active Abilities)")
    hormone_cols = st.columns(8)
    cooldowns = st.session_state.cooldowns

    with hormone_cols[0]:
        doping_disabled = cooldowns["doping"] > 0
        btn_label_doping = f"⚡ Doping Dopamine ({cooldowns['doping']}s)" if doping_disabled else "⚡ Doping Dopamine"
        if st.button(btn_label_doping, disabled=doping_disabled, use_container_width=True, help="Tự động sạc đầy tất cả Sensory cells, +30 Dopamine, +25 Stress. Cooldown 15s. Kích hoạt buff liên tục 8 ticks."):
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    if st.session_state.neuron_grid[r][c]["type"] == "Sensory":
                        st.session_state.neuron_grid[r][c]["charge"] = 1.0
            st.session_state.chemicals["dopamine"] = min(100.0, st.session_state.chemicals["dopamine"] + 30.0)
            st.session_state.chemicals["stress"] = min(100.0, st.session_state.chemicals["stress"] + 25.0)
            st.session_state.active_buffs["doping"] = 8
            cooldowns["doping"] = 15
            add_log("⚡ HORMONE: Kích hoạt Doping Dopamine! Đồng loạt Sensory cells bùng nổ xung điện. Buff dopamine liên tục kích hoạt.")
            st.rerun()

    with hormone_cols[1]:
        ssri_disabled = cooldowns["ssri"] > 0
        btn_label_ssri = f"💊 Serotonin (SSRI) ({cooldowns['ssri']}s)" if ssri_disabled else "💊 Serotonin (SSRI)"
        if st.button(btn_label_ssri, disabled=ssri_disabled, use_container_width=True, help="Hạ 50% Stress, hồi phục 30 Tỉnh táo lập tức. Cooldown 25s. Kích hoạt buff liên tục 12 ticks giảm 50% sinh stress."):
            st.session_state.chemicals["stress"] = max(0.0, st.session_state.chemicals["stress"] - 50.0)
            st.session_state.chemicals["sanity"] = min(100.0, st.session_state.chemicals["sanity"] + 30.0)

            slc6a4_mult_val = 1.5 if "SLC6A4" in st.session_state.get("active_genes", []) else 1.0
            ssri_dur = int(12 * slc6a4_mult_val)
            st.session_state.active_buffs["ssri"] = ssri_dur
            cooldowns["ssri"] = 25
            add_log(f"💊 HORMONE: Kích hoạt liệu pháp Serotonin! Xoa dịu vỏ não, triệt tiêu căng thẳng. Serotonin buff liên tục {ssri_dur} ticks kích hoạt.")
            st.rerun()

    with hormone_cols[2]:
        focus_disabled = cooldowns["focus"] > 0
        btn_label_focus = f"🧠 Tập trung ({cooldowns['focus']}s)" if focus_disabled else "🧠 Tập trung"
        if st.button(btn_label_focus, disabled=focus_disabled, use_container_width=True, help="Tăng Acetylcholine (+50) và nạp thêm +50 IQ. Cooldown 20s. Kích hoạt buff liên tục 10 ticks."):
            st.session_state.chemicals["acetylcholine"] = min(100.0, st.session_state.chemicals["acetylcholine"] + 50.0)
            st.session_state.stats["iq"] += 50.0
            st.session_state.active_buffs["focus"] = 10
            cooldowns["focus"] = 20
            add_log("🧠 HORMONE: Kích hoạt Tập trung cao độ! Khóa chặt Acetylcholine, nâng cao nhận thức (+50 IQ). Focus buff liên tục kích hoạt.")
            st.rerun()

    # Transcranial Magnetic Stimulation (rTMS) Clinical Therapy Active Ability
    with hormone_cols[3]:
        rtms_disabled = cooldowns["rtms"] > 0
        btn_label_rtms = f"🏥 Liệu pháp rTMS ({cooldowns['rtms']}s)" if rtms_disabled else "🏥 Liệu pháp rTMS"
        if st.button(btn_label_rtms, disabled=rtms_disabled, use_container_width=True, help="Kích thích Từ trường xuyên sọ: Reset ngay tất cả ngưỡng kích hoạt nơ-ron về mặc định (chữa trị Alzheimer), hồi phục 40% Sanity. Cooldown 35s"):
            # Reset thresholds of all cells
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    t_name = st.session_state.neuron_grid[r][c]["type"]
                    if t_name != "Empty":
                        st.session_state.neuron_grid[r][c]["threshold"] = 0.4 if t_name == "Sensory" else (0.6 if t_name == "Motor" else 0.5)
            st.session_state.chemicals["sanity"] = min(100.0, st.session_state.chemicals["sanity"] + 40.0)
            cooldowns["rtms"] = 35
            add_log("🏥 LÂM SÀNG: Kích hoạt Từ trường xuyên sọ rTMS! Ngưỡng điện tích nơ-ron được khôi phục về trạng thái khỏe mạnh ban đầu.")
            st.rerun()

    # Sleep State recovery action
    with hormone_cols[4]:
        is_sleeping = st.session_state.stats.get("sleep_state", False)
        btn_label_sleep = "🌞 Thức Dậy" if is_sleeping else "😴 Đi Ngủ (Sleep)"
        if st.button(btn_label_sleep, use_container_width=True, help="Cho bộ não ngủ sâu phục hồi: Sensory và Motor tạm ngừng, sạc nhanh Năng lượng (+15), Tỉnh táo (+8) và thải Stress (-12) cực tốc."):
            st.session_state.stats["sleep_state"] = not is_sleeping
            if not is_sleeping:
                add_log("😴 [Circadian] Bộ não bắt đầu chìm vào giấc ngủ phục hồi sâu. Các sóng liên kết tạm dừng hoạt động.")
            else:
                add_log("🌞 [Circadian] Bộ não thức dậy sớm theo yêu cầu của bạn!")
            st.rerun()

    # UPGRADE: Clinical Anti-Inflammatory Cortisol Wash Active Ability
    with hormone_cols[5]:
        cortisol_disabled = cooldowns.get("cortisol", 0) > 0 or st.session_state.stats["memory"] < 30.0
        btn_label_cortisol = f"🧪 Cortisol Wash ({cooldowns.get('cortisol', 0)}s)" if cooldowns.get("cortisol", 0) > 0 else "🧪 Cortisol Wash"
        if st.button(btn_label_cortisol, disabled=cortisol_disabled, use_container_width=True, help="Liệu pháp rửa giải Kháng Viêm Cortisol: Phí tiêu thụ 30 MB bộ nhớ. Đặt mức Viêm thần kinh về 10%, phục hồi +15 Sanity lập tức. Cooldown 20s."):
            st.session_state.stats["memory"] -= 30.0
            st.session_state.chemicals["neuro_inflammation"] = 10.0
            st.session_state.chemicals["sanity"] = min(100.0, st.session_state.chemicals["sanity"] + 15.0)
            cooldowns["cortisol"] = 20
            add_log("🧪 LÂM SÀNG: Kích hoạt liệu pháp Cortisol Wash! Rửa giải toàn bộ cytokine kháng viêm, dập tắt việm nơ-ron cấp tính.")
            st.rerun()

    # UPGRADE: Clinical Beta-Blocker Propranolol Active Ability
    with hormone_cols[6]:
        propranolol_disabled = cooldowns.get("propranolol", 0) > 0
        btn_label_propranolol = f"🩺 Propranolol ({cooldowns.get('propranolol', 0)}s)" if cooldowns.get("propranolol", 0) > 0 else "🩺 Propranolol"
        if st.button(btn_label_propranolol, disabled=propranolol_disabled, use_container_width=True, help="Liệu pháp chặn Beta Propranolol: Reset lập tức nồng độ Norepinephrine hoảng loạn về mức 10.0% và làm dịu nhịp sinh học vỏ não. Cooldown 20s."):
            st.session_state.chemicals["norepinephrine"] = 10.0
            cooldowns["propranolol"] = 20
            add_log("🩺 LÂM SÀNG: Sử dụng Propranolol Beta-Blocker! Chặn đứng Norepinephrine kích thích, dập tắt hoàn toàn các triệu chứng hoảng loạn cấp tính.")
            st.rerun()

    # UPGRADE: Clinical Synaptic Sprouting Active Ability (lateral collateral growth)
    with hormone_cols[7]:
        sprouting_disabled = cooldowns.get("sprouting", 0) > 0 or st.session_state.stats["memory"] < 40.0
        btn_label_sprouting = f"🌱 Sprouting ({cooldowns.get('sprouting', 0)}s)" if cooldowns.get("sprouting", 0) > 0 else "🌱 Sprouting"
        if st.button(btn_label_sprouting, disabled=sprouting_disabled, use_container_width=True, help="Kích thích nảy mầm liên kết (Synaptic Sprouting): Chi phí 40 MB Bộ nhớ. Sao chép cấu hình của một nơ-ron liên kết (Interneuron) ngẫu nhiên sang một ô trống lân cận nó để nhân bản miễn phí. Cooldown 30s."):
            grid = st.session_state.neuron_grid
            # Find Interneurons with adjacent Empty cells
            interneurons = []
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    if grid[r][c]["type"] == "Interneuron":
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                                if grid[nr][nc]["type"] == "Empty":
                                    interneurons.append((r, c, nr, nc))
            if interneurons:
                parent_r, parent_c, child_r, child_c = random.choice(interneurons)
                parent_cell = grid[parent_r][parent_c]

                grid[child_r][child_c] = {
                    "type": "Interneuron",
                    "charge": 0.0,
                    "threshold": parent_cell["threshold"],
                    "fire_rate": parent_cell.get("fire_rate", 0.0),
                    "last_fired": -1,
                    "direction": parent_cell.get("direction", "All"),
                    "weight": parent_cell.get("weight", 1.0),
                    "amyloid_plaque": False
                }
                st.session_state.stats["memory"] -= 40.0
                cooldowns["sprouting"] = 30
                add_log(f"🌱 LÂM SÀNG: Kích hoạt mọc mầm Synaptic Sprouting! Sao chép liên kết từ [{parent_r+1},{parent_c+1}] sang ô trống [{child_r+1},{child_c+1}] (-40 MB Memory).")
                st.rerun()
            else:
                st.warning("Không tìm thấy Interneuron nào có ô trống lân cận để mọc mầm!")

    # Neurotransmitter Synthesis Precursors & Diet System Layout
    st.markdown("##### 🧬 Dinh Dưỡng Học & Tiền Chất Thần Kinh (Precursor Dietary Intake)")
    st.caption("Tổng hợp trực tiếp các chất dẫn truyền thông qua bồi bổ dinh dưỡng. Phí tiêu thụ: 15 MB Bộ nhớ.")

    diet_cols = st.columns(4)
    with diet_cols[0]:
        tyrosine_disabled = st.session_state.stats["memory"] < 15.0
        if st.button("🥩 Bổ sung L-Tyrosine (-15 MB)", disabled=tyrosine_disabled, use_container_width=True, help="Tiền chất Dopamine: Tăng +3.0 Dopamine/tick liên tục trong 15 ticks."):
            st.session_state.stats["memory"] -= 15.0
            st.session_state.active_buffs["tyrosine"] = 15
            add_log("🥩 DINH DƯỠNG: Bổ sung L-Tyrosine! Thúc đẩy tổng hợp dopamine nội sinh (+3.0 DA/tick trong 15s).")
            st.rerun()

    with diet_cols[1]:
        tryptophan_disabled = st.session_state.stats["memory"] < 15.0
        if st.button("🍌 Bổ sung L-Tryptophan (-15 MB)", disabled=tryptophan_disabled, use_container_width=True, help="Tiền chất Serotonin: Tăng +2.0 Serotonin/tick liên tục trong 15 ticks."):
            st.session_state.stats["memory"] -= 15.0
            st.session_state.active_buffs["tryptophan"] = 15
            add_log("🍌 DINH DƯỠNG: Bổ sung L-Tryptophan! Thúc đẩy tổng hợp serotonin giúp ổn định tinh thần (+2.0 SE/tick trong 15s).")
            st.rerun()

    with diet_cols[2]:
        choline_disabled = st.session_state.stats["memory"] < 15.0
        if st.button("🥚 Bổ sung Choline (-15 MB)", disabled=choline_disabled, use_container_width=True, help="Tiền chất Acetylcholine: Tăng +2.5 Acetylcholine/tick liên tục trong 15 ticks."):
            st.session_state.stats["memory"] -= 15.0
            st.session_state.active_buffs["choline"] = 15
            add_log("🥚 DINH DƯỠNG: Bổ sung Choline! Gia tăng nguyên liệu Acetylcholine tăng độ tập trung (+2.5 ACh/tick trong 15s).")
            st.rerun()

    with diet_cols[3]:
        glutamate_disabled = st.session_state.stats["memory"] < 15.0
        if st.button("🥦 Bổ sung Glutamate (-15 MB)", disabled=glutamate_disabled, use_container_width=True, help="Tiền chất GABA: Tăng +3.0 GABA/tick liên tục trong 15 ticks."):
            st.session_state.stats["memory"] -= 15.0
            st.session_state.active_buffs["glutamate"] = 15
            add_log("🥦 DINH DƯỠNG: Bổ sung Glutamate! Thúc đẩy tổng hợp GABA giúp dập tắt hưng phấn động kinh và giảm stress (+3.0 GABA/tick trong 15s).")
            st.rerun()

    # Dynamic Game Event Modal/Alert
    if st.session_state.current_event:
        ev = st.session_state.current_event
        st.info(f"### 🛑 BIẾN CỐ NÃO BỘ: {ev['title']}")
        st.write(ev["desc"])

        choice_cols = st.columns(len(ev["choices"]))
        for idx, choice in enumerate(ev["choices"]):
            with choice_cols[idx]:
                if st.button(f"{choice['label']}\n\n({choice['effect']})", key=f"ev_btn_{idx}"):
                    choice["apply"]()
                    st.rerun()

    # Left Column: Interactive Grid | Right Column: Control & Upgrades
    game_cols = st.columns([5, 4])

    with game_cols[0]:
        st.markdown("#### ⚡ Bản Đồ Điện Thế Nơ-ron (6x6 Grid)")
        st.caption("Nhấp vào bất kỳ ô nào trong lưới để lựa chọn và định hướng nơ-ron ở Bảng điều khiển phía dưới.")

        # Grid rendering via Streamlit buttons
        grid = st.session_state.neuron_grid
        selected_r, selected_c = st.session_state.selected_cell

        for r in range(GRID_SIZE):
            row_cols = st.columns(GRID_SIZE)
            for c in range(GRID_SIZE):
                cell = grid[r][c]
                ctype = cell["type"]
                charge = cell["charge"]
                threshold = cell["threshold"]
                direction = cell.get("direction", "All")

                emoji = "⚫"
                if ctype == "Sensory":
                    emoji = "⚡"
                elif ctype == "Interneuron":
                    emoji = "🧠"
                elif ctype == "Motor":
                    emoji = "💪"

                dir_symbols = {
                    "All": "🌐",
                    "Up": "⬆️",
                    "Right": "➡️",
                    "Down": "⬇️",
                    "Left": "⬅️"
                }
                sym = dir_symbols.get(direction, "🌐")

                plaque_marker = "🧬" if (cell.get("amyloid_plaque", False) and ctype != "Empty") else ""
                label = f"{emoji}{sym}{plaque_marker}\n({charge:.2f})"
                is_selected = (r == selected_r and c == selected_c)
                border_style = "🔴 " if is_selected else ""

                if cell["charge"] >= threshold and ctype != "Empty":
                    btn_label = f"🔥 {label}"
                else:
                    btn_label = f"{border_style}{label}"

                if row_cols[c].button(btn_label, key=f"cell_{r}_{c}", use_container_width=True):
                    st.session_state.selected_cell = (r, c)
                    st.rerun()

        # Save & Load Circuit Codes Panel
        st.markdown("---")
        st.markdown("##### 💾 Lưu & Tải Sơ Đồ Mạch Thần Kinh (Circuit Share Codes)")
        st.caption("Sao chép mã chia sẻ mạch nơ-ron hiện tại hoặc nhập mã của người khác để xây dựng nhanh!")

        share_cols = st.columns([3, 1])
        with share_cols[0]:
            current_code = serialize_grid(st.session_state.neuron_grid)
            code_input = st.text_input("Mã sơ đồ mạch hiện tại (Copy-paste):", value=current_code, key="cur_code_text")
        with share_cols[1]:
            if st.button("📥 Tải sơ đồ (Load Code)", use_container_width=True):
                loaded_grid = deserialize_grid(code_input)
                if loaded_grid:
                    st.session_state.neuron_grid = loaded_grid
                    add_log("📥 TẢI SƠ ĐỒ: Khôi phục và cấy ghép sơ đồ mạch nơ-ron thành công!")
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

        # Cell configuration section below grid
        st.markdown("---")
        st.markdown(f"#### 🛠️ Bảng Điều Khiển Nơ-ron được chọn: **Hàng {selected_r + 1}, Cột {selected_c + 1}**")

        current_cell = grid[selected_r][selected_c]
        st.write(f"Trạng thái hiện tại: **{current_cell['type']}** (Điện tích tích lũy: `{current_cell['charge']:.2f}/{current_cell['threshold']:.2f}`)")

        ach_discount = 1.0 - (st.session_state.chemicals["acetylcholine"] / 200.0)
        cost_sensory = int(25 * ach_discount)
        cost_inter = int(15 * ach_discount)
        cost_motor = int(40 * ach_discount)

        edit_cols = st.columns(4)

        with edit_cols[0]:
            sensory_disabled = current_cell["type"] == "Sensory" or st.session_state.stats["memory"] < cost_sensory
            if st.button(f"⚡ Sensory Neuron\n(Phí: {cost_sensory} MB)", disabled=sensory_disabled, use_container_width=True):
                st.session_state.stats["memory"] -= cost_sensory
                grid[selected_r][selected_c] = {
                    "type": "Sensory",
                    "charge": 0.0,
                    "threshold": 0.4,
                    "fire_rate": 0.3,
                    "last_fired": -1,
                    "direction": "All",
                    "weight": 1.0,
                    "amyloid_plaque": False
                }
                add_log(f"Cấy ghép Nơ-ron cảm giác (Sensory) tại [{selected_r+1},{selected_c+1}] (-{cost_sensory} MB)")
                st.rerun()

        with edit_cols[1]:
            inter_disabled = current_cell["type"] == "Interneuron" or st.session_state.stats["memory"] < cost_inter
            if st.button(f"🧠 Interneuron\n(Phí: {cost_inter} MB)", disabled=inter_disabled, use_container_width=True):
                st.session_state.stats["memory"] -= cost_inter
                grid[selected_r][selected_c] = {
                    "type": "Interneuron",
                    "charge": 0.0,
                    "threshold": 0.5,
                    "fire_rate": 0.0,
                    "last_fired": -1,
                    "direction": "All",
                    "weight": 1.0,
                    "amyloid_plaque": False
                }
                add_log(f"Cấy ghép Nơ-ron liên kết (Interneuron) tại [{selected_r+1},{selected_c+1}] (-{cost_inter} MB)")
                st.rerun()

        with edit_cols[2]:
            motor_disabled = current_cell["type"] == "Motor" or st.session_state.stats["memory"] < cost_motor
            if st.button(f"💪 Motor Neuron\n(Phí: {cost_motor} MB)", disabled=motor_disabled, use_container_width=True):
                st.session_state.stats["memory"] -= cost_motor
                grid[selected_r][selected_c] = {
                    "type": "Motor",
                    "charge": 0.0,
                    "threshold": 0.6,
                    "fire_rate": 0.0,
                    "last_fired": -1,
                    "direction": "All",
                    "weight": 1.0,
                    "amyloid_plaque": False
                }
                add_log(f"Cấy ghép Nơ-ron vận động (Motor) tại [{selected_r+1},{selected_c+1}] (-{cost_motor} MB)")
                st.rerun()

        with edit_cols[3]:
            delete_disabled = current_cell["type"] == "Empty"
            if st.button("❌ Loại bỏ / Xóa\n(Thu hồi 50%)", disabled=delete_disabled, use_container_width=True):
                refund = 0
                if current_cell["type"] == "Sensory":
                    refund = int(cost_sensory * 0.5)
                elif current_cell["type"] == "Interneuron":
                    refund = int(cost_inter * 0.5)
                elif current_cell["type"] == "Motor":
                    refund = int(cost_motor * 0.5)

                st.session_state.stats["memory"] += refund
                grid[selected_r][selected_c] = {
                    "type": "Empty",
                    "charge": 0.0,
                    "threshold": 0.5,
                    "fire_rate": 0.0,
                    "last_fired": -1,
                    "direction": "All",
                    "weight": 1.0,
                    "amyloid_plaque": False
                }
                add_log(f"Xóa bỏ nơ-ron tại [{selected_r+1},{selected_c+1}] (Thu hồi +{refund} MB)")
                st.rerun()

        if current_cell["type"] != "Empty":
            st.write("---")
            axon_cols = st.columns(2)
            with axon_cols[0]:
                st.markdown("**🧭 Định hướng sợi trục (Axon Target):**")
                cur_dir = current_cell.get("direction", "All")
                dirs_list = ["All", "Up", "Right", "Down", "Left"]
                dir_names = {
                    "All": "🌐 Bốn phía (All)",
                    "Up": "⬆️ Phía trên (Up)",
                    "Right": "➡️ Phía phải (Right)",
                    "Down": "⬇️ Phía dưới (Down)",
                    "Left": "⬅️ Phía trái (Left)"
                }
                selected_new_dir = st.selectbox(
                    "Chọn hướng nơ-ron truyền tải:",
                    dirs_list,
                    index=dirs_list.index(cur_dir),
                    format_func=lambda x: dir_names[x],
                    key=f"dir_select_{selected_r}_{selected_c}"
                )
                if not isinstance(selected_new_dir, str):
                    selected_new_dir = cur_dir
                if selected_new_dir != cur_dir and selected_new_dir in dir_names:
                    grid[selected_r][selected_c]["direction"] = selected_new_dir
                    add_log(f"Định hướng lại trục nơ-ron [{selected_r+1},{selected_c+1}] thành {dir_names[selected_new_dir]}")
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
                    key=f"weight_select_{selected_r}_{selected_c}"
                )
                if not isinstance(new_weight, (int, float)):
                    new_weight = cur_weight
                if new_weight != cur_weight:
                    grid[selected_r][selected_c]["weight"] = new_weight
                    try:
                        weight_str = f"x{new_weight:.1f}"
                    except TypeError:
                        weight_str = f"x{new_weight}"
                    add_log(f"Thay đổi trọng số nơ-ron [{selected_r+1},{selected_c+1}] thành {weight_str}!")
                    st.rerun()

    with game_cols[1]:
        st.markdown("#### ⚙️ Trung Tâm Điều Hành & Nâng Cấp Lỗ Não")

        sim_controls = st.columns(3)
        with sim_controls[0]:
            play_label = "⏸️ Tạm Dừng" if st.session_state.playing else "▶️ Chạy Mô Phỏng"
            if st.button(play_label, use_container_width=True):
                st.session_state.playing = not st.session_state.playing
                st.rerun()
        with sim_controls[1]:
            if st.button("⏭️ Bước Tiếp Theo (Step)", use_container_width=True, disabled=st.session_state.playing):
                run_simulation_tick()
                st.rerun()
        with sim_controls[2]:
            if st.button("🔄 Khởi Tạo Lại Não", use_container_width=True):
                if "game_initialized" in st.session_state:
                    del st.session_state.game_initialized
                init_game_state()
                st.rerun()

        tick_speed_val = st.slider("Tốc độ mô phỏng (Giây/Tick)", min_value=0.2, max_value=2.0, value=1.0, step=0.1)
        if not isinstance(tick_speed_val, (int, float)):
            tick_speed_val = 1.0
        st.session_state.tick_speed = tick_speed_val

        # UPGRADE: Render Anatomical Brain Lobe Status with Dentate Gyrus upgrade!
        st.markdown("---")
        st.markdown("##### 🗺️ Bản Đồ Giải Phẫu Thùy Não (Anatomical Lobe Status)")
        st.caption("Trạng thái nâng cấp các cấu trúc giải phẫu sinh học quan trọng.")

        upgrades = st.session_state.upgrades
        pfc_status = "Đã tích hợp 🟢" if upgrades.get("pfc", 0) == 1 else "Chưa mở khóa ⚪"
        pruning_status = "Đã kích hoạt 🟢" if upgrades.get("pruning", 0) == 1 else "Chưa mở khóa ⚪"
        gly_status = "Đã tích hợp (Max Fuel 150) 🟢" if upgrades.get("glycogen_shunt", 0) == 1 else "Chưa mở khóa ⚪"
        dentate_status = f"Đã tích hợp (Tế bào gốc) [Lv.{upgrades.get('dentate_gyrus', 0)}] 🟢" if upgrades.get("dentate_gyrus", 0) >= 1 else "Chưa mở khóa ⚪"

        plaques_count = sum(1 for r in range(GRID_SIZE) for c in range(GRID_SIZE) if st.session_state.neuron_grid[r][c].get("amyloid_plaque", False))
        adra2a_active = "Có 🟢" if "ADRA2A" in st.session_state.active_genes else "Không ⚪"
        trem2_active = "Có 🟢" if "TREM2" in st.session_state.active_genes else "Không ⚪"
        occipital_status = f"Lv.{upgrades.get('occipital_lobe', 0)}" if upgrades.get("occipital_lobe", 0) >= 1 else "Chưa mở khóa ⚪"
        temporal_status = f"Lv.{upgrades.get('temporal_lobe', 0)}" if upgrades.get("temporal_lobe", 0) >= 1 else "Chưa mở khóa ⚪"
        parietal_status = f"Lv.{upgrades.get('parietal_lobe', 0)}" if upgrades.get("parietal_lobe", 0) >= 1 else "Chưa mở khóa ⚪"
        pituitary_status = f"Lv.{upgrades.get('pituitary_gland', 0)}" if upgrades.get("pituitary_gland", 0) >= 1 else "Chưa mở khóa ⚪"

        brain_art = f"""
        [ Frontal Cortex: Lv.{upgrades['cortex']} ] ---------.
                   |                         |
         [ Prefrontal PFC: {pfc_status} ]    |--- ( Cortex )
                   |                         |
        [ Myelin Sheath: Lv.{upgrades['myelin']} ] <--------'
                   |
        [ Hippocampus: Lv.{upgrades['hippocampus']} ] (Ghi nhớ)
                   |
         [ Synaptic Pruning: {pruning_status} ]
                   |
        [ Thalamus (Đồi Thị): Lv.{upgrades.get('thalamus', 0)} ] (Kích sensory)
                   |
        [ Dentate Gyrus (Thùy Răng): {dentate_status} ] (Neurogenesis)
                   |
        [ Occipital Lobe (Thùy Chẩm): {occipital_status} ] (Xử lý thị giác 2x)
                   |
        [ Temporal Lobe (Thùy Thái Dương): {temporal_status} ] (Xử lý âm thanh & LTP)
                   |
        [ Parietal Lobe (Thùy Đỉnh): {parietal_status} ] (Xử lý không gian & Gating)
                   |
        [ Pituitary Gland (Tuyến Yên): {pituitary_status} ] (Hormone cascade & Oxytocin)
                   |
        [ Amygdala (Hạch Hạnh Nhân): Lv.{upgrades.get('amygdala', 0)} ] (Hạ stress)
                   |
        [ Astrocytic Glycogen Shunt: {gly_status} ] (Kho trữ năng lượng)
                   |
        [ Cerebellum: Lv.{upgrades['cerebellum']} ] (Tiểu não hạ stress)
                   |
        [ Brainstem: Lv.{upgrades['brainstem']} ] (Hành não cấp năng lượng)

        -------------------------------------------------------------
        🔬 Chỉ số Alzheimer: Tích lũy {plaques_count} mảng bám Amyloid-Beta
        🧬 Đột biến ADRA2A (Bình ổn hoảng loạn): {adra2a_active}
        🧬 Đột biến TREM2 (Tăng dọn dẹp mảng bám): {trem2_active}
        """
        st.code(brain_art, language="text")

        st.markdown("---")

        # UPGRADE: Genetic Mutation Modifier Board Panel (Now supports DRD2, COMT-Met, ADRA2A, TREM2!)
        st.markdown("##### 🧬 Bản Đồ Biến Dị Di Truyền (Genetic Mutation Modifiers)")
        st.caption("Kích hoạt đột biến gen để áp dụng các thay đổi hóa học và liên kết vĩnh viễn cho tế bào.")

        genes_list = ["APOE4", "BDNF", "COMT", "GABRA1", "DRD4", "SHANK3", "MAOA", "CHRNA7", "PGC-1alpha", "SLC6A4", "DRD2", "COMT-Met", "ADRA2A", "TREM2"]
        genes_desc = {
            "APOE4": "👵 APOE4: Nhân đôi tốc độ Alzheimer, nhưng +50% IQ ban đầu.",
            "BDNF": "🌱 BDNF: Plasticity dẻo dai hơn 1.5x, đẩy nhanh Hebbian learning.",
            "COMT": "🥤 COMT: Dopamine bền vững giảm chậm hơn 40%, nhưng Stress giảm chậm 20%.",
            "GABRA1": "🛡️ GABRA1: Giảm 35% Stress quá kích sinh ra do Động kinh.",
            "DRD4": "🎰 DRD4: Nhân đôi Dopamine khi Motor phát xung, nhưng thiếu hụt Dopamine nhân đôi stress hại sanity.",
            "SHANK3": "🧱 SHANK3: Thần kinh giáp tự, tăng +15% hiệu suất truyền tải tín hiệu nơ-ron.",
            "MAOA": "🧘 MAOA: Dopamine và Serotonin phân hủy chậm hơn 30%, nhưng phát xung sinh stress +40%.",
            "CHRNA7": "⚡ CHRNA7: Acetylcholine gia tăng +25% hiệu quả, nhưng tiêu hao năng lượng cơ thể tăng 15%.",
            "PGC-1alpha": "🔋 PGC-1alpha: Đột biến nguyên sinh tế bào, gia tăng +40 Max Fuel của não bộ.",
            "SLC6A4": "🧬 SLC6A4: Đột biến vận chuyển Serotonin, tăng thời gian duy trì SSRI thêm 1.5 lần.",
            "DRD2": "🕹️ DRD2: Tăng cường thụ thể Dopamine, tăng +50% hiệu ứng động lực Sensory, nhưng stress cao hại gấp rưỡi sanity.",
            "COMT-Met": "🧠 COMT-Met: Đột biến thùy trán siêu trí tuệ, +30% IQ nhận thức từ Motor, nhưng stress giảm chậm đi 30%.",
            "ADRA2A": "🧘 ADRA2A: Bình ổn hoảng loạn, giảm 40% sát thương lên Sanity của Panic Attack và nâng ngưỡng hoảng loạn lên 100%.",
            "TREM2": "🧹 TREM2: Kích hoạt hệ thực bào siêu vi, nhân đôi tốc độ/tỷ lệ thực bào dọn dẹp mảng bám Amyloid-Beta (lên 40% mỗi tick)."
        }

        # Multiselect for genes selection
        active_genes = st.multiselect(
            "Đột biến gen kích hoạt vĩnh viễn:",
            genes_list,
            default=st.session_state.active_genes,
            format_func=lambda x: genes_desc[x]
        )
        if not isinstance(active_genes, list):
            active_genes = st.session_state.active_genes
        if active_genes != st.session_state.active_genes:
            st.session_state.active_genes = active_genes
            add_log(f"🧬 DI TRUYỀN: Điều chỉnh đột biến gen kích hoạt: {active_genes}")
            st.rerun()

        st.markdown("---")
        st.markdown("##### 🛒 Nâng Cấp Thùy Não (Mở rộng cấu trúc nhận thức)")
        st.caption("Sử dụng điểm IQ tích lũy từ các hành động Motor thành công để tiến hóa các vùng não.")

        # Upgrade Item 1: Brainstem
        cost_stem = int(25 * (1.5 ** upgrades["brainstem"]))
        upgrade_cols_1 = st.columns([3, 1])
        with upgrade_cols_1[0]:
            st.write(f"**Hành não (Brainstem) [Lv.{upgrades['brainstem']}]**\nTăng tốc sản sinh Oxy & Glucose (+{4.0 + upgrades['brainstem']*2.0} đơn vị/tick)")
        with upgrade_cols_1[1]:
            if st.button(f"Mua ({cost_stem} IQ)", key="up_stem", disabled=st.session_state.stats["iq"] < cost_stem, use_container_width=True):
                st.session_state.stats["iq"] -= cost_stem
                upgrades["brainstem"] += 1
                add_log(f"Nâng cấp Hành não lên cấp {upgrades['brainstem']}!")
                st.rerun()

        # Upgrade Item 2: Cerebellum
        cost_cere = int(35 * (1.6 ** upgrades["cerebellum"]))
        upgrade_cols_2 = st.columns([3, 1])
        with upgrade_cols_2[0]:
            st.write(f"**Tiểu não (Cerebellum) [Lv.{upgrades['cerebellum']}]**\nGiảm căng thẳng nhanh hơn (-{1.5 + upgrades['cerebellum']*1.0} stress/tick)")
        with upgrade_cols_2[1]:
            if st.button(f"Mua ({cost_cere} IQ)", key="up_cere", disabled=st.session_state.stats["iq"] < cost_cere, use_container_width=True):
                st.session_state.stats["iq"] -= cost_cere
                upgrades["cerebellum"] += 1
                add_log(f"Nâng cấp Tiểu não lên cấp {upgrades['cerebellum']}!")
                st.rerun()

        # Upgrade Item 3: Hippocampus
        cost_hippo = int(50 * (1.7 ** upgrades["hippocampus"]))
        upgrade_cols_3 = st.columns([3, 1])
        with upgrade_cols_3[0]:
            st.write(f"**Hải mã (Hippocampus) [Lv.{upgrades['hippocampus']}]**\nTăng sản lượng Trí nhớ tích lũy (+{40 * upgrades['hippocampus']}% hiệu ứng)")
        with upgrade_cols_3[1]:
            if st.button(f"Mua ({cost_hippo} IQ)", key="up_hippo", disabled=st.session_state.stats["iq"] < cost_hippo, use_container_width=True):
                st.session_state.stats["iq"] -= cost_hippo
                upgrades["hippocampus"] += 1
                add_log(f"Nâng cấp Thùy Hải mã lên cấp {upgrades['hippocampus']}!")
                st.rerun()

        # Upgrade Item 4: Cortex
        cost_cortex = int(75 * (1.8 ** upgrades["cortex"]))
        upgrade_cols_4 = st.columns([3, 1])
        with upgrade_cols_4[0]:
            st.write(f"**Vỏ não Frontal (Cortex) [Lv.{upgrades['cortex']}]**\nKhuyếch đại điểm IQ hành động nhận thức (+{60 * upgrades['cortex']}% hiệu ứng)")
        with upgrade_cols_4[1]:
            if st.button(f"Mua ({cost_cortex} IQ)", key="up_cortex", disabled=st.session_state.stats["iq"] < cost_cortex, use_container_width=True):
                st.session_state.stats["iq"] -= cost_cortex
                upgrades["cortex"] += 1
                add_log(f"Nâng cấp Vỏ não Frontal lên cấp {upgrades['cortex']}!")
                st.rerun()

        # Upgrade Item 5: Myelin
        cost_myelin = int(40 * (1.5 ** upgrades["myelin"]))
        upgrade_cols_5 = st.columns([3, 1])
        with upgrade_cols_5[0]:
            st.write(f"**Bao Myelin (Sợi trục) [Lv.{upgrades['myelin']}]**\nTối ưu hóa truyền dẫn xung, hạn chế tiêu hao điện thế (+5% hiệu quả truyền tải)")
        with upgrade_cols_5[1]:
            if st.button(f"Mua ({cost_myelin} IQ)", key="up_myelin", disabled=st.session_state.stats["iq"] < cost_myelin, use_container_width=True):
                st.session_state.stats["iq"] -= cost_myelin
                upgrades["myelin"] += 1
                add_log(f"Nâng cấp mức độ Myelin hóa sợi trục lên cấp {upgrades['myelin']}!")
                st.rerun()

        # Amygdala Anatomy Upgrade Item
        cost_amygdala = int(45 * (1.6 ** upgrades.get("amygdala", 0)))
        upgrade_cols_amygdala = st.columns([3, 1])
        with upgrade_cols_amygdala[0]:
            st.write(f"**Hạch Hạnh Nhân (Amygdala) [Lv.{upgrades.get('amygdala', 0)}]**\nGiảm stress phát sinh do phát xung thần kinh (-15% stress phát sinh mỗi cấp)")
        with upgrade_cols_amygdala[1]:
            if st.button(f"Mua ({cost_amygdala} IQ)", key="up_amygdala", disabled=st.session_state.stats["iq"] < cost_amygdala, use_container_width=True):
                st.session_state.stats["iq"] -= cost_amygdala
                upgrades["amygdala"] = upgrades.get("amygdala", 0) + 1
                add_log(f"Nâng cấp Hạch hạnh nhân Amygdala lên cấp {upgrades['amygdala']}!")
                st.rerun()

        # Thalamus Anatomy Upgrade Item
        cost_thalamus = int(45 * (1.6 ** upgrades.get("thalamus", 0)))
        upgrade_cols_thalamus = st.columns([3, 1])
        with upgrade_cols_thalamus[0]:
            st.write(f"**Đồi Thị (Thalamus) [Lv.{upgrades.get('thalamus', 0)}]**\nTăng tốc độ tích lũy điện tích tự động của Sensory cells (+20% tốc độ mỗi cấp)")
        with upgrade_cols_thalamus[1]:
            if st.button(f"Mua ({cost_thalamus} IQ)", key="up_thalamus", disabled=st.session_state.stats["iq"] < cost_thalamus, use_container_width=True):
                st.session_state.stats["iq"] -= cost_thalamus
                upgrades["thalamus"] = upgrades.get("thalamus", 0) + 1
                add_log(f"Nâng cấp Đồi thị Thalamus lên cấp {upgrades['thalamus']}!")
                st.rerun()

        # UPGRADE: Dentate Gyrus Anatomy Upgrade Item
        cost_dentate = int(60 * (1.6 ** upgrades.get("dentate_gyrus", 0)))
        upgrade_cols_dentate = st.columns([3, 1])
        with upgrade_cols_dentate[0]:
            st.write(f"**Thùy Răng (Dentate Gyrus) [Lv.{upgrades.get('dentate_gyrus', 0)}]**\nTự động sản sinh tế bào liên kết thần kinh mới (Neurogenesis) khi nồng độ Serotonin dồi dào (>60%).")
        with upgrade_cols_dentate[1]:
            if st.button(f"Mua ({cost_dentate} IQ)", key="up_dentate", disabled=st.session_state.stats["iq"] < cost_dentate, use_container_width=True):
                st.session_state.stats["iq"] -= cost_dentate
                upgrades["dentate_gyrus"] = upgrades.get("dentate_gyrus", 0) + 1
                add_log(f"Nâng cấp Thùy Răng Dentate Gyrus lên cấp {upgrades['dentate_gyrus']}!")
                st.rerun()

        # UPGRADE: Occipital Lobe Upgrade Item
        cost_occipital = int(50 * (1.6 ** upgrades.get("occipital_lobe", 0)))
        upgrade_cols_occipital = st.columns([3, 1])
        with upgrade_cols_occipital[0]:
            st.write(f"**Thùy Chẩm (Occipital Lobe) [Lv.{upgrades.get('occipital_lobe', 0)}]**\nXử lý kích thích thị giác ngẫu nhiên (Visual Spark) mỗi 10 ticks: Khớp hướng nơ-ron cảm giác để nhận sạc nơ-ron x2 và gấp đôi điểm IQ nhận thức từ Motor.")
        with upgrade_cols_occipital[1]:
            if st.button(f"Mua ({cost_occipital} IQ)", key="up_occipital", disabled=st.session_state.stats["iq"] < cost_occipital, use_container_width=True):
                st.session_state.stats["iq"] -= cost_occipital
                upgrades["occipital_lobe"] = upgrades.get("occipital_lobe", 0) + 1
                add_log(f"Nâng cấp Thùy Chẩm Occipital Lobe lên cấp {upgrades['occipital_lobe']}!")
                st.rerun()

        # UPGRADE: Temporal Lobe Upgrade Item
        cost_temporal = int(50 * (1.6 ** upgrades.get("temporal_lobe", 0)))
        upgrade_cols_temporal = st.columns([3, 1])
        with upgrade_cols_temporal[0]:
            st.write(f"**Thùy Thái Dương (Temporal Lobe) [Lv.{upgrades.get('temporal_lobe', 0)}]**\nXử lý kích thích thính giác ngẫu nhiên (Auditory Stimulus) mỗi 15 ticks: Tần số cao >600 Hz tăng Dopamine/GABA, tần số thấp <=600 Hz tăng Acetylcholine. Tần số cộng hưởng 400-500 Hz nhân ba (3x) sản lượng Trí nhớ từ Motor.")
        with upgrade_cols_temporal[1]:
            if st.button(f"Mua ({cost_temporal} IQ)", key="up_temporal", disabled=st.session_state.stats["iq"] < cost_temporal, use_container_width=True):
                st.session_state.stats["iq"] -= cost_temporal
                upgrades["temporal_lobe"] = upgrades.get("temporal_lobe", 0) + 1
                add_log(f"Nâng cấp Thùy Thái Dương Temporal Lobe lên cấp {upgrades['temporal_lobe']}!")
                st.rerun()

        # UPGRADE: Parietal Lobe Upgrade Item
        cost_parietal = int(50 * (1.6 ** upgrades.get("parietal_lobe", 0)))
        upgrade_cols_parietal = st.columns([3, 1])
        with upgrade_cols_parietal[0]:
            st.write(f"**Thùy Đỉnh (Parietal Lobe) [Lv.{upgrades.get('parietal_lobe', 0)}]**\nXử lý bản đồ định vị không gian mỗi 18 ticks: Khi dòng điện tích truyền qua điểm Gating ngẫu nhiên sẽ kích hoạt Somatosensory Gating hạ -20% Stress và cắt 50% tiêu hao năng lượng trong 3 ticks.")
        with upgrade_cols_parietal[1]:
            if st.button(f"Mua ({cost_parietal} IQ)", key="up_parietal", disabled=st.session_state.stats["iq"] < cost_parietal, use_container_width=True):
                st.session_state.stats["iq"] -= cost_parietal
                upgrades["parietal_lobe"] = upgrades.get("parietal_lobe", 0) + 1
                add_log(f"Nâng cấp Thùy Đỉnh Parietal Lobe lên cấp {upgrades['parietal_lobe']}!")
                st.rerun()

        # UPGRADE: Pituitary Gland Upgrade Item
        cost_pituitary = int(50 * (1.6 ** upgrades.get("pituitary_gland", 0)))
        upgrade_cols_pituitary = st.columns([3, 1])
        with upgrade_cols_pituitary[0]:
            st.write(f"**Tuyến Yên (Pituitary Gland) [Lv.{upgrades.get('pituitary_gland', 0)}]**\nĐiều hòa giải phóng hormone tuyến yên mỗi 20 ticks: Kích hoạt Oxytocin Surge kéo dài 5 ticks giúp triệt tiêu 50% Stress phát sinh và nhân đôi tốc độ hồi phục Tỉnh táo (Sanity).")
        with upgrade_cols_pituitary[1]:
            if st.button(f"Mua ({cost_pituitary} IQ)", key="up_pituitary", disabled=st.session_state.stats["iq"] < cost_pituitary, use_container_width=True):
                st.session_state.stats["iq"] -= cost_pituitary
                upgrades["pituitary_gland"] = upgrades.get("pituitary_gland", 0) + 1
                add_log(f"Nâng cấp Tuyến Yên Pituitary Gland lên cấp {upgrades['pituitary_gland']}!")
                st.rerun()

        # UPGRADE: LTP Consolidator Upgrade Item
        cost_ltp = 100
        upgrade_cols_ltp = st.columns([3, 1])
        with upgrade_cols_ltp[0]:
            st.write(f"**Hebbian LTP Consolidator [Lv.{upgrades.get('ltp_consolidator', 0)}/1]**\nCủng cố liên kết trí nhớ dài hạn: Tự động chuyển đổi 30% bộ nhớ sang IQ vĩnh viễn Hebbian LTP sau mỗi 12 ticks hoạt động.")
        with upgrade_cols_ltp[1]:
            ltp_disabled = upgrades.get("ltp_consolidator", 0) >= 1 or st.session_state.stats["iq"] < cost_ltp
            if st.button(f"Mua ({cost_ltp} IQ)", key="up_ltp", disabled=ltp_disabled, use_container_width=True):
                st.session_state.stats["iq"] -= cost_ltp
                upgrades["ltp_consolidator"] = 1
                add_log("💾 NÂNG CẤP: Kích hoạt Hebbian LTP Consolidator củng cố bộ nhớ dài hạn tự động!")
                st.rerun()

        # Astrocytic Glycogen Shunt Upgrade Item
        cost_shunt = 150
        upgrade_cols_shunt = st.columns([3, 1])
        with upgrade_cols_shunt[0]:
            st.write(f"**Kho Astrocytic Glycogen [Lv.{upgrades.get('glycogen_shunt', 0)}/1]**\nTế bào hình sao liên kết mạch máu: Tăng sức chứa Năng lượng cực đại lên 150% (bình thường 100%).")
        with upgrade_cols_shunt[1]:
            shunt_disabled = upgrades.get("glycogen_shunt", 0) >= 1 or st.session_state.stats["iq"] < cost_shunt
            if st.button(f"Mua ({cost_shunt} IQ)", key="up_shunt", disabled=shunt_disabled, use_container_width=True):
                st.session_state.stats["iq"] -= cost_shunt
                upgrades["glycogen_shunt"] = 1
                add_log("🔋 NÂNG CẤP: Kích hoạt Astrocytic Glycogen Shunt nâng tối đa Năng lượng bộ não lên 150!")
                st.rerun()

        # Synaptic Pruning Upgrade Panel
        cost_pruning = 120
        upgrade_cols_6 = st.columns([3, 1])
        with upgrade_cols_6[0]:
            st.write(f"**Cắt tỉa nơ-ron (Synaptic Pruning) [Lv.{upgrades['pruning']}/1]**\nTự động xóa nơ-ron liên kết nhàn rỗi (>15 ticks không phát xung) và hoàn lại +75% MB phí cấy ghép.")
        with upgrade_cols_6[1]:
            pruning_disabled = upgrades["pruning"] >= 1 or st.session_state.stats["iq"] < cost_pruning
            if st.button(f"Mua ({cost_pruning} IQ)", key="up_pruning", disabled=pruning_disabled, use_container_width=True):
                st.session_state.stats["iq"] -= cost_pruning
                upgrades["pruning"] = 1
                add_log("✂️ NÂNG CẤP: Kích hoạt Synaptic Pruning tự động cắt tỉa liên kết dư thừa!")
                st.rerun()

        # Prefrontal Cortex PFC Decision Maker Panel
        cost_pfc = 200
        upgrade_cols_7 = st.columns([3, 1])
        with upgrade_cols_7[0]:
            st.write(f"**Vỏ não trước trán (Prefrontal Cortex PFC) [Lv.{upgrades['pfc']}/1]**\nTự động phân tích và đưa ra quyết định tối ưu cho mọi biến cố ngẫu nhiên của não bộ.")
        with upgrade_cols_7[1]:
            pfc_disabled = upgrades["pfc"] >= 1 or st.session_state.stats["iq"] < cost_pfc
            if st.button(f"Mua ({cost_pfc} IQ)", key="up_pfc", disabled=pfc_disabled, use_container_width=True):
                st.session_state.stats["iq"] -= cost_pfc
                upgrades["pfc"] = 1
                add_log("🧠 NÂNG CẤP: Kích hoạt Thùy trán trước PFC AI tự động ra quyết định biến cố!")
                st.rerun()

    # Dynamic run loop inside Streamlit using st.empty for real-time updates
    if st.session_state.playing:
        time.sleep(st.session_state.tick_speed)
        run_simulation_tick()
        st.rerun()

    with st.expander("🏆 Hệ Thống Nhiệm Vụ Nhận Thức (Cognitive Challenges)", expanded=True):
        st.caption("Đạt các mốc kỹ thuật cấu trúc mạng lưới nơ-ron hoặc hóa chất đặc thù để kích hoạt quà tặng tài nguyên.")

        missions = st.session_state.missions
        for m_key, m_info in missions.items():
            m_cols = st.columns([3, 2, 1])
            with m_cols[0]:
                st.write(f"**{m_info['name']}**\n*Mục tiêu:* {m_info['target']}")
            with m_cols[1]:
                if m_info["status"] == "Completed":
                    if m_info["reward_claimed"]:
                        st.success("Đã nhận thưởng ✅")
                    else:
                        st.info("Sẵn sàng nhận thưởng 🎁")
                else:
                    st.write("🔄 Đang thực hiện...")
            with m_cols[2]:
                claim_disabled = m_info["status"] != "Completed" or m_info["reward_claimed"]
                if st.button(f"Nhận thưởng ({m_info['desc']})", key=f"claim_{m_key}", disabled=claim_disabled, use_container_width=True):
                    m_info["reward_claimed"] = True
                    if m_key == "reflex":
                        st.session_state.stats["memory"] = min(1000.0, st.session_state.stats["memory"] + 100.0)
                        add_log("🎁 NHẬN THƯỞNG: Nhận +100 MB Trí nhớ thành công!")
                    elif m_key == "loop":
                        st.session_state.stats["iq"] += 300.0
                        add_log("🎁 NHẬN THƯỞNG: Nhận +300 IQ nhận thức thành công!")
                    elif m_key == "zen":
                        st.session_state.chemicals["dopamine"] = min(100.0, st.session_state.chemicals["dopamine"] + 40.0)
                        st.session_state.chemicals["serotonin"] = min(100.0, st.session_state.chemicals["serotonin"] + 40.0)
                        add_log("🎁 NHẬN THƯỞNG: Nhận +40 Dopamine và +40 Serotonin thành công!")
                    elif m_key == "marathon":
                        st.session_state.chemicals["acetylcholine"] = min(100.0, st.session_state.chemicals["acetylcholine"] + 50.0)
                        st.session_state.stats["memory"] = min(1000.0, st.session_state.stats["memory"] + 200.0)
                        add_log("🎁 NHẬN THƯỞNG: Nhận +50 Acetylcholine và +200 MB Trí nhớ thành công!")
                    st.rerun()

    st.markdown("---")

    # Real-time analytics charts and logs
    chart_cols = st.columns([5, 4])
    with chart_cols[0]:
        st.markdown("##### 📈 Biểu Đồ Chỉ Số Hóa Sinh Thực Tế (40 ticks gần nhất)")
        hist_df = pd.DataFrame(st.session_state.history_data)
        st.line_chart(hist_df.set_index("tick"))

    with chart_cols[1]:
        st.markdown("##### 📝 Nhật Ký Xử Lý Thần Kinh (Neural Log)")
        st.code("\n".join(st.session_state.game_log), language="text")

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

# ----------------- TAB 2: VBOT1 WEB CHAT & SUMMARIZE -----------------
with tab2:
    st.subheader("🤖 Trợ Lý Trí Tuệ Nhân Tạo Song Song")
    st.write("Sử dụng mô hình Meta Llama 3 8B cho hội thoại tự nhiên và Google Gemini 1.5 Flash cho xử lý văn bản PDF hiệu quả cao.")

    vbot_cols = st.columns(3)
    with vbot_cols[0]:
        st.info(f"**Cổng Telegram**: {'Đang chạy ngầm' if TELEGRAM_TOKEN else 'Chưa cấu hình'}")
    with vbot_cols[1]:
        st.success(f"**Hệ Thống Llama 3**: {'Sẵn sàng (HuggingFace API)' if hf_client else 'Không khả dụng'}")
    with vbot_cols[2]:
        st.warning(f"**Hệ Thống Gemini**: {'Sẵn sàng (Google GenAI)' if GOOGLE_API_KEY else 'Không khả dụng'}")

    chat_sec, pdf_sec = st.columns(2)

    with chat_sec:
        st.markdown("#### 💬 Trò chuyện trực tuyến (Llama 3 8B)")

        if "web_chat_history" not in st.session_state:
            st.session_state.web_chat_history = []

        for speaker, message in st.session_state.web_chat_history:
            if speaker == "User":
                st.markdown(f"**🧑 Bạn**: {message}")
            else:
                st.markdown(f"**🤖 VBot1**: {message}")

        user_input = st.text_input("Nhập tin nhắn của bạn...", key="web_chat_input")
        if st.button("Gửi tin nhắn", key="web_send_btn") and user_input:
            st.session_state.web_chat_history.append(("User", user_input))
            if hf_client:
                try:
                    with st.spinner("VBot1 đang suy nghĩ..."):
                        messages = [{"role": "user", "content": user_input}]
                        completion = hf_client.chat_completion(
                            model="meta-llama/Meta-Llama-3-8B-Instruct",
                            messages=messages,
                            max_tokens=400
                        )
                        reply = completion.choices[0].message.content
                        st.session_state.web_chat_history.append(("VBot1", reply))
                except Exception as e:
                    st.session_state.web_chat_history.append(("VBot1", f"Lỗi kết nối API: {e}"))
            else:
                st.session_state.web_chat_history.append(("VBot1", "Tính năng chat Llama 3 chưa được cấu hình token (HF_TOKEN trống)."))
            st.rerun()

    with pdf_sec:
        st.markdown("#### 📄 Tóm Tắt PDF Chuyên Sâu (Gemini 1.5 Flash)")
        uploaded_file = st.file_uploader("Kéo thả hoặc chọn tệp tin PDF để phân tích:", type=["pdf"])

        if uploaded_file is not None:
            if st.button("🚀 Bắt đầu tóm tắt", use_container_width=True):
                with st.spinner("Đang xử lý tệp PDF và tạo tóm tắt chuyên sâu từ Gemini..."):
                    file_bytes = uploaded_file.read()
                    extracted_text = extract_pdf_text(file_bytes)
                    if extracted_text:
                        summary_result = summarize_with_gemini(extracted_text)
                        st.markdown("**📝 Kết quả tóm tắt từ AI Gemini 1.5 Flash:**")
                        st.write(summary_result)
                    else:
                        st.error("Không thể trích xuất văn bản từ tệp tin PDF này. Vui lòng kiểm tra lại định dạng tệp.")
