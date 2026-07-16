import streamlit as st
import time
import os
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
def check_network(host="8.8.8.8", port=53, timeout=3):
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

# UPGRADE: Serialization/Deserialization
def serialize_grid(grid):
    cells = []
    dir_map = {"All": "A", "Up": "U", "Right": "R", "Down": "D", "Left": "L"}
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            cell = grid[r][c]
            t_char = cell["type"][0] # E, S, I, M
            d_char = dir_map.get(cell.get("direction", "All"), "A")
            cells.append(f"{t_char}{d_char}")
    return "-".join(cells)

def deserialize_grid(code_string):
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
            if len(part) != 2:
                return None
            t_char, d_char = part[0], part[1]
            t_name = type_map.get(t_char, "Empty")
            d_name = dir_map.get(d_char, "All")

            row.append({
                "type": t_name,
                "charge": 0.0,
                "threshold": 0.4 if t_name == "Sensory" else (0.6 if t_name == "Motor" else 0.5),
                "fire_rate": 0.3 if t_name == "Sensory" else 0.0,
                "last_fired": -1,
                "direction": d_name
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
                    "direction": "All"
                })
            grid.append(row)

        # Default starting network structure
        grid[0][0] = {"type": "Sensory", "charge": 0.0, "threshold": 0.4, "fire_rate": 0.35, "last_fired": -1, "direction": "All"}
        grid[2][2] = {"type": "Interneuron", "charge": 0.0, "threshold": 0.5, "fire_rate": 0.0, "last_fired": -1, "direction": "All"}
        grid[5][5] = {"type": "Motor", "charge": 0.0, "threshold": 0.6, "fire_rate": 0.0, "last_fired": -1, "direction": "All"}

        st.session_state.neuron_grid = grid

        # Chemistry metrics
        st.session_state.chemicals = {
            "dopamine": 50.0,
            "serotonin": 50.0,
            "acetylcholine": 50.0,
            "energy": 100.0,
            "stress": 10.0,
            "sanity": 100.0
        }

        # Hormone Active Cooldowns
        st.session_state.cooldowns = {
            "doping": 0,
            "ssri": 0,
            "focus": 0
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
            "burnout_count": 0
        }

        # Upgrades
        st.session_state.upgrades = {
            "brainstem": 1,
            "cerebellum": 1,
            "hippocampus": 1,
            "cortex": 1,
            "myelin": 0,
            "plasticity": 0,
            "pruning": 0, # UPGRADE: Synaptic Pruning level (0 or 1)
            "pfc": 0      # UPGRADE: Prefrontal Cortex AI decision (0 or 1)
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
            "stress": [10.0]
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

    # Decrement active hormone abilities cooldowns
    cooldowns = st.session_state.get("cooldowns", {"doping": 0, "ssri": 0, "focus": 0})
    for k in cooldowns:
        if cooldowns[k] > 0:
            cooldowns[k] -= 1

    # UPGRADE: Synaptic Pruning (Forget idle connections)
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
                        # Pruning activated! Refund 75% memory
                        refund = int(cost_inter * 0.75)
                        st.session_state.stats["memory"] = min(1000.0, st.session_state.stats["memory"] + refund)
                        grid[r][c] = {
                            "type": "Empty",
                            "charge": 0.0,
                            "threshold": 0.5,
                            "fire_rate": 0.0,
                            "last_fired": -1,
                            "direction": "All"
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
    chems["energy"] = max(0.0, min(100.0, chems["energy"] + energy_generation - metabolic_cost))

    if chems["energy"] <= 0.0:
        add_log("⚠️ Cảnh báo: Bộ não cạn kiệt Glucose và Oxy! Không thể truyền tín hiệu.")
        chems["stress"] = max(0.0, min(100.0, chems["stress"] + 5.0))
        record_history(ticks, chems)
        return

    # 2. Sensory Stimuli Fire Check
    sensory_fires = 0
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            cell = grid[r][c]
            if cell["type"] == "Sensory":
                boost = 1.0 + (chems["dopamine"] / 100.0)
                cell["charge"] += cell["fire_rate"] * boost
                if cell["charge"] >= cell["threshold"]:
                    sensory_fires += 1

    # 3. Signal Propagation Model with Directional Synapses & Axon Growth
    next_charges = [[grid[r][c]["charge"] for c in range(GRID_SIZE)] for r in range(GRID_SIZE)]
    signals_fired = 0
    fired_cells = set()

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            cell = grid[r][c]
            if cell["type"] != "Empty" and cell["charge"] >= cell["threshold"]:
                fired_cells.add((r, c))
                cell["last_fired"] = ticks
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
                    signal_efficiency = 0.35 + (upgrades["myelin"] * 0.05)
                    transfer_charge = (cell["charge"] * signal_efficiency) / len(neighbors)

                    for nr, nc in neighbors:
                        next_charges[nr][nc] = min(1.0, next_charges[nr][nc] + transfer_charge)
                        if upgrades["plasticity"] > 0 and grid[nr][nc]["charge"] > 0.3:
                            grid[nr][nc]["threshold"] = max(0.2, grid[nr][nc]["threshold"] - 0.01)

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
                motor_yield_iq += 5.0 * iq_multiplier * focus_bonus

                mem_multiplier = 1.0 + (upgrades["hippocampus"] * 0.4)
                motor_yield_mem += 2.0 * mem_multiplier

                chems["dopamine"] = min(100.0, chems["dopamine"] + 8.0)
                chems["acetylcholine"] = max(0.0, chems["acetylcholine"] - 4.0)

    # Apply motor accomplishments
    if motor_fired_count > 0:
        st.session_state.stats["iq"] += motor_yield_iq
        st.session_state.stats["memory"] = min(1000.0, st.session_state.stats["memory"] + motor_yield_mem)
        add_log(f"🎯 Hành động Motor kích hoạt! Trùng hợp phát xung thành công (+{motor_yield_iq:.1f} IQ, +{motor_yield_mem:.1f} Trí nhớ)")

    # UPGRADE: Populate Web Audio Trigger
    st.session_state.audio_trigger = {
        "sensory": sensory_fires,
        "motor": motor_fired_count
    }

    # 5. Chemistry & Health Delta Calculations
    fire_stress = signals_fired * 1.5
    stress_clearance = 1.5 + (upgrades["cerebellum"] * 1.0)

    chems["stress"] = max(0.0, min(100.0, chems["stress"] + fire_stress - stress_clearance))

    serotonin_dampening = chems["serotonin"] * 0.1
    effective_stress = max(0.0, chems["stress"] - serotonin_dampening)

    if effective_stress > 60.0:
        sanity_damage = (effective_stress - 60.0) * 0.35
        chems["sanity"] = max(0.0, min(100.0, chems["sanity"] - sanity_damage))
        if sanity_damage > 1.0:
            add_log(f"⚡ Căng thẳng cực độ gây tổn hại myelin và nơ-ron! (-{sanity_damage:.1f} Tỉnh táo)")
    else:
        healing = 0.5 + (chems["serotonin"] * 0.02)
        chems["sanity"] = max(0.0, min(100.0, chems["sanity"] + healing))

    chems["dopamine"] += (50.0 - chems["dopamine"]) * 0.08
    chems["serotonin"] += (50.0 - chems["serotonin"]) * 0.08
    chems["acetylcholine"] += (50.0 - chems["acetylcholine"]) * 0.08

    # Burnout Check: Sanity is 0
    if chems["sanity"] <= 0.0:
        st.session_state.stats["burnout_count"] += 1
        st.session_state.playing = False
        chems["sanity"] = 25.0
        chems["stress"] = 10.0
        chems["energy"] = 50.0
        chems["dopamine"] = 20.0
        chems["serotonin"] = 30.0

        degraded = 0
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if grid[r][c]["type"] == "Interneuron" and random.random() < 0.4:
                    grid[r][c] = {"type": "Empty", "charge": 0.0, "threshold": 0.5, "fire_rate": 0.0, "last_fired": -1, "direction": "All"}
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

    # Record history for plot
    record_history(ticks, chems)

def record_history(ticks, chems):
    hist = st.session_state.history_data
    hist["tick"].append(ticks)
    hist["sanity"].append(chems["sanity"])
    hist["energy"].append(chems["energy"])
    hist["dopamine"].append(chems["dopamine"])
    hist["stress"].append(chems["stress"])

    if len(hist["tick"]) > 40:
        for key in hist:
            hist[key] = hist[key][-40:]

# Start the Telegram background bot thread if not already running
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
</style>
""", unsafe_allow_html=True)

st.title("🧠 Siêu Hệ Thống VBot1 & Game Mô Phỏng Não Bộ")
st.write("Dự án tích hợp: Game mô phỏng tiến hóa nơ-ron sinh học kết hợp Trợ lý AI Telegram Llama 3 & Gemini 1.5.")

tab1, tab2 = st.tabs(["🧠 Game Mô Phỏng Não Bộ", "🤖 Trợ Lý AI VBot1 (Llama & Gemini)"])

# Initialize game session state
init_game_state()

# ----------------- TAB 1: BRAIN GAME -----------------
with tab1:
    st.subheader("Trình Mô Phỏng Mạng Lưới Nơ-ron và Tiến Hóa Hóa Học Não")

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
        st.metric("Mức năng lượng (Fuel)", f"{st.session_state.chemicals['energy']:.1f}%")
    with cols[5]:
        st.metric("Số lần quá tải (Burnout)", st.session_state.stats["burnout_count"])

    # Progress bars for detailed chemistry
    chem_cols = st.columns(4)
    with chem_cols[0]:
        val = st.session_state.chemicals["dopamine"]
        st.progress(val / 100.0, text=f"Dopamine (Động lực, Thưởng): {val:.1f}%")
    with chem_cols[1]:
        val = st.session_state.chemicals["serotonin"]
        st.progress(val / 100.0, text=f"Serotonin (Ổn định cảm xúc): {val:.1f}%")
    with chem_cols[2]:
        val = st.session_state.chemicals["acetylcholine"]
        st.progress(val / 100.0, text=f"Acetylcholine (Sự tập trung): {val:.1f}%")
    with chem_cols[3]:
        val = st.session_state.chemicals["stress"]
        st.progress(val / 100.0, text=f"Căng thẳng (Stress Level): {val:.1f}%")

    # Hormone active abilities layout
    st.markdown("##### 🧪 Trung tâm nội tiết tố (Active Hormone Abilities)")
    hormone_cols = st.columns(3)
    cooldowns = st.session_state.cooldowns

    with hormone_cols[0]:
        doping_disabled = cooldowns["doping"] > 0
        btn_label_doping = f"⚡ Doping Dopamine ({cooldowns['doping']}s)" if doping_disabled else "⚡ Doping Dopamine"
        if st.button(btn_label_doping, disabled=doping_disabled, use_container_width=True, help="Tự động sạc đầy tất cả Sensory cells, +30 Dopamine, +25 Stress. Cooldown 15s"):
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    if st.session_state.neuron_grid[r][c]["type"] == "Sensory":
                        st.session_state.neuron_grid[r][c]["charge"] = 1.0
            st.session_state.chemicals["dopamine"] = min(100.0, st.session_state.chemicals["dopamine"] + 30.0)
            st.session_state.chemicals["stress"] = min(100.0, st.session_state.chemicals["stress"] + 25.0)
            cooldowns["doping"] = 15
            add_log("⚡ HORMONE: Kích hoạt Doping Dopamine! Đồng loạt Sensory cells bùng nổ xung điện.")
            st.rerun()

    with hormone_cols[1]:
        ssri_disabled = cooldowns["ssri"] > 0
        btn_label_ssri = f"💊 Tái hấp thu Serotonin (SSRI) ({cooldowns['ssri']}s)" if ssri_disabled else "💊 Tái hấp thu Serotonin (SSRI)"
        if st.button(btn_label_ssri, disabled=ssri_disabled, use_container_width=True, help="Hạ 50% Stress, hồi phục 30 Tỉnh táo lập tức. Cooldown 25s"):
            st.session_state.chemicals["stress"] = max(0.0, st.session_state.chemicals["stress"] - 50.0)
            st.session_state.chemicals["sanity"] = min(100.0, st.session_state.chemicals["sanity"] + 30.0)
            cooldowns["ssri"] = 25
            add_log("💊 HORMONE: Kích hoạt liệu pháp Serotonin! Xoa dịu vỏ não, triệt tiêu căng thẳng.")
            st.rerun()

    with hormone_cols[2]:
        focus_disabled = cooldowns["focus"] > 0
        btn_label_focus = f"🧠 Tập trung cao độ ({cooldowns['focus']}s)" if focus_disabled else "🧠 Tập trung cao độ"
        if st.button(btn_label_focus, disabled=focus_disabled, use_container_width=True, help="Tăng Acetylcholine (+50) và nạp thêm +50 IQ. Cooldown 20s"):
            st.session_state.chemicals["acetylcholine"] = min(100.0, st.session_state.chemicals["acetylcholine"] + 50.0)
            st.session_state.stats["iq"] += 50.0
            cooldowns["focus"] = 20
            add_log("🧠 HORMONE: Kích hoạt Tập trung cao độ! Khóa chặt Acetylcholine, nâng cao nhận thức (+50 IQ).")
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

                label = f"{emoji}{sym}\n({charge:.2f})"
                is_selected = (r == selected_r and c == selected_c)
                border_style = "🔴 " if is_selected else ""

                if cell["charge"] >= threshold and ctype != "Empty":
                    btn_label = f"🔥 {label}"
                else:
                    btn_label = f"{border_style}{label}"

                if row_cols[c].button(btn_label, key=f"cell_{r}_{c}", use_container_width=True):
                    st.session_state.selected_cell = (r, c)
                    st.rerun()

        # UPGRADE: Save & Load Circuit Codes Panel
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
                    "direction": "All"
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
                    "direction": "All"
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
                    "direction": "All"
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
                    "direction": "All"
                }
                add_log(f"Xóa bỏ nơ-ron tại [{selected_r+1},{selected_c+1}] (Thu hồi +{refund} MB)")
                st.rerun()

        if current_cell["type"] != "Empty":
            st.write("---")
            st.markdown("**🧭 Định hướng sợi trục (Axon Direction Target):**")
            st.caption("Khống chế xung điện thế chỉ lan truyền tới mục tiêu theo hướng chỉ định.")

            cur_dir = current_cell.get("direction", "All")
            dirs_list = ["All", "Up", "Right", "Down", "Left"]
            dir_names = {
                "All": "🌐 Bốn phía (All Direction)",
                "Up": "⬆️ Phía trên (Up)",
                "Right": "➡️ Phía phải (Right)",
                "Down": "⬇️ Phía dưới (Down)",
                "Left": "⬅️ Phía trái (Left)"
            }

            selected_new_dir = st.selectbox(
                "Chọn hướng trục nơ-ron truyền tải:",
                dirs_list,
                index=dirs_list.index(cur_dir),
                format_func=lambda x: dir_names[x],
                key=f"dir_select_{selected_r}_{selected_c}"
            )
            if selected_new_dir != cur_dir and selected_new_dir in dir_names:
                grid[selected_r][selected_c]["direction"] = selected_new_dir
                add_log(f"Định hướng lại trục nơ-ron [{selected_r+1},{selected_c+1}] thành {dir_names[selected_new_dir]}")
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

        st.session_state.tick_speed = st.slider("Tốc độ mô phỏng (Giây/Tick)", min_value=0.2, max_value=2.0, value=1.0, step=0.1)

        st.markdown("---")
        st.markdown("##### 🛒 Nâng Cấp Thùy Não (Mở rộng cấu trúc nhận thức)")
        st.caption("Sử dụng điểm IQ tích lũy từ các hành động Motor thành công để tiến hóa các vùng não.")

        upgrades = st.session_state.upgrades

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

        # UPGRADE: Synaptic Pruning Upgrade Panel
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

        # UPGRADE: Prefrontal Cortex PFC Decision Maker Panel
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

# UPGRADE: Visual/Audio Feedback Synthesizer Element via Web Audio API
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
