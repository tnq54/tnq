import streamlit as st
import time
import os
import threading
import asyncio
import io
import logging
import random
import re
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

# PDF Text Extraction
def extract_pdf_text(file_bytes):
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text = "".join(page.extract_text() or "" for page in reader.pages)
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

# Telegram Bot Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Welcome to VBot1!\n"
        "- Chat with me (Llama 3).\n"
        "- Send a PDF to summarize (Gemini 1.5 Flash)."
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

        # Split long messages
        if len(summary) > 4000:
            for i in range(0, len(summary), 4000):
                await update.message.reply_text(summary[i:i+4000])
        else:
            await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status_msg.message_id, text=summary)

    except Exception as e:
        logger.error(f"Document Error: {e}")
        await update.message.reply_text(f"Error processing document: {e}")

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
            application.run_polling(stop_signals=None, close_loop=False)
        except NetworkError as e:
            logger.error(f"Network error during polling: {e}. Retrying in 10s...")
            time.sleep(10)
        except Exception as e:
            logger.error(f"Critical error during polling: {e}")
            time.sleep(10)

# Background Thread for Telegram Bot - Guarded so app.py can be imported for testing
if __name__ == "__main__":
    if "bot_thread" not in st.session_state:
        st.session_state.bot_thread = True
        thread = threading.Thread(target=run_bot, daemon=True)
        thread.start()


# ==========================================
#         AI COLONY SIMULATION CLASSES
# ==========================================

class Agent:
    def __init__(self, name, role, x, y):
        self.name = name
        self.role = role
        self.x = x
        self.y = y
        self.health = 100
        self.energy = 100
        self.wealth = 100
        self.inventory = {"wood": 0, "wheat": 0, "gold": 0, "tech": 0}
        self.last_action = "Spawned in the colony."

    def to_dict(self):
        return {
            "Name": self.name,
            "Role": self.role,
            "X": self.x,
            "Y": self.y,
            "Health": self.health,
            "Energy": self.energy,
            "Wealth": self.wealth,
            "Inventory": str(self.inventory),
            "Last Action": self.last_action
        }

class SimulationEngine:
    def __init__(self, width=6, height=6):
        self.width = width
        self.height = height
        self.tick = 0
        self.logs = ["Simulation initialized."]
        self.agents = [
            Agent("Alice", "Farmer", 0, 1),
            Agent("Bob", "Scientist", 4, 5),
            Agent("Charlie", "Explorer", 3, 2),
            Agent("Diana", "Warrior", 2, 2)
        ]
        # Initial locations of key colony landmarks
        self.map_grid = {
            (0, 1): "🌾 Wheat Field",
            (1, 4): "🌲 Forest",
            (3, 2): "💎 Gold Mine",
            (4, 5): "🏥 Tech Lab",
            (2, 2): "🏠 Colony Base",
            (5, 0): "🌋 Volcano"
        }
        # Metrics history for plotting
        self.history = {
            "tick": [0],
            "avg_health": [100.0],
            "avg_energy": [100.0],
            "avg_wealth": [100.0],
            "wood": [0],
            "wheat": [0],
            "gold": [0],
            "tech": [0]
        }

    def log(self, message):
        timestamp = f"[Tick {self.tick}]"
        self.logs.insert(0, f"{timestamp} {message}")
        if len(self.logs) > 150:
            self.logs.pop()

    def get_ai_decision(self, agent, client=None):
        if client:
            try:
                grid_desc = ", ".join(f"{v} at {k}" for k, v in self.map_grid.items())
                prompt = (
                    f"You are controlling the AI agent '{agent.name}' (Role: {agent.role}) in a 2D grid simulation (size 6x6).\n"
                    f"Current Stats: Health={agent.health}/100, Energy={agent.energy}/100, Wealth={agent.wealth}, Inventory={agent.inventory}, Position=({agent.x},{agent.y}).\n"
                    f"Colony landmarks: {grid_desc}.\n"
                    f"Select the NEXT single optimal action (e.g., 'Move to (2,2)', 'Harvest at (0,1)', 'Rest at (2,2)', 'Research at (4,5)').\n"
                    f"Provide your answer in exactly one brief sentence starting with the action keyword (e.g. 'Move to (2, 2) to recover health' or 'Harvest wheat at (0, 1) to earn wealth'). Do not write any other explanation."
                )
                completion = client.chat_completion(
                    model="meta-llama/Meta-Llama-3-8B-Instruct",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=40
                )
                decision = completion.choices[0].message.content.strip()
                if decision:
                    return decision
            except Exception as e:
                logger.error(f"Failed Llama AI decision for {agent.name}: {e}")

        # Fallback to high-fidelity rule-based heuristic AI decision
        if agent.energy < 25 or agent.health < 30:
            if agent.x == 2 and agent.y == 2:
                return "Rest at Colony Base (2,2) to recover health and energy."
            else:
                return "Move to (2, 2) to rest at Colony Base."

        if agent.role == "Farmer":
            if agent.x == 0 and agent.y == 1:
                return "Harvest wheat at Wheat Field (0,1)."
            else:
                return "Move to (0, 1) to harvest wheat."

        elif agent.role == "Scientist":
            if agent.x == 4 and agent.y == 5:
                return "Research at Tech Lab (4,5)."
            else:
                return "Move to (4, 5) to research tech."

        elif agent.role == "Explorer":
            target = (3, 2) if random.random() < 0.5 else (1, 4)
            if (agent.x, agent.y) == target:
                res_name = "gold" if target == (3, 2) else "wood"
                return f"Harvest {res_name} at location {target}."
            else:
                return f"Move to {target} to explore."

        else: # Warrior
            if (agent.x, agent.y) == (2, 2):
                target = random.choice([(1, 2), (2, 1), (3, 2), (2, 3)])
                return f"Move to {target} to patrol boundaries."
            else:
                return "Move to (2, 2) to guard Colony Base."

    def execute_action(self, agent, action_str):
        action_lower = action_str.lower()
        coords = re.findall(r'\(?\s*([0-5])\s*,\s*([0-5])\s*\)?', action_str)
        tx, ty = None, None

        if coords:
            tx, ty = int(coords[0][0]), int(coords[0][1])
        else:
            if "base" in action_lower or "rest" in action_lower:
                tx, ty = 2, 2
            elif "wheat" in action_lower or "farmer" in action_lower:
                tx, ty = 0, 1
            elif "forest" in action_lower or "wood" in action_lower:
                tx, ty = 1, 4
            elif "gold" in action_lower or "mine" in action_lower:
                tx, ty = 3, 2
            elif "lab" in action_lower or "research" in action_lower:
                tx, ty = 4, 5
            elif "volcano" in action_lower:
                tx, ty = 5, 0

        if tx is not None and ty is not None:
            if agent.x == tx and agent.y == ty:
                landmark = self.map_grid.get((tx, ty), "")

                if "Colony Base" in landmark or (tx == 2 and ty == 2):
                    agent.energy = min(100, agent.energy + 35)
                    agent.health = min(100, agent.health + 20)
                    agent.last_action = "Rested at Colony Base (+35 Energy, +20 Health)"
                    self.log(f"{agent.name} is resting at Colony Base.")

                elif "Wheat Field" in landmark or (tx == 0 and ty == 1):
                    agent.inventory["wheat"] += 1
                    agent.wealth += 15
                    agent.energy = max(0, agent.energy - 10)
                    agent.last_action = "Harvested wheat (+1 Wheat, +15 Wealth)"
                    self.log(f"{agent.name} harvested wheat at Wheat Field.")

                elif "Forest" in landmark or (tx == 1 and ty == 4):
                    agent.inventory["wood"] += 1
                    agent.wealth += 12
                    agent.energy = max(0, agent.energy - 8)
                    agent.last_action = "Gathered wood (+1 Wood, +12 Wealth)"
                    self.log(f"{agent.name} gathered wood in Forest.")

                elif "Gold Mine" in landmark or (tx == 3 and ty == 2):
                    agent.inventory["gold"] += 1
                    agent.wealth += 30
                    agent.energy = max(0, agent.energy - 15)
                    agent.last_action = "Mined gold (+1 Gold, +30 Wealth)"
                    self.log(f"{agent.name} mined gold at Gold Mine.")

                elif "Tech Lab" in landmark or (tx == 4 and ty == 5):
                    agent.inventory["tech"] += 1
                    agent.wealth += 20
                    agent.energy = max(0, agent.energy - 10)
                    agent.last_action = "Researched tech (+1 Tech, +20 Wealth)"
                    self.log(f"{agent.name} completed scientific research at Tech Lab.")

                elif "Volcano" in landmark or (tx == 5 and ty == 0):
                    agent.health = max(0, agent.health - 30)
                    agent.last_action = "Burned by Lava! (-30 Health)"
                    self.log(f"⚠️ {agent.name} is burning in Volcano lava!")

                else:
                    agent.energy = max(0, agent.energy - 5)
                    agent.last_action = f"Interacted at empty area ({tx}, {ty})"
            else:
                dx = 1 if tx > agent.x else (-1 if tx < agent.x else 0)
                dy = 1 if ty > agent.y else (-1 if ty < agent.y else 0)
                agent.x += dx
                agent.y += dy
                agent.energy = max(0, agent.energy - 6)
                agent.last_action = f"Moving towards ({tx}, {ty}) -> currently at ({agent.x}, {agent.y})"
        else:
            agent.energy = max(0, agent.energy - 5)
            agent.last_action = "Idled near base."

    def step(self, client=None):
        self.tick += 1

        for agent in self.agents:
            if agent.health <= 0:
                agent.health = 50
                agent.energy = 50
                agent.wealth = max(0, agent.wealth - 40)
                agent.x, agent.y = 2, 2
                agent.last_action = "Revived at Colony Base with partial energy and health."
                self.log(f"🏥 {agent.name} fainted and was revived at Colony Base.")
                continue

            agent.energy = max(0, agent.energy - 5)
            if agent.energy <= 0:
                agent.health = max(0, agent.health - 10)
                self.log(f"⚠️ {agent.name} is starving (0 energy)! Health is decreasing.")

            action_desc = self.get_ai_decision(agent, client)
            self.execute_action(agent, action_desc)

            if agent.x == 5 and agent.y == 0:
                agent.health = max(0, agent.health - 30)
                self.log(f"🌋 WARNING: {agent.name} walked into the Volcano hazard!")

        self.history["tick"].append(self.tick)

        avg_h = sum(a.health for a in self.agents) / len(self.agents)
        avg_e = sum(a.energy for a in self.agents) / len(self.agents)
        avg_w = sum(a.wealth for a in self.agents) / len(self.agents)

        self.history["avg_health"].append(avg_h)
        self.history["avg_energy"].append(avg_e)
        self.history["avg_wealth"].append(avg_w)

        self.history["wood"].append(sum(a.inventory["wood"] for a in self.agents))
        self.history["wheat"].append(sum(a.inventory["wheat"] for a in self.agents))
        self.history["gold"].append(sum(a.inventory["gold"] for a in self.agents))
        self.history["tech"].append(sum(a.inventory["tech"] for a in self.agents))

    def trigger_meteor_strike(self):
        mx = random.randint(0, self.width - 1)
        my = random.randint(0, self.height - 1)
        self.map_grid[(mx, my)] = "🔥 Meteor Ruins"
        self.log(f"☄️ METEOR STRIKE! A meteor crashed at ({mx}, {my}) turning it into ruins!")

        for agent in self.agents:
            dist = max(abs(agent.x - mx), abs(agent.y - my))
            if dist == 0:
                agent.health = max(0, agent.health - 60)
                self.log(f"💥 {agent.name} was hit directly by the meteor! (-60 Health)")
            elif dist == 1:
                agent.health = max(0, agent.health - 30)
                self.log(f"💨 {agent.name} was caught in the meteor shockwave! (-30 Health)")

    def trigger_gold_rush(self):
        gx = random.randint(0, self.width - 1)
        gy = random.randint(0, self.height - 1)
        if (gx, gy) not in self.map_grid:
            self.map_grid[(gx, gy)] = "💎 Gold Mine"
            self.log(f"💰 GOLD RUSH! A rich new gold vein spawned at ({gx}, {gy})!")
        else:
            self.log("💰 GOLD RUSH! Geologists reported minor findings but no new veins could form.")

    def trigger_pandemic(self):
        self.log("🦠 PANDEMIC! A toxic virus spreads through the dome colony!")
        for agent in self.agents:
            agent.health = max(5, agent.health - 25)
            self.log(f"🤮 {agent.name} is sick and lost 25 health.")

    def trigger_custom_event(self, event_text):
        self.log(f"✨ EVENT: {event_text}")
        event_lower = event_text.lower()

        if "heal" in event_lower or "cure" in event_lower:
            for agent in self.agents:
                agent.health = min(100, agent.health + 40)
                agent.energy = min(100, agent.energy + 20)
            self.log("💚 Status update: All agents received emergency medical treatment (+40 Health).")
        elif "gift" in event_lower or "wealth" in event_lower or "money" in event_lower or "gold" in event_lower:
            for agent in self.agents:
                agent.wealth += 100
            self.log("🪙 Status update: A wealthy benefactor gifted every agent 100 wealth.")
        elif "hazard" in event_lower or "curse" in event_lower or "earthquake" in event_lower:
            for agent in self.agents:
                agent.health = max(10, agent.health - 20)
            self.log("⚠️ Status update: Environmental tremors caused mild injuries (-20 Health).")
        else:
            self.log("📜 Narrative effect: The environment adapts, and agents remain alert.")


# ==========================================
#          STREAMLIT WEB INTERFACE
# ==========================================

st.set_page_config(
    page_title="VBot1 Hub & AI Colony Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Render tab headers
tabs = st.tabs(["🎮 AI Colony Simulator", "🤖 Telegram Bot Control & Summary Tools"])

# Initialize Simulation State in st.session_state
if "sim_engine" not in st.session_state:
    st.session_state.sim_engine = SimulationEngine()
if "sim_playing" not in st.session_state:
    st.session_state.sim_playing = False
if "sim_speed" not in st.session_state:
    st.session_state.sim_speed = 1.0


# ------------------------------------------
#     TAB 1: AI COLONY SIMULATOR
# ------------------------------------------
with tabs[0]:
    st.title("🎮 AI Colony Simulator")
    st.write(
        "Welcome to the **AI Colony Simulator**! This is a simulated dome colony where autonomous AI agents live, "
        "interact with landmarks, manage survival resources, and make decisions dynamically. "
        "The agents make decisions using **Llama 3** (if HF_TOKEN is configured) or a robust rules-based heuristic model."
    )

    engine = st.session_state.sim_engine

    # Balanced Layout with 2 Columns: left is sidebar controls, right is map grid.
    # Logs, telemetry table, and analytics are moved to full-width lower sections to prevent any overlap!
    col_ctrl, col_map = st.columns([1.5, 3.0])

    with col_ctrl:
        st.subheader("⚙️ Controls")

        # Play/Pause & Speed
        play_label = "⏸️ Pause Simulation" if st.session_state.sim_playing else "▶️ Play Simulation"
        if st.button(play_label, use_container_width=True):
            st.session_state.sim_playing = not st.session_state.sim_playing
            st.rerun()

        if st.button("叙 Step 1 Tick", use_container_width=True, disabled=st.session_state.sim_playing):
            engine.step(hf_client)
            st.rerun()

        if st.button("🔄 Reset Colony", use_container_width=True):
            st.session_state.sim_engine = SimulationEngine()
            st.session_state.sim_playing = False
            st.rerun()

        st.session_state.sim_speed = st.slider(
            "Sim Speed (seconds/tick)",
            min_value=0.5,
            max_value=3.0,
            value=st.session_state.sim_speed,
            step=0.5
        )

        st.divider()
        st.subheader("💥 Trigger Events")

        if st.button("☄️ Meteor Strike", use_container_width=True):
            engine.trigger_meteor_strike()
            st.toast("Meteor launched!")
            st.rerun()

        if st.button("💰 Gold Rush", use_container_width=True):
            engine.trigger_gold_rush()
            st.toast("Gold rush started!")
            st.rerun()

        if st.button("🦠 Pandemic Virus", use_container_width=True):
            engine.trigger_pandemic()
            st.toast("Pandemic declared!")
            st.rerun()

        st.write("**Custom Scenario Event**")
        custom_event_text = st.text_input("Type an event (e.g. 'A medical crate lands' or 'Gold flood')", key="custom_evt_input")
        if st.button("🚀 Execute Scenario", use_container_width=True) and custom_event_text:
            engine.trigger_custom_event(custom_event_text)
            st.success(f"Dispatched custom event: {custom_event_text}")
            st.rerun()

    with col_map:
        st.subheader(f"🗺️ Colony Map (Tick: {engine.tick})")

        grid_size = engine.width
        cells = [["" for _ in range(grid_size)] for _ in range(grid_size)]

        # Populate with landmarks, retrieving the FIRST character (the Emoji) instead of last word
        for (x, y), val in engine.map_grid.items():
            cells[x][y] = val.split()[0] # e.g. "🌾"

        # Overlay Agent positions
        for a in engine.agents:
            emoji_map = {"Farmer": "👨‍🌾", "Scientist": "🔬", "Explorer": "🧭", "Warrior": "🛡️"}
            char_emoji = emoji_map.get(a.role, "👤")
            if cells[a.x][a.y]:
                # If there's already an emoji in that cell, append the agent emoji cleanly
                if len(cells[a.x][a.y]) <= 2:
                    cells[a.x][a.y] = cells[a.x][a.y] + char_emoji
                else:
                    cells[a.x][a.y] += char_emoji
            else:
                cells[a.x][a.y] = char_emoji

        # Build clean HTML table with table-layout fixed to avoid any stretching or horizontal drift!
        map_html = (
            "<div style='display: flex; justify-content: flex-start; align-items: center; margin-bottom: 15px;'>"
            "<table style='table-layout: fixed; width: 360px; border-collapse: collapse; text-align: center; font-size: 24px; font-weight: bold;'>"
        )
        for r in range(grid_size):
            map_html += "<tr style='height: 60px;'>"
            for c in range(grid_size):
                content = cells[r][c] or "⬜"
                bg_color = "#15181c" if (r + c) % 2 == 0 else "#23282f"
                # Volcano hazard highlight
                if r == 5 and c == 0:
                    bg_color = "#451c1c"
                # Meteor Ruins highlight
                elif engine.map_grid.get((r, c), "") == "🔥 Meteor Ruins":
                    bg_color = "#352912"

                map_html += (
                    f"<td style='width: 60px; height: 60px; border: 2px solid #3c444d; background-color: {bg_color}; "
                    f"padding: 0px; text-align: center; vertical-align: middle; overflow: hidden; white-space: nowrap;'>{content}</td>"
                )
            map_html += "</tr>"
        map_html += "</table></div>"

        st.markdown(map_html, unsafe_allow_html=True)

        st.write("💡 **Map Legend:**")
        st.write(
            "🌾 Wheat Field | 🌲 Forest | 💎 Gold Mine | 🏥 Tech Lab | "
            "🏠 Colony Base (Safe resting) | 🌋 Volcano (Hazard) | 🔥 Meteor Ruins (Hazard)"
        )
        st.write(
            "👨‍🌾 Alice (Farmer) | 🔬 Bob (Scientist) | 🧭 Charlie (Explorer) | 🛡️ Diana (Warrior)"
        )

    # Lower section: telemetry & logs side-by-side below the map to prevent overlap completely!
    st.divider()
    col_tel, col_log = st.columns([1.0, 1.0])

    with col_tel:
        st.subheader("📊 Colony Telemetry")
        df_agents = pd.DataFrame([a.to_dict() for a in engine.agents])
        st.dataframe(df_agents, hide_index=True, use_container_width=True)

    with col_log:
        st.subheader("📜 Simulation Console Logs")
        logs_text = "\n".join(engine.logs)
        st.text_area("Console Output", value=logs_text, height=180, disabled=True, label_visibility="collapsed")

    # Bottom section: Interactive analytics and real-time line charts
    st.divider()
    st.subheader("📈 Real-time Colony Growth Analytics")
    col_chart_h, col_chart_w = st.columns(2)

    hist = engine.history
    chart_df = pd.DataFrame({
        "Tick": hist["tick"],
        "Avg Health": hist["avg_health"],
        "Avg Energy": hist["avg_energy"],
        "Avg Wealth": hist["avg_wealth"]
    }).set_index("Tick")

    res_df = pd.DataFrame({
        "Tick": hist["tick"],
        "Wood Collected": hist["wood"],
        "Wheat Harvested": hist["wheat"],
        "Gold Extracted": hist["gold"],
        "Tech Discovered": hist["tech"]
    }).set_index("Tick")

    with col_chart_h:
        st.write("**Average Agent Vitality Metrics**")
        st.line_chart(chart_df)

    with col_chart_w:
        st.write("**Cumulative Resource Production Counts**")
        st.line_chart(res_df)

    # Autoplay loops: if sim_playing is set, wait sim_speed and trigger refresh
    if st.session_state.sim_playing:
        time.sleep(st.session_state.sim_speed)
        engine.step(hf_client)
        st.rerun()


# ------------------------------------------
#     TAB 2: ORIGINAL BOT STATUS & CONTROL
# ------------------------------------------
with tabs[1]:
    st.title("🤖 Telegram Bot Control & Summary Tools")
    st.write("Keep track of background task threads, verify API settings, and interact with the PDF text summary pipeline.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("System Configuration")
        st.write(f"- **Telegram Bot Thread**: `Running` (Polling Telegram API)")
        st.write(f"- **Llama 3 Status (Hugging Face)**: {'🟢 Active' if hf_client else '🔴 Inactive (HF_TOKEN missing)'}")
        st.write(f"- **Gemini 1.5 Flash Status**: {'🟢 Active' if GOOGLE_API_KEY else '🔴 Inactive (GOOGLE_API_KEY missing)'}")

    with col2:
        st.subheader("PDF Summarizer Testing Tool")
        uploaded_file = st.file_uploader("Upload a PDF to summarize with Gemini", type=["pdf"])
        if uploaded_file is not None:
            if st.button("Summarize Document"):
                with st.spinner("Extracting & summarizing document content..."):
                    bytes_data = uploaded_file.read()
                    extracted_text = extract_pdf_text(bytes_data)
                    if extracted_text:
                        summary = summarize_with_gemini(extracted_text)
                        st.subheader("Gemini 1.5 Flash Summary Result")
                        st.write(summary)
                    else:
                        st.error("Could not extract text from the selected PDF.")
