import streamlit as st
import time
import os
import threading
import asyncio
import io
import logging
import shlex
import subprocess
from pypdf import PdfReader
from telegram import Update
from telegram.error import NetworkError
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from huggingface_hub import InferenceClient

# Set Page Config MUST be the very first Streamlit call
st.set_page_config(
    page_title="HuggingFace Lab Workspace",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Module level global workflow config
GLOBAL_WORKFLOW_CONFIG = {
    "active_model": "openbmb/MiniCPM-Llama3-V-2_5",
    "status": "ready"
}

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

# Secure Shell Command Execution
def execute_shell_command(command_str):
    if not command_str or not command_str.strip():
        return ""
    cmd = command_str.strip()
    if cmd in ["clear", "cls"]:
        return "__CLEAR__"

    # Block shell operators / command chaining
    disallowed_chars = [";", "&&", "||", "|", "`", "$", "(", ")", "<", ">", "&"]
    if any(char in cmd for char in disallowed_chars):
        return "Error: Command chaining or special shell operators are restricted for security."

    try:
        parts = shlex.split(cmd)
    except Exception as e:
        return f"Command parsing error: {e}"

    if not parts:
        return ""

    allowed_commands = ["ls", "cat", "pwd", "echo", "mkdir", "python", "pytest", "git", "pip", "whoami", "uname", "date", "touch"]
    first_word = parts[0]
    if first_word not in allowed_commands:
        return f"Command '{first_word}' is restricted for security. Allowed: {', '.join(allowed_commands)}"

    try:
        if first_word == "mkdir":
            target_dir = parts[1] if len(parts) > 1 else "new_folder"
            os.makedirs(target_dir, exist_ok=True)
            return f"Directory created: {target_dir}"

        result = subprocess.run(parts, capture_output=True, text=True, timeout=15)
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        return output if output else "(No output)"
    except Exception as e:
        return f"Execution error: {e}"

# Telegram Bot Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Welcome to VBot1 HF Lab Workspace!\n"
        "- Chat with me (Llama 3 / MiniCPM5).\n"
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
        model_name = GLOBAL_WORKFLOW_CONFIG.get("active_model", "meta-llama/Meta-Llama-3-8B-Instruct")
        completion = hf_client.chat_completion(
            model=model_name,
            messages=messages,
            max_tokens=500
        )
        reply = completion.choices[0].message.content
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status_msg.message_id, text=reply)
    except Exception as e:
        logger.error(f"HF Inference Error: {e}")
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

# Bot Runner
def run_bot():
    logger.info("Waiting 10s for network initialization...")
    time.sleep(10)

    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN is missing")
        return

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    logger.info("Starting Telegram bot polling loop...")

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

# Start background thread
if "bot_thread_started" not in st.session_state:
    st.session_state.bot_thread_started = True
    thread = threading.Thread(target=run_bot, daemon=True)
    thread.start()

# Initialize session state for selected model
if "active_model" not in st.session_state:
    st.session_state.active_model = "openbmb/MiniCPM-Llama3-V-2_5"

# --- Custom Styling with High Contrast ---
st.markdown("""
<style>
    .stApp {
        background-color: #13131a;
        color: #f3f4f6;
    }
    .top-bar {
        background-color: #1c1c24;
        padding: 12px 20px;
        border-radius: 8px;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border: 1px solid #2e2e38;
    }
    .badge-webgpu {
        background-color: #10b981;
        color: #000000;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .mascot-box {
        background-color: #1c1c24;
        border: 1px solid #3b82f6;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 15px;
    }
    div[data-baseweb="tab-list"] button {
        color: #f3f4f6 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stChatMessage"] {
        color: #f3f4f6 !important;
        background-color: #1c1c24 !important;
        border-radius: 8px;
        border: 1px solid #2e2e38;
        margin-bottom: 8px;
    }
    div[data-testid="stChatMessage"] p {
        color: #f3f4f6 !important;
    }
</style>
""", unsafe_allow_html=True)

# Top Bar Header
top_col1, top_col2, top_col3 = st.columns([3, 2, 2])

with top_col1:
    st.title("🤖 HuggingFace Lab Workspace")

with top_col2:
    st.markdown("""
    <div style="padding-top: 15px;">
        <span class="badge-webgpu">⚡ WebGPU Active</span>
        <span style="margin-left: 10px; color: #e5e7eb; font-size: 0.95rem; font-weight: 500;">Status: Online</span>
    </div>
    """, unsafe_allow_html=True)

with top_col3:
    model_choice = st.selectbox(
        "Active Model:",
        ["MiniCPM5-2B", "Meta-Llama-3-8B-Instruct", "Gemini-1.5-Flash"],
        index=0
    )
    if model_choice == "MiniCPM5-2B":
        st.session_state.active_model = "openbmb/MiniCPM-Llama3-V-2_5"
        GLOBAL_WORKFLOW_CONFIG["active_model"] = "openbmb/MiniCPM-Llama3-V-2_5"
    elif model_choice == "Meta-Llama-3-8B-Instruct":
        st.session_state.active_model = "meta-llama/Meta-Llama-3-8B-Instruct"
        GLOBAL_WORKFLOW_CONFIG["active_model"] = "meta-llama/Meta-Llama-3-8B-Instruct"

st.divider()

# Workspace Main Columns
col_left, col_right = st.columns([1, 1])

# --- LEFT COLUMN: Pi TUI Chat Studio ---
with col_left:
    st.subheader("💬 Pi TUI Chat Studio")

    # Pixel Cat Mascot Callout
    st.markdown("""
    <div class="mascot-box">
        <div style="font-size: 1.2rem; font-weight: bold; color: #60a5fa;">🐱 Pi Mascot</div>
        <div style="font-size: 0.95rem; color: #f3f4f6;">"Meow! Welcome to the HF Lab Studio. Type <code>/help</code> or ask me anything!"</div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your HF Lab Assistant. How can I help you today?"}
        ]

    # Display Chat History
    chat_container = st.container(height=380)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Chat Input
    if user_input := st.chat_input("Type a message or slash command (/help, /clear, /status, /models)..."):
        # Append User Message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Slash command handling
        if user_input.strip() == "/help":
            reply = "Available Commands:\n- `/help`: Show command list\n- `/clear`: Clear chat history\n- `/status`: Show system & API status\n- `/models`: Show available models"
        elif user_input.strip() == "/clear":
            st.session_state.messages = []
            reply = "Chat history cleared!"
        elif user_input.strip() == "/status":
            reply = f"System Status:\n- HF Token: {'Present' if HF_TOKEN else 'Missing'}\n- Telegram Token: {'Present' if TELEGRAM_TOKEN else 'Missing'}\n- Google API Key: {'Present' if GOOGLE_API_KEY else 'Missing'}\n- Active Model: {st.session_state.active_model}"
        elif user_input.strip() == "/models":
            reply = "Models:\n1. MiniCPM5-2B (`openbmb/MiniCPM-Llama3-V-2_5`)\n2. Meta-Llama-3-8B-Instruct (`meta-llama/Meta-Llama-3-8B-Instruct`)\n3. Gemini 1.5 Flash (`gemini-1.5-flash`)"
        else:
            # Generate AI Reply
            if hf_client:
                try:
                    completion = hf_client.chat_completion(
                        model=st.session_state.active_model,
                        messages=[{"role": "user", "content": user_input}],
                        max_tokens=400
                    )
                    reply = completion.choices[0].message.content
                except Exception as e:
                    reply = f"HF API response: Hello! Received your request: '{user_input}'. (Notice: {e})"
            else:
                reply = f"Echo Response: '{user_input}'. (Note: Add HF_TOKEN to enable Llama 3 / MiniCPM)"

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

# --- RIGHT COLUMN: Workspace Split (File Manager/Editor + Shell Terminal) ---
with col_right:
    st.subheader("🛠️ Workspace Tools")

    tab_files, tab_terminal = st.tabs(["📂 File Manager / Editor", "🖥️ Interactive Shell Terminal"])

    with tab_files:
        st.write("Browse and edit project files directly:")
        files = [f for f in os.listdir(".") if os.path.isfile(f)]
        selected_file = st.selectbox("Select File to View/Edit:", files, index=files.index("README.md") if "README.md" in files else 0)

        if selected_file:
            try:
                with open(selected_file, "r", encoding="utf-8") as f:
                    file_content = f.read()
            except Exception as e:
                file_content = f"Error reading file: {e}"

            updated_content = st.text_area(f"Editing `{selected_file}`:", value=file_content, height=250)

            if st.button(f"💾 Save {selected_file}"):
                try:
                    with open(selected_file, "w", encoding="utf-8") as f:
                        f.write(updated_content)
                    st.success(f"Saved changes to `{selected_file}` successfully!")
                except Exception as e:
                    st.error(f"Failed to save file: {e}")

    with tab_terminal:
        st.write("Execute shell commands in the workspace sandbox:")

        if "terminal_history" not in st.session_state:
            st.session_state.terminal_history = []

        term_cmd = st.text_input("Terminal Shell Prompt ($):", placeholder="e.g. ls -la, pwd, echo Hello")

        if st.button("Run Command") or (term_cmd and st.session_state.get("last_cmd") != term_cmd):
            st.session_state.last_cmd = term_cmd
            output = execute_shell_command(term_cmd)
            if output == "__CLEAR__":
                st.session_state.terminal_history = []
            else:
                st.session_state.terminal_history.append(f"$ {term_cmd}\n{output}")

        if st.button("Clear Terminal Log"):
            st.session_state.terminal_history = []
            st.rerun()

        # Render Terminal Output
        history_text = "\n\n".join(st.session_state.terminal_history) if st.session_state.terminal_history else "Terminal initialized. Type a command above."
        st.text_area("Terminal Output:", value=history_text, height=220, disabled=True)
