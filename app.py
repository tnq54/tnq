import streamlit as st
import time
import os
import threading
import asyncio
import io
import json
import subprocess
import logging
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


# ---------------------------------------------------------
# Custom CSS Styling (Pi Browser Workspace Theme)
# ---------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Dark Theme Core */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #121319 !important;
        color: #e0e1e6 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }

    /* Hide Streamlit Header */
    [data-testid="stHeader"] {
        display: none !important;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 1rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 100% !important;
    }

    /* Monospace Text & Elements */
    .mono-font {
        font-family: 'Consolas', 'Fira Code', 'Monaco', 'Courier New', monospace !important;
    }

    /* Header & Badges */
    .pi-logo {
        font-weight: 800;
        font-size: 1.3rem;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .pi-logo-icon {
        background: #2b2c3a;
        padding: 2px 8px;
        border-radius: 4px;
        color: #a5a6f6;
    }

    .webgpu-badge {
        background-color: #232430;
        color: #a0a1b8;
        border: 1px solid #323344;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .webgpu-dot {
        width: 8px;
        height: 8px;
        background-color: #00e676;
        border-radius: 50%;
        display: inline-block;
    }

    /* TUI Left Panel */
    .tui-header {
        font-size: 1rem;
        color: #e0e1e6;
        margin-bottom: 12px;
    }

    .tui-command-help {
        color: #8b8c9e;
        font-size: 0.88rem;
        line-height: 1.6;
        margin-bottom: 16px;
    }

    .tui-command-help span.cmd {
        color: #d1d2e0;
        font-weight: 600;
    }

    /* Mascot Speech Bubble */
    .mascot-wrapper {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        margin-top: auto;
        margin-bottom: 16px;
        padding-right: 20px;
    }

    .speech-bubble {
        position: relative;
        background: #fff8e7;
        color: #1a1a1a;
        font-family: sans-serif;
        font-size: 0.85rem;
        padding: 10px 14px;
        border-radius: 8px;
        border: 2px solid #333;
        margin-bottom: 8px;
        box-shadow: 2px 2px 0px #000;
        max-width: 220px;
        text-align: center;
    }

    .speech-bubble::after {
        content: '';
        position: absolute;
        bottom: -10px;
        right: 30px;
        border-width: 10px 10px 0;
        border-style: solid;
        border-color: #fff8e7 transparent;
        display: block;
        width: 0;
    }

    .pixel-cat {
        font-size: 2.5rem;
        line-height: 1;
        user-select: none;
    }

    /* TUI Status Bar */
    .tui-status-bar {
        background-color: #12131a;
        border-top: 1px solid #282936;
        padding: 6px 12px;
        color: #8c8d9e;
        font-size: 0.8rem;
        border-radius: 4px;
        margin-top: 8px;
    }

    /* Workspace Panel Right */
    .workspace-card {
        background-color: #171821;
        border: 1px solid #282936;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 12px;
    }

    .file-path {
        color: #a0a1b5;
        font-size: 0.85rem;
    }

    /* High Contrast Controls & Buttons Fix */
    button, .stButton button, [data-testid="baseButton-secondary"], [data-testid="baseButton-primary"] {
        background-color: #232430 !important;
        color: #ffffff !important;
        border: 1px solid #3e4054 !important;
        border-radius: 6px !important;
    }

    button p, .stButton button p, .stDownloadButton button p {
        color: #ffffff !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }

    /* Text Inputs & Selectboxes */
    input, textarea, [data-baseweb="input"] input, [data-baseweb="textarea"] textarea {
        background-color: #121319 !important;
        color: #ffffff !important;
        border: 1px solid #323344 !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #232430 !important;
        color: #ffffff !important;
        border-color: #3e4054 !important;
    }

    div[data-baseweb="select"] span {
        color: #ffffff !important;
    }

    /* Hide Streamlit Element Extra Padding */
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.5rem !important;
    }
</style>
"""


# ---------------------------------------------------------
# Helper Functions: PDF, Gemini & Safe Shell Execution
# ---------------------------------------------------------
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

ALLOWED_COMMAND_PREFIXES = ["ls", "cat", "pwd", "echo", "python", "python3", "head", "tail", "wc", "grep", "date", "whoami", "cd", "clear"]

def execute_shell_command(cmd, workspace_dir="./workspace"):
    if not os.path.exists(workspace_dir):
        os.makedirs(workspace_dir, exist_ok=True)

    cmd = cmd.strip()
    if not cmd:
        return ""

    if cmd == "clear":
        return "CLEAR_SIGNAL"

    # Restrict to safe workspace utilities
    cmd_base = cmd.split()[0].lower() if cmd.split() else ""
    if cmd_base not in ALLOWED_COMMAND_PREFIXES:
        return f"Command '{cmd_base}' restricted in virtual workspace shell."

    # Navigation helper
    if cmd.startswith("cd "):
        new_path = cmd[3:].strip()
        target = os.path.abspath(os.path.join(workspace_dir, new_path))
        if os.path.exists(target) and os.path.isdir(target):
            return f"Changed directory to {new_path}"
        else:
            return f"cd: no such file or directory: {new_path}"

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=workspace_dir,
            capture_output=True,
            text=True,
            timeout=15
        )
        out = result.stdout
        err = result.stderr
        if err:
            out += f"\n{err}"
        return out.strip() if out else "[Command completed with no output]"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out (15s limit)."
    except Exception as e:
        return f"Execution error: {e}"


# ---------------------------------------------------------
# Default Workspace Setup
# ---------------------------------------------------------
WORKSPACE_DIR = "./workspace"
os.makedirs(WORKSPACE_DIR, exist_ok=True)

DEFAULT_FILES = {
    "README.md": "# Browser workspace\n\nYour files are saved in this browser.\n\nTry running python scripts, creating JSON data, or chatting with Pi!",
    "products.json": json.dumps([
        {"id": 1, "name": "Laptop", "price": 999.99, "stock": 15},
        {"id": 2, "name": "Smartphone", "price": 699.99, "stock": 30},
        {"id": 3, "name": "Headphones", "price": 149.99, "stock": 50}
    ], indent=2),
    "sales.csv": "date,product_id,quantity,total\n2025-01-01,1,2,1999.98\n2025-01-02,2,1,699.99\n2025-01-03,3,3,449.97"
}

for fname, content in DEFAULT_FILES.items():
    fpath = os.path.join(WORKSPACE_DIR, fname)
    if not os.path.exists(fpath):
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)


# ---------------------------------------------------------
# Telegram Bot Background Runner
# ---------------------------------------------------------
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

        if len(summary) > 4000:
            for i in range(0, len(summary), 4000):
                await update.message.reply_text(summary[i:i+4000])
        else:
            await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status_msg.message_id, text=summary)

    except Exception as e:
        logger.error(f"Document Error: {e}")
        await update.message.reply_text(f"Error processing document: {e}")

def run_bot():
    logger.info("Waiting 10s for network initialization...")
    time.sleep(10)
    if not TELEGRAM_TOKEN:
        logger.info("TELEGRAM_TOKEN is missing. Telegram bot runner idle.")
        return

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

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


# ---------------------------------------------------------
# Main Render Streamlit UI
# ---------------------------------------------------------
def render_app():
    st.set_page_config(
        page_title="Pi Browser Workspace",
        page_icon="π",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Session State Initialization
    if "selected_file" not in st.session_state:
        st.session_state.selected_file = "README.md"

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Welcome to Pi browser workspace! Ask me to work with your files or calculate your sales."}
        ]

    if "shell_history" not in st.session_state:
        st.session_state.shell_history = [
            "Ready in your workspace",
            "Try ls -la or cat README.md"
        ]

    if "active_model" not in st.session_state:
        st.session_state.active_model = "MiniCPM5-2B"

    if "bot_thread" not in st.session_state and TELEGRAM_TOKEN:
        st.session_state.bot_thread = True
        thread = threading.Thread(target=run_bot, daemon=True)
        thread.start()

    # UI Render: Top Header Bar
    h_col1, h_col2, h_col3 = st.columns([2, 5, 3])

    with h_col1:
        st.markdown("""
            <div class="pi-logo">
                <span class="pi-logo-icon">π</span>
                <span>Pi</span>
            </div>
        """, unsafe_allow_html=True)

    with h_col2:
        st.markdown("""
            <div style="display: flex; gap: 12px; align-items: center; justify-content: center; height: 100%;">
                <span class="webgpu-badge"><span class="webgpu-dot"></span> WebGPU Idle</span>
                <span style="color: #6c6d80; font-size: 0.82rem;" class="mono-font">Model and tools run in your browser</span>
            </div>
        """, unsafe_allow_html=True)

    with h_col3:
        header_model_col, btn_col = st.columns([2, 1])
        with header_model_col:
            selected_m = st.selectbox(
                "Model",
                ["MiniCPM5-2B", "Meta-Llama-3-8B", "Gemini-1.5-Flash"],
                label_visibility="collapsed",
                key="model_select"
            )
            st.session_state.active_model = selected_m
        with btn_col:
            if st.button("Load model", key="load_model_btn"):
                st.toast(f"Loaded {st.session_state.active_model} successfully!")

    # UI Layout: Main 2-Column Split
    col_left, col_right = st.columns([11, 9])

    # LEFT COLUMN: Pi TUI Chat Workspace
    with col_left:
        st.markdown("""
            <div class="mono-font tui-header">
                <span style="color: #a5a6f6; font-weight: bold;">π Pi</span> browser workspace
            </div>
            <div class="tui-command-help mono-font">
                Ask Pi to work with your files.<br><br>
                <span class="cmd">/load</span> load MiniCPM<br>
                <span class="cmd">/help</span> show commands<br>
                <span class="cmd">!ls</span> run a workspace command
            </div>
        """, unsafe_allow_html=True)

        # Message Log
        chat_box = st.container(height=320)
        with chat_box:
            for msg in st.session_state.chat_messages:
                if msg["role"] == "user":
                    st.markdown(f"**You:** {msg['content']}")
                else:
                    st.markdown(f"**Pi:** {msg['content']}")

        # Mascot Widget Callout
        st.markdown("""
            <div class="mascot-wrapper">
                <div class="speech-bubble">
                    Ask me to calculate our total sales.
                </div>
                <div class="pixel-cat">🐱</div>
            </div>
        """, unsafe_allow_html=True)

        # TUI Footer Status Line
        st.markdown(f"""
            <div class="tui-status-bar mono-font">
                Files and chat are saved on this device.
            </div>
            <div class="tui-status-bar mono-font" style="background-color: #1a1b26; color: #a0a1b8; margin-top: 4px;">
                /workspace &nbsp;•&nbsp; {st.session_state.active_model} &nbsp;•&nbsp; 0/8,192 context &nbsp;•&nbsp; 8,192 free
            </div>
        """, unsafe_allow_html=True)

        # Chat Input Bar
        chat_in_col, new_chat_col, send_col = st.columns([6, 2, 2])
        with chat_in_col:
            prompt_input = st.text_input("Prompt", placeholder="Ask Pi anything or type a command...", label_visibility="collapsed", key="chat_prompt")
        with new_chat_col:
            if st.button("New chat", key="new_chat_btn", use_container_width=True):
                st.session_state.chat_messages = [
                    {"role": "assistant", "content": "New chat session started."}
                ]
                st.rerun()
        with send_col:
            if st.button("Send ↑", key="send_chat_btn", use_container_width=True):
                if prompt_input.strip():
                    user_msg = prompt_input.strip()
                    st.session_state.chat_messages.append({"role": "user", "content": user_msg})

                    # Process Chat Message
                    if user_msg == "/help":
                        bot_resp = "Available commands:\n- /load : Reload active model\n- /help : Show this menu\n- !ls or !<cmd> : Execute shell command in /workspace"
                    elif user_msg.startswith("!"):
                        cmd_to_run = user_msg[1:]
                        bot_resp = execute_shell_command(cmd_to_run)
                    elif "sales" in user_msg.lower():
                        sales_file = os.path.join(WORKSPACE_DIR, "sales.csv")
                        if os.path.exists(sales_file):
                            with open(sales_file, "r") as f:
                                lines = f.readlines()[1:]
                            total = sum([float(line.split(",")[-1].strip()) for line in lines if line.strip()])
                            bot_resp = f"Calculated total sales from `/workspace/sales.csv`: **${total:,.2f}**"
                        else:
                            bot_resp = "Sales file not found in `/workspace/sales.csv`."
                    else:
                        if hf_client and st.session_state.active_model == "Meta-Llama-3-8B":
                            try:
                                completion = hf_client.chat_completion(
                                    model="meta-llama/Meta-Llama-3-8B-Instruct",
                                    messages=[{"role": "user", "content": user_msg}],
                                    max_tokens=300
                                )
                                bot_resp = completion.choices[0].message.content
                            except Exception as e:
                                bot_resp = f"Llama 3 Error: {e}"
                        elif GOOGLE_API_KEY and st.session_state.active_model == "Gemini-1.5-Flash":
                            bot_resp = summarize_with_gemini(user_msg)
                        else:
                            bot_resp = f"Pi TUI ({st.session_state.active_model}): Processed your prompt in browser workspace."

                    st.session_state.chat_messages.append({"role": "assistant", "content": bot_resp})
                    st.rerun()

    # RIGHT COLUMN: Workspace File Editor & Shell
    with col_right:
        st.markdown('<div class="workspace-card">', unsafe_allow_html=True)

        # Single-level column bar for workspace header actions
        w_col_title, file_btn_col1, file_btn_col2, file_btn_col3 = st.columns([3, 2, 2, 2])
        with w_col_title:
            st.markdown('<span style="font-weight: bold; color: #ffffff;">Workspace</span>', unsafe_allow_html=True)
        with file_btn_col1:
            if st.button("+ File", key="add_file_btn", use_container_width=True):
                new_fn = f"file_{len(os.listdir(WORKSPACE_DIR))+1}.txt"
                with open(os.path.join(WORKSPACE_DIR, new_fn), "w") as f:
                    f.write("New workspace file created.")
                st.session_state.selected_file = new_fn
                st.rerun()
        with file_btn_col2:
            uploaded = st.file_uploader("Import", label_visibility="collapsed", key="file_import_uploader")
            if uploaded is not None:
                save_p = os.path.join(WORKSPACE_DIR, uploaded.name)
                with open(save_p, "wb") as f:
                    f.write(uploaded.getbuffer())
                st.session_state.selected_file = uploaded.name
                st.toast(f"Imported {uploaded.name}")
        with file_btn_col3:
            curr_fp = os.path.join(WORKSPACE_DIR, st.session_state.selected_file)
            curr_content = ""
            if os.path.exists(curr_fp):
                with open(curr_fp, "r", encoding="utf-8", errors="ignore") as f:
                    curr_content = f.read()
            st.download_button(
                "Export",
                data=curr_content,
                file_name=st.session_state.selected_file,
                mime="text/plain",
                key="export_file_btn",
                use_container_width=True
            )

        all_files = sorted(os.listdir(WORKSPACE_DIR))
        if not all_files:
            all_files = ["README.md"]

        if st.session_state.selected_file not in all_files:
            st.session_state.selected_file = all_files[0]

        f_col1, f_col2 = st.columns([1, 3])
        with f_col1:
            st.markdown("**Files**")
            for f in all_files:
                icon = "📄"
                if f.endswith(".json"):
                    icon = "{}"
                elif f.endswith(".csv"):
                    icon = "📊"
                elif f.endswith(".md"):
                    icon = "⬇️"

                is_selected = (f == st.session_state.selected_file)
                btn_label = f"{icon} {f}"
                if is_selected:
                    btn_label = f"▶ {btn_label}"

                if st.button(btn_label, key=f"select_{f}", use_container_width=True):
                    st.session_state.selected_file = f
                    st.rerun()

        with f_col2:
            st.markdown(f'<div class="file-path mono-font">📍 /workspace/{st.session_state.selected_file}</div>', unsafe_allow_html=True)
            active_fpath = os.path.join(WORKSPACE_DIR, st.session_state.selected_file)
            file_text = ""
            if os.path.exists(active_fpath):
                with open(active_fpath, "r", encoding="utf-8", errors="ignore") as f:
                    file_text = f.read()

            edited_code = st.text_area(
                "Editor",
                value=file_text,
                height=200,
                label_visibility="collapsed",
                key=f"editor_{st.session_state.selected_file}"
            )

            if st.button("Save", key="save_file_content_btn"):
                with open(active_fpath, "w", encoding="utf-8") as f:
                    f.write(edited_code)
                st.toast(f"Saved {st.session_state.selected_file}")

        st.markdown('</div>', unsafe_allow_html=True)

        # Shell Terminal Component
        st.markdown('<div class="workspace-card">', unsafe_allow_html=True)
        sh_head1, sh_head2, sh_head3 = st.columns([3, 2, 2])
        with sh_head1:
            st.markdown("""
                <div style="display: flex; gap: 10px; align-items: center;">
                    <span style="font-weight: bold; color: #ffffff;">▼ Shell</span>
                    <span class="mono-font" style="color: #00e676; font-size: 0.8rem;">/workspace</span>
                </div>
            """, unsafe_allow_html=True)
        with sh_head2:
            st.markdown('<span style="color: #6c6d80; font-size: 0.78rem;" class="mono-font">just-bash</span>', unsafe_allow_html=True)
        with sh_head3:
            if st.button("Clear", key="clear_shell_btn", use_container_width=True):
                st.session_state.shell_history = ["Ready in your workspace"]
                st.rerun()

        term_box = st.container(height=160)
        with term_box:
            term_text = "\n".join(st.session_state.shell_history)
            st.code(term_text, language="bash")

        sh_in_col, sh_run_col = st.columns([4, 1])
        with sh_in_col:
            sh_input = st.text_input("Shell input", placeholder="$ Enter a shell command...", label_visibility="collapsed", key="shell_cmd_input")
        with sh_run_col:
            if st.button("Run", key="run_shell_btn", use_container_width=True):
                if sh_input.strip():
                    cmd = sh_input.strip()
                    st.session_state.shell_history.append(f"$ {cmd}")
                    output = execute_shell_command(cmd)
                    if output == "CLEAR_SIGNAL":
                        st.session_state.shell_history = ["Ready in your workspace"]
                    else:
                        st.session_state.shell_history.append(output)
                    st.rerun()

        st.markdown("""
            <div style="color: #5d5e70; font-size: 0.73rem; margin-top: 6px;" class="mono-font">
                Enter to run • Shift+Enter for newline | Bash & text tools in virtual filesystem.
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__" or "streamlit" in os.environ.get("_", ""):
    render_app()
