import streamlit as st
import time
import os
import sys
import subprocess
import threading
import asyncio
import io
import logging
import platform
import shutil
from pypdf import PdfReader
from telegram import Update
from telegram.error import NetworkError
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from huggingface_hub import InferenceClient

# Page configuration
st.set_page_config(
    page_title="macOS Workstation & Docker Studio",
    page_icon="🍎",
    layout="wide"
)

# Inject macOS Theme CSS
st.markdown("""
<style>
    /* macOS Menu Bar */
    .macos-menubar {
        background-color: rgba(30, 30, 30, 0.85);
        backdrop-filter: blur(10px);
        color: #f0f0f0;
        padding: 6px 16px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-size: 13px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-radius: 8px 8px 0 0;
        margin-bottom: 15px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    .macos-menu-left {
        display: flex;
        gap: 18px;
        align-items: center;
        font-weight: 500;
    }
    .apple-logo {
        font-size: 16px;
        margin-right: 4px;
    }

    /* macOS Terminal Window Frame */
    .macos-window {
        background: #1e1e1e;
        border-radius: 10px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.5);
        border: 1px solid #333333;
        overflow: hidden;
        margin-bottom: 20px;
    }
    .macos-titlebar {
        background: #2d2d2d;
        padding: 8px 12px;
        display: flex;
        align-items: center;
        border-bottom: 1px solid #1a1a1a;
    }
    .traffic-lights {
        display: flex;
        gap: 8px;
    }
    .light {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
    }
    .light-red { background-color: #ff5f56; border: 1px solid #e0443e; }
    .light-yellow { background-color: #ffbd2e; border: 1px solid #dea123; }
    .light-green { background-color: #27c93f; border: 1px solid #1aab29; }
    .window-title {
        color: #a0a0a0;
        font-size: 12px;
        font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
        margin-left: auto;
        margin-right: auto;
        font-weight: 600;
    }

    /* macOS Card Containers */
    .macos-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        backdrop-filter: blur(12px);
    }
</style>

<div class="macos-menubar">
    <div class="macos-menu-left">
        <span class="apple-logo">🍎</span>
        <span style="font-weight: 700;">macOS Terminal Studio</span>
        <span>File</span>
        <span>Edit</span>
        <span>View</span>
        <span>Docker</span>
        <span>Window</span>
        <span>Help</span>
    </div>
    <div>
        <span>🔋 100%</span> &nbsp;|&nbsp;
        <span>📶 Connected</span> &nbsp;|&nbsp;
        <span>Docker Engine Ready 🐳</span>
    </div>
</div>
""", unsafe_allow_html=True)

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


# --- Linux & Docker Shell Engine & Helpers ---

def get_current_working_dir():
    if "working_dir" not in st.session_state or not os.path.isdir(st.session_state.working_dir):
        st.session_state.working_dir = os.getcwd()
    return st.session_state.working_dir

def execute_shell_command(cmd_str, cwd=None, timeout=60):
    """
    Executes a shell command in Linux environment.
    Supports cd command navigation with working directory persistence.
    """
    if cwd is None:
        cwd = get_current_working_dir()

    cmd_str = cmd_str.strip()
    if not cmd_str:
        return {"stdout": "", "stderr": "", "returncode": 0, "cwd": cwd}

    # Handle 'cd' commands directly
    if cmd_str == "cd" or cmd_str.startswith("cd "):
        target_dir = cmd_str[3:].strip() if len(cmd_str) > 2 else os.path.expanduser("~")
        if not target_dir:
            target_dir = os.path.expanduser("~")

        # Resolve path
        new_path = os.path.abspath(os.path.join(cwd, target_dir))
        if os.path.isdir(new_path):
            st.session_state.working_dir = new_path
            return {
                "stdout": f"Changed directory to: {new_path}",
                "stderr": "",
                "returncode": 0,
                "cwd": new_path
            }
        else:
            return {
                "stdout": "",
                "stderr": f"bash: cd: {target_dir}: No such file or directory",
                "returncode": 1,
                "cwd": cwd
            }

    try:
        result = subprocess.run(
            cmd_str,
            shell=True,
            executable="/bin/bash",
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "cwd": cwd
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": f"Command timed out after {timeout} seconds.",
            "returncode": 124,
            "cwd": cwd
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": f"Execution error: {str(e)}",
            "returncode": 1,
            "cwd": cwd
        }

def get_docker_status():
    """Checks if Docker CLI and daemon are available."""
    res = subprocess.run("docker --version", shell=True, capture_output=True, text=True)
    if res.returncode == 0:
        version_str = res.stdout.strip()
        ps_res = subprocess.run("docker ps", shell=True, capture_output=True, text=True)
        daemon_ready = (ps_res.returncode == 0)
        return {
            "installed": True,
            "version": version_str,
            "daemon_ready": daemon_ready,
            "error": ps_res.stderr if not daemon_ready else ""
        }
    return {
        "installed": False,
        "version": "Not Found",
        "daemon_ready": False,
        "error": "Docker CLI not installed"
    }

def run_docker_command(docker_cmd):
    """Runs a docker command safely."""
    if not docker_cmd.startswith("docker"):
        docker_cmd = f"docker {docker_cmd}"
    return execute_shell_command(docker_cmd)

def get_system_info():
    """Gathers detailed system and environment information."""
    info = {}
    info["os"] = f"{platform.system()} {platform.release()} ({platform.machine()})"
    info["python_version"] = sys.version.split()[0]

    # OS Release info
    if os.path.exists("/etc/os-release"):
        try:
            with open("/etc/os-release", "r") as f:
                for line in f:
                    if line.startswith("PRETTY_NAME="):
                        info["distro"] = line.split("=")[1].strip().strip('"')
                        break
        except Exception:
            info["distro"] = "Linux"
    else:
        info["distro"] = platform.platform()

    # User & Host
    try:
        info["user"] = subprocess.getoutput("whoami")
        info["hostname"] = subprocess.getoutput("hostname")
    except Exception:
        info["user"] = "unknown"
        info["hostname"] = "localhost"

    # Memory & Disk
    try:
        mem_output = subprocess.getoutput("free -h")
        info["memory"] = mem_output
    except Exception:
        info["memory"] = "N/A"

    try:
        disk = shutil.disk_usage("/")
        info["disk_total_gb"] = round(disk.total / (1024**3), 2)
        info["disk_used_gb"] = round(disk.used / (1024**3), 2)
        info["disk_free_gb"] = round(disk.free / (1024**3), 2)
        info["disk_percent"] = round((disk.used / disk.total) * 100, 1)
    except Exception:
        info["disk_total_gb"] = 0
        info["disk_used_gb"] = 0
        info["disk_free_gb"] = 0
        info["disk_percent"] = 0

    return info


# --- PDF & Gemini Helpers ---

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


# --- Telegram Bot Handlers ---

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


# Start Telegram Bot Background Thread
if "bot_thread" not in st.session_state:
    st.session_state.bot_thread = True
    thread = threading.Thread(target=run_bot, daemon=True)
    thread.start()


# --- Streamlit UI Layout ---

st.title("🖥️ macOS Workstation & Docker Studio")
st.markdown("Môi trường làm việc phong cách **macOS** tích hợp **Docker CLI** và **Linux Terminal** trên Hugging Face Spaces.")

# Session state initialization for history
if "cmd_history" not in st.session_state:
    st.session_state.cmd_history = []

cwd = get_current_working_dir()
sys_info = get_system_info()
docker_info = get_docker_status()

# Header status bar
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("macOS Workstation Host", sys_info.get("user", "jules"))
with col2:
    st.metric("Docker CLI", docker_info["version"].split(",")[0] if docker_info["installed"] else "N/A")
with col3:
    st.metric("Disk Free", f"{sys_info.get('disk_free_gb', 0)} GB")
with col4:
    st.metric("Python Runtime", sys_info.get("python_version", "3.x"))


# Create Tabs
tab_cli, tab_docker, tab_sys, tab_bot = st.tabs(["💻 macOS Terminal CLI", "🐳 Docker Studio", "📊 System & Environment", "🤖 Telegram Bot & AI Studio"])

# --- TAB 1: macOS Terminal CLI ---
with tab_cli:
    # macOS Terminal Window Frame Header
    st.markdown("""
    <div class="macos-window">
        <div class="macos-titlebar">
            <div class="traffic-lights">
                <span class="light light-red"></span>
                <span class="light light-yellow"></span>
                <span class="light light-green"></span>
            </div>
            <div class="window-title">bash — 80x24</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"**Current Directory (`PWD`):** `{cwd}`")

    # Quick command buttons
    st.markdown("**Quick macOS / Linux Preset Commands:**")
    btn_col1, btn_col2, btn_col3, btn_col4, btn_col5, btn_col6, btn_col7 = st.columns(7)
    preset_cmd = None
    if btn_col1.button("📁 `ls -la`"):
        preset_cmd = "ls -la"
    if btn_col2.button("📍 `pwd`"):
        preset_cmd = "pwd"
    if btn_col3.button("🐍 `python3 -V`"):
        preset_cmd = "python3 --version"
    if btn_col4.button("📦 `pip list`"):
        preset_cmd = "pip list"
    if btn_col5.button("💾 `df -h`"):
        preset_cmd = "df -h"
    if btn_col6.button("🧠 `free -h`"):
        preset_cmd = "free -h"
    if btn_col7.button("⚙️ `ps aux`"):
        preset_cmd = "ps aux | head -n 20"

    # Command execution form
    with st.form("cli_form", clear_on_submit=True):
        cmd_input = st.text_input(f"[{sys_info.get('user', 'user')}@{sys_info.get('hostname', 'macbook')} {os.path.basename(cwd) or '/'}]#", value=preset_cmd or "")
        submitted = st.form_submit_button("🚀 Run Command", use_container_width=True)

    if (submitted and cmd_input) or preset_cmd:
        exec_cmd = cmd_input if submitted and cmd_input else preset_cmd
        res = execute_shell_command(exec_cmd, cwd=cwd)
        st.session_state.cmd_history.append({
            "command": exec_cmd,
            "cwd": cwd,
            "result": res
        })
        st.rerun()

    # Clear history option
    if st.session_state.cmd_history:
        c_left, c_right = st.columns([6, 1])
        with c_right:
            if st.button("🗑️ Clear Output", key="clear_hist"):
                st.session_state.cmd_history = []
                st.rerun()

    # Terminal Output Window
    st.markdown("---")
    st.markdown("### Terminal Console Output")

    if not st.session_state.cmd_history:
        st.info("Nhập lệnh Linux / macOS vào ô trên và nhấn **Run Command** (hoặc chọn lệnh nhanh) để bắt đầu.")
    else:
        for idx, entry in enumerate(reversed(st.session_state.cmd_history)):
            c_user = sys_info.get('user', 'user')
            c_host = sys_info.get('hostname', 'macbook')
            e_cwd = entry['cwd']
            cmd = entry['command']
            res = entry['result']

            st.code(f"{c_user}@{c_host}:{e_cwd}$ {cmd}", language="bash")

            stdout = res.get("stdout", "")
            stderr = res.get("stderr", "")
            retcode = res.get("returncode", 0)

            if stdout:
                st.code(stdout, language="text")
            if stderr:
                st.error(f"Error (Exit Code {retcode}):\n{stderr}")
            if not stdout and not stderr:
                st.caption(f"Command executed (Exit code: {retcode}). No output returned.")


# --- TAB 2: Docker Studio ---
with tab_docker:
    st.subheader("🐳 Docker Container & Image Management Studio")

    if docker_info["installed"]:
        st.success(f"**Docker Engine:** {docker_info['version']}")
        if not docker_info["daemon_ready"]:
            st.warning("⚠️ Docker CLI is available, but the Docker daemon socket is not currently running or restricted in this container environment.")
            if docker_info["error"]:
                st.caption(f"Daemon response: {docker_info['error']}")

        st.markdown("#### Quick Docker Controls")
        d_col1, d_col2, d_col3, d_col4 = st.columns(4)
        docker_preset = None
        if d_col1.button("📦 `docker ps -a`"):
            docker_preset = "docker ps -a"
        if d_col2.button("🖼️ `docker images`"):
            docker_preset = "docker images"
        if d_col3.button("ℹ️ `docker info`"):
            docker_preset = "docker info"
        if d_col4.button("🏷️ `docker version`"):
            docker_preset = "docker version"

        with st.form("docker_form"):
            docker_input = st.text_input("Docker Command:", value=docker_preset or "docker ps -a")
            d_submitted = st.form_submit_button("🚀 Run Docker Command", use_container_width=True)

        if d_submitted or docker_preset:
            target_docker_cmd = docker_input if d_submitted else docker_preset
            d_res = run_docker_command(target_docker_cmd)
            st.markdown(f"**Command Output (`{target_docker_cmd}`):**")
            if d_res["stdout"]:
                st.code(d_res["stdout"], language="text")
            if d_res["stderr"]:
                st.error(d_res["stderr"])
    else:
        st.error("Docker is not installed on this environment.")


# --- TAB 3: System Info & Resources ---
with tab_sys:
    st.subheader("macOS & Linux System Details")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### System Overview")
        st.json({
            "Operating System": sys_info.get("os"),
            "Distribution": sys_info.get("distro"),
            "Current User": sys_info.get("user"),
            "Hostname": sys_info.get("hostname"),
            "Python Executable": sys.executable,
            "Python Version": sys_info.get("python_version"),
            "Working Directory": cwd
        })

    with col_b:
        st.markdown("#### Disk Space Usage (`/`)")
        st.progress(sys_info.get("disk_percent", 0) / 100.0)
        st.write(f"**Used:** {sys_info.get('disk_used_gb')} GB / {sys_info.get('disk_total_gb')} GB ({sys_info.get('disk_percent')}%)")
        st.write(f"**Free:** {sys_info.get('disk_free_gb')} GB")

        st.markdown("#### RAM Memory Stats (`free -h`)")
        st.code(sys_info.get("memory", "N/A"), language="text")

    st.markdown("---")
    st.markdown("#### Environment Variables")
    if st.checkbox("Show Environment Variables (Masked Tokens)"):
        env_vars = {}
        for k, v in os.environ.items():
            if any(secret in k.upper() for secret in ["TOKEN", "KEY", "SECRET", "PASS", "AUTH"]):
                env_vars[k] = v[:4] + "..." + v[-4:] if len(v) > 8 else "********"
            else:
                env_vars[k] = v
        st.json(env_vars)


# --- TAB 4: Telegram Bot & AI Studio ---
with tab_bot:
    st.subheader("VBot1 Hybrid AI & Telegram Status")
    st.write("Status: Telegram Bot is running in background thread.")

    bot_col1, bot_col2 = st.columns(2)
    with bot_col1:
        st.info(f"**Llama 3 (Chat):** {'🟢 Active' if hf_client else '🔴 Inactive (HF_TOKEN missing)'}")
    with bot_col2:
        st.info(f"**Gemini 1.5 Flash (PDF Summary):** {'🟢 Active' if GOOGLE_API_KEY else '🔴 Inactive (GOOGLE_API_KEY missing)'}")

    st.markdown("---")
    st.markdown("### Interactive Gemini PDF Summarizer Test")
    uploaded_pdf = st.file_uploader("Upload a PDF file to test summarization", type=["pdf"])
    if uploaded_pdf is not None:
        if st.button("Summarize PDF via Gemini"):
            with st.spinner("Extracting text and summarizing..."):
                text = extract_pdf_text(uploaded_pdf.read())
                if text:
                    summary = summarize_with_gemini(text)
                    st.subheader("Summary Result:")
                    st.write(summary)
                else:
                    st.error("Could not extract text from the PDF file.")
