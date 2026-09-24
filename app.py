import streamlit as st
import os
import subprocess
import sys
import platform
import psutil
import pandas as pd

# Set page config at the very top
st.set_page_config(
    page_title="Termux Terminal CLI",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Termux Dark Theme CSS
st.markdown("""
<style>
    .stApp {
        background-color: #000000;
        color: #00ff66;
        font-family: 'Courier New', Courier, monospace;
    }
    .termux-banner {
        background-color: #050505;
        color: #00ff66;
        border: 1px solid #004411;
        border-radius: 4px;
        padding: 12px;
        font-family: monospace;
        white-space: pre-wrap;
        margin-bottom: 10px;
    }
    .touch-key-btn {
        background-color: #111111;
        color: #00ff66;
        border: 1px solid #005522;
        border-radius: 4px;
        padding: 5px;
        text-align: center;
        font-weight: bold;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

TERMUX_BANNER = """Welcome to Termux (Linux Web CLI)!

Community forum: https://termux.com/community
Gitter chat:     https://gitter.im/termux/termux

Working with packages:
 * Search packages:   pkg search <query>
 * Install a package: pkg install <package>
 * Upgrade packages:  pkg upgrade

Subscribed repositories:
 * main (Linux Debian/Ubuntu sandbox)
 * python / pip package manager

Termux commands available: termux-info, termux-setup-storage, pkg, apt
"""

# Initialize Session State
if "cwd" not in st.session_state:
    st.session_state.cwd = os.getcwd()

if "cmd_history" not in st.session_state:
    st.session_state.cmd_history = []

if "input_buffer" not in st.session_state:
    st.session_state.input_buffer = ""

def execute_shell_command(cmd, cwd=None):
    if cwd is None:
        cwd = st.session_state.cwd if "cwd" in st.session_state else os.getcwd()

    cmd_str = cmd.strip()
    if not cmd_str:
        return "", "", 0

    # Termux native command handlers
    if cmd_str == "termux-info":
        info = (
            f"Termux Variables:\n"
            f"TERMUX_VERSION=0.118.0\n"
            f"TERMUX_MAIN_PACKAGE_FORMAT=debian\n"
            f"ARCH={platform.machine()}\n"
            f"OS={platform.system()} {platform.release()}\n"
            f"PREFIX=/data/data/com.termux/files/usr\n"
            f"HOME={cwd}\n"
        )
        return info, "", 0

    if cmd_str == "termux-setup-storage":
        storage_msg = (
            "Creating storage directory ~/storage...\n"
            "Symlinks created:\n"
            " ~/storage/dcim -> /sdcard/DCIM\n"
            " ~/storage/downloads -> /sdcard/Download\n"
            " ~/storage/shared -> /sdcard\n"
        )
        os.makedirs(os.path.join(cwd, "storage"), exist_ok=True)
        return storage_msg, "", 0

    if cmd_str == "clear":
        if "cmd_history" in st.session_state:
            st.session_state.cmd_history = []
        return "", "", 0

    # Handle 'pkg' and 'apt' commands
    if cmd_str.startswith("pkg ") or cmd_str == "pkg" or cmd_str.startswith("apt ") or cmd_str == "apt":
        parts = cmd_str.split(maxsplit=2)
        sub = parts[1] if len(parts) > 1 else "help"

        if sub in ["update", "upgrade"]:
            cmd_str = "apt-get update"
        elif sub == "install" and len(parts) > 2:
            pkg_name = parts[2]
            cmd_str = f"pip install {pkg_name} || apt-get install -y {pkg_name}"
        elif sub in ["search", "list-all", "list"]:
            query = parts[2] if len(parts) > 2 else ""
            cmd_str = f"pip search {query}" if query else "pip list"
        elif sub in ["help", "-h", "--help"]:
            help_msg = (
                "Termux package manager (pkg / apt emulation):\n"
                "  pkg install <pkg>  Install a Python/Linux package\n"
                "  pkg update         Update package lists\n"
                "  pkg list           List installed packages\n"
            )
            return help_msg, "", 0

    # Handle 'cd' command
    if cmd_str.startswith("cd ") or cmd_str == "cd":
        target = cmd_str[3:].strip() if len(cmd_str) > 2 else os.path.expanduser("~")
        if not target:
            target = os.path.expanduser("~")
        new_path = os.path.abspath(os.path.join(cwd, target))
        if os.path.exists(new_path) and os.path.isdir(new_path):
            if "cwd" in st.session_state:
                st.session_state.cwd = new_path
            return f"Changed directory to {new_path}\n", "", 0
        else:
            return "", f"cd: {target}: No such file or directory\n", 1

    try:
        res = subprocess.run(
            cmd_str,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30
        )
        return res.stdout, res.stderr, res.returncode
    except subprocess.TimeoutExpired:
        return "", "Error: Command timed out after 30 seconds.\n", 124
    except Exception as e:
        return "", f"Execution error: {str(e)}\n", 1

# System Information Helpers
def get_system_metrics():
    cpu_percent = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(st.session_state.cwd if "cwd" in st.session_state else os.getcwd())
    return {
        "cpu_percent": cpu_percent,
        "mem_percent": mem.percent,
        "mem_used_gb": round(mem.used / (1024**3), 2),
        "mem_total_gb": round(mem.total / (1024**3), 2),
        "disk_percent": disk.percent,
        "disk_used_gb": round(disk.used / (1024**3), 2),
        "disk_total_gb": round(disk.total / (1024**3), 2)
    }

def get_process_list():
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 'memory_percent']):
        try:
            processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return pd.DataFrame(processes).sort_values(by='cpu_percent', ascending=False) if processes else pd.DataFrame()

# Main Header
st.title("📱 Termux Real CLI Terminal")
st.caption(f"Termux Terminal Session | Working Directory: `{st.session_state.cwd}`")

tab_terminal, tab_filemanager, tab_sysinfo = st.tabs(["💻 Termux CLI Terminal", "📁 File Manager & Editor", "📊 System & Resources"])

# TAB 1: TERMINAL / CLI
with tab_terminal:
    # Display Termux Banner
    st.markdown(f'<div class="termux-banner">{TERMUX_BANNER}</div>', unsafe_allow_html=True)

    # Termux Touch Keys Bar
    st.write("Termux Touch Keys:")
    tk1, tk2, tk3, tk4, tk5, tk6, tk7, tk8, tk9 = st.columns(9)

    append_key = None
    if tk1.button("`ESC`"):
        append_key = ""
    if tk2.button("`TAB`"):
        append_key = "  "
    if tk3.button("`CTRL`"):
        append_key = "^C"
    if tk4.button("`ALT`"):
        append_key = ""
    if tk5.button("`-`"):
        append_key = " - "
    if tk6.button("`/`"):
        append_key = "/"
    if tk7.button("`|`"):
        append_key = " | "
    if tk8.button("`~`"):
        append_key = "~"
    if tk9.button("`CLEAR`"):
        st.session_state.cmd_history = []
        st.rerun()

    # Quick Command Shortcuts
    st.write("Quick Commands:")
    q_col1, q_col2, q_col3, q_col4, q_col5, q_col6 = st.columns(6)

    run_quick_cmd = None
    if q_col1.button("`termux-info`"):
        run_quick_cmd = "termux-info"
    if q_col2.button("`pkg list`"):
        run_quick_cmd = "pkg list"
    if q_col3.button("`ls -la`"):
        run_quick_cmd = "ls -la"
    if q_col4.button("`pwd`"):
        run_quick_cmd = "pwd"
    if q_col5.button("`top`"):
        run_quick_cmd = "top -bn1 | head -n 20"
    if q_col6.button("`python --version`"):
        run_quick_cmd = "python3 --version"

    # Command Input Box
    termux_prompt = f"u0_a241@localhost:{st.session_state.cwd} $"
    with st.form(key="terminal_form", clear_on_submit=True):
        user_input = st.text_input(f"{termux_prompt}", value=append_key if append_key else "", placeholder="Enter command: e.g. pkg install htop, termux-info, ls -la, python3")
        submit_btn = st.form_submit_button("Run Command 🚀")

    cmd_to_run = run_quick_cmd or (user_input if submit_btn else None)

    if cmd_to_run:
        stdout, stderr, code = execute_shell_command(cmd_to_run)
        if cmd_to_run != "clear":
            st.session_state.cmd_history.append({
                "cmd": cmd_to_run,
                "cwd": st.session_state.cwd,
                "prompt": f"u0_a241@localhost:{st.session_state.cwd} $",
                "stdout": stdout,
                "stderr": stderr,
                "code": code
            })

    # Terminal History Controls
    col_hist_title, col_hist_clear = st.columns([4, 1])
    with col_hist_clear:
        if st.button("Clear Terminal"):
            st.session_state.cmd_history = []
            st.rerun()

    # Display Terminal Log
    if st.session_state.cmd_history:
        for idx, item in enumerate(reversed(st.session_state.cmd_history)):
            st.markdown(f"**`{item.get('prompt', '$')} {item['cmd']}`** (Exit: `{item['code']}`)")
            if item["stdout"]:
                st.text_area(f"stdout-{idx}", value=item["stdout"], height=150, key=f"stdout_{idx}")
            if item["stderr"]:
                st.error(f"Error:\n{item['stderr']}")
            st.divider()

# TAB 2: FILE MANAGER & EDITOR
with tab_filemanager:
    st.subheader("Termux Workspace File Browser & Code Editor")

    current_dir = st.session_state.cwd

    col_dir_nav, col_actions = st.columns([3, 1])
    with col_dir_nav:
        st.write(f"📁 Directory: **`{current_dir}`**")
    with col_actions:
        if st.button("Up One Directory (`..`)"):
            st.session_state.cwd = os.path.dirname(current_dir)
            st.rerun()

    try:
        dir_contents = os.listdir(current_dir)
        files_dirs = []
        for name in dir_contents:
            full_path = os.path.join(current_dir, name)
            is_dir = os.path.isdir(full_path)
            size_str = "-" if is_dir else f"{os.path.getsize(full_path):,} bytes"
            files_dirs.append({
                "Type": "📁 Folder" if is_dir else "📄 File",
                "Name": name,
                "Size": size_str
            })

        df_files = pd.DataFrame(files_dirs)
        st.dataframe(df_files, use_container_width=True)
    except Exception as e:
        st.error(f"Cannot list directory: {e}")

    st.divider()

    # File Operations
    fm_tab1, fm_tab2, fm_tab3 = st.tabs(["✏️ Edit / View File", "➕ Create File/Folder", "📤 Upload File"])

    with fm_tab1:
        file_to_edit = st.text_input("File path to edit / view:", value=os.path.join(current_dir, "README.md"))
        if os.path.exists(file_to_edit) and os.path.isfile(file_to_edit):
            try:
                with open(file_to_edit, "r", encoding="utf-8", errors="ignore") as f:
                    file_content = f.read()

                new_content = st.text_area("File Content:", value=file_content, height=300)
                if st.button("Save Changes"):
                    with open(file_to_edit, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    st.success(f"Saved changes to `{file_to_edit}` successfully!")
            except Exception as e:
                st.error(f"Error reading file: {e}")
        else:
            st.warning("File does not exist or is a directory.")

    with fm_tab2:
        c1, c2 = st.columns(2)
        with c1:
            new_file_name = st.text_input("New file name:", placeholder="example.txt")
            new_file_content = st.text_area("Initial Content:", placeholder="Hello Termux World")
            if st.button("Create File"):
                if new_file_name:
                    file_path = os.path.join(current_dir, new_file_name)
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(new_file_content)
                    st.success(f"Created file `{new_file_name}`!")
                    st.rerun()
        with c2:
            new_folder_name = st.text_input("New directory name:", placeholder="my_folder")
            if st.button("Create Directory"):
                if new_folder_name:
                    dir_path = os.path.join(current_dir, new_folder_name)
                    os.makedirs(dir_path, exist_ok=True)
                    st.success(f"Created directory `{new_folder_name}`!")
                    st.rerun()

    with fm_tab3:
        uploaded_file = st.file_uploader("Upload file to current directory:")
        if uploaded_file is not None:
            save_path = os.path.join(current_dir, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Uploaded `{uploaded_file.name}` to `{current_dir}`!")
            st.rerun()

# TAB 3: SYSTEM & RESOURCES
with tab_sysinfo:
    st.subheader("System Resources & Environment")

    metrics = get_system_metrics()

    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("CPU Usage", f"{metrics['cpu_percent']}%")
    m_col2.metric("Memory Usage", f"{metrics['mem_percent']}%", f"{metrics['mem_used_gb']} GB / {metrics['mem_total_gb']} GB")
    m_col3.metric("Disk Usage", f"{metrics['disk_percent']}%", f"{metrics['disk_used_gb']} GB / {metrics['disk_total_gb']} GB")

    st.divider()

    st.write("### OS Kernel & Python Runtime")
    st.json({
        "System OS": platform.system(),
        "OS Release": platform.release(),
        "Architecture": platform.machine(),
        "Python Version": sys.version,
        "Executable": sys.executable
    })

    st.divider()

    st.write("### Active System Processes")
    if st.button("Refresh Processes"):
        st.rerun()
    proc_df = get_process_list()
    if not proc_df.empty:
        st.dataframe(proc_df.head(20), use_container_width=True)

    st.divider()

    st.write("### Environment Variables")
    masked_env = []
    sensitive_keywords = ["TOKEN", "KEY", "SECRET", "PASSWORD", "AUTH"]
    for k, v in os.environ.items():
        if any(keyword in k.upper() for keyword in sensitive_keywords):
            masked_env.append((k, "********"))
        else:
            masked_env.append((k, v))
    env_df = pd.DataFrame(masked_env, columns=["Variable", "Value"])
    st.dataframe(env_df, use_container_width=True)
