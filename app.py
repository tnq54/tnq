import os
import sys
import socket
import shlex
import subprocess
import pandas as pd
import psutil
import streamlit as st

# Ensure page configuration is set at the very top before any Streamlit calls or session_state access
st.set_page_config(
    page_title="Termux Web Workspace",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color Themes
THEMES = {
    "Classic Green": {
        "bg": "#000000",
        "text": "#00ff66",
        "prompt": "#00ff66",
        "accent": "#00cc55",
        "card_bg": "#111111"
    },
    "Monokai": {
        "bg": "#272822",
        "text": "#f8f8f2",
        "prompt": "#a6e22e",
        "accent": "#f92672",
        "card_bg": "#3e3d32"
    },
    "Dracula": {
        "bg": "#282a36",
        "text": "#f8f8f2",
        "prompt": "#50fa7b",
        "accent": "#bd93f9",
        "card_bg": "#44475a"
    },
    "Solarized Dark": {
        "bg": "#002b36",
        "text": "#839496",
        "prompt": "#859900",
        "accent": "#268bd2",
        "card_bg": "#073642"
    },
    "Cyan Cyberpunk": {
        "bg": "#050811",
        "text": "#00f0ff",
        "prompt": "#ff007f",
        "accent": "#7000ff",
        "card_bg": "#0d1527"
    }
}

CLIPBOARD_STORAGE = ""

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def get_system_metrics():
    cpu_percent = psutil.cpu_percent(interval=0.5)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    return {
        "cpu_percent": cpu_percent,
        "memory_percent": memory.percent,
        "memory_used_gb": round(memory.used / (1024 ** 3), 2),
        "memory_total_gb": round(memory.total / (1024 ** 3), 2),
        "disk_percent": disk.percent,
        "disk_used_gb": round(disk.used / (1024 ** 3), 2),
        "disk_total_gb": round(disk.total / (1024 ** 3), 2)
    }

def get_process_list():
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 'memory_percent']):
        try:
            pinfo = proc.info
            processes.append({
                "PID": pinfo['pid'],
                "Name": pinfo['name'] or "N/A",
                "User": pinfo['username'] or "N/A",
                "CPU %": round(pinfo['cpu_percent'] or 0.0, 1),
                "Mem %": round(pinfo['memory_percent'] or 0.0, 1)
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return processes

def parse_pkg_command(cmd_str: str) -> str:
    """Translates Termux pkg commands to pip commands and strips apt flags."""
    tokens = shlex.split(cmd_str)
    if not tokens or tokens[0] != "pkg":
        return cmd_str

    # Strip apt flags
    flags_to_strip = {"-y", "--yes", "-q", "--quiet"}
    args = [t for t in tokens[1:] if t not in flags_to_strip]

    if not args:
        return "echo 'pkg: missing subcommand. Try: pkg help'"

    action = args[0]
    pkg_args = args[1:]

    if action in ("install", "i"):
        if not pkg_args:
            return "echo 'pkg install: missing package name'"
        return f"{sys.executable} -m pip install {' '.join(pkg_args)}"
    elif action in ("uninstall", "remove"):
        if not pkg_args:
            return "echo 'pkg uninstall: missing package name'"
        return f"{sys.executable} -m pip uninstall -y {' '.join(pkg_args)}"
    elif action in ("update", "upgrade"):
        return f"{sys.executable} -m pip list --outdated"
    elif action == "search":
        if not pkg_args:
            return f"{sys.executable} -m pip list"
        return f"{sys.executable} -m pip list | grep -i '{pkg_args[0]}'"
    elif action == "list-installed":
        return f"{sys.executable} -m pip list"
    elif action == "help":
        return "echo 'Termux pkg Emulation:\n  pkg install <pkg>\n  pkg uninstall <pkg>\n  pkg update / upgrade\n  pkg search <query>\n  pkg list-installed'"
    else:
        return f"echo 'pkg: unknown subcommand {action}'"

def execute_shell_command(cmd_str: str, cwd: str = None) -> tuple[str, str, int, str]:
    """
    Executes shell command with directory tracking, pkg parsing, termux emulation, and background support.
    Returns (stdout, stderr, returncode, updated_cwd).
    """
    global CLIPBOARD_STORAGE
    if cwd is None:
        cwd = os.getcwd()

    cmd_str = cmd_str.strip()
    if not cmd_str:
        return "", "", 0, cwd

    # Handle 'cd' command
    if cmd_str.startswith("cd ") or cmd_str == "cd":
        parts = cmd_str.split(maxsplit=1)
        target_dir = parts[1].strip() if len(parts) > 1 else os.path.expanduser("~")
        target_dir = os.path.expanduser(target_dir)
        new_path = os.path.abspath(os.path.join(cwd, target_dir))
        if os.path.exists(new_path) and os.path.isdir(new_path):
            return f"Changed directory to {new_path}", "", 0, new_path
        else:
            return "", f"cd: no such file or directory: {target_dir}", 1, cwd

    # Handle tsu / su / root emulation
    if cmd_str in ("tsu", "su", "sudo su"):
        return "Emulated root environment active (tsu). Superuser access enabled.", "", 0, cwd

    # Handle pkg commands
    if cmd_str.startswith("pkg"):
        cmd_str = parse_pkg_command(cmd_str)

    # Handle termux-* API emulation
    if cmd_str.startswith("termux-"):
        parts = cmd_str.split(maxsplit=1)
        t_cmd = parts[0]
        arg = parts[1] if len(parts) > 1 else ""

        if t_cmd == "termux-info":
            info = (
                f"Termux Emulation Environment (Hugging Face Spaces)\n"
                f"OS: {sys.platform}\n"
                f"Python: {sys.version.split()[0]}\n"
                f"Working Directory: {cwd}"
            )
            return info, "", 0, cwd
        elif t_cmd == "termux-setup-storage":
            storage_path = os.path.expanduser("~/storage")
            os.makedirs(storage_path, exist_ok=True)
            return f"Storage folder initialized at {storage_path}", "", 0, cwd
        elif t_cmd == "termux-battery-status":
            battery_json = '{\n  "health": "GOOD",\n  "percentage": 100,\n  "plugged": "PLUGGED_AC",\n  "status": "CHARGING",\n  "temperature": 25.0\n}'
            return battery_json, "", 0, cwd
        elif t_cmd == "termux-toast":
            msg = arg.strip("\"'") or "Termux Toast Notification"
            return f"[TOAST]: {msg}", "", 0, cwd
        elif t_cmd == "termux-clipboard-get":
            return CLIPBOARD_STORAGE, "", 0, cwd
        elif t_cmd == "termux-clipboard-set":
            CLIPBOARD_STORAGE = arg.strip("\"'")
            return f"Clipboard set: {CLIPBOARD_STORAGE}", "", 0, cwd

    # Background execution
    if cmd_str.endswith("&"):
        actual_cmd = cmd_str[:-1].strip()
        try:
            proc = subprocess.Popen(actual_cmd, shell=True, cwd=cwd)
            return f"[Background Process Started PID: {proc.pid}]", "", 0, cwd
        except Exception as e:
            return "", str(e), 1, cwd

    # Execute standard shell command
    try:
        res = subprocess.run(
            cmd_str,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30
        )
        return res.stdout, res.stderr, res.returncode, cwd
    except subprocess.TimeoutExpired:
        return "", "Command timed out after 30 seconds", 124, cwd
    except Exception as e:
        return "", str(e), 1, cwd

def main():
    # Session state initialization
    if "cwd" not in st.session_state:
        st.session_state.cwd = os.getcwd()
    if "history" not in st.session_state:
        st.session_state.history = []
    if "selected_theme" not in st.session_state:
        st.session_state.selected_theme = "Classic Green"
    if "cli_input_val" not in st.session_state:
        st.session_state.cli_input_val = ""

    theme = THEMES[st.session_state.selected_theme]

    # Custom CSS for Termux Dark Linux UI
    st.markdown(f"""
        <style>
        .stApp {{
            background-color: {theme['bg']};
            color: {theme['text']};
            font-family: 'Courier New', Courier, monospace;
        }}
        .stSidebar {{
            background-color: {theme['card_bg']};
        }}
        .termux-banner {{
            font-family: monospace;
            white-space: pre;
            color: {theme['prompt']};
            background-color: {theme['card_bg']};
            padding: 12px;
            border-radius: 6px;
            border-left: 4px solid {theme['accent']};
            margin-bottom: 15px;
        }}
        .termux-output {{
            background-color: {theme['card_bg']};
            color: {theme['text']};
            font-family: 'Courier New', Courier, monospace;
            padding: 10px;
            border-radius: 4px;
            white-space: pre-wrap;
            margin-bottom: 10px;
        }}
        </style>
    """, unsafe_allow_html=True)

    # Sidebar Controls
    with st.sidebar:
        st.title("⚙️ Termux Config")
        chosen_theme = st.selectbox(
            "Color Theme",
            options=list(THEMES.keys()),
            index=list(THEMES.keys()).index(st.session_state.selected_theme)
        )
        if chosen_theme != st.session_state.selected_theme:
            st.session_state.selected_theme = chosen_theme
            st.rerun()

        st.subheader("📌 Quick Commands")
        if st.button("pkg update"):
            st.session_state.pending_cmd = "pkg update"
        if st.button("termux-info"):
            st.session_state.pending_cmd = "termux-info"
        if st.button("termux-battery-status"):
            st.session_state.pending_cmd = "termux-battery-status"
        if st.button("Clear Terminal"):
            st.session_state.history = []
            st.rerun()

    # App Tabs
    tab_cli, tab_fm, tab_sys = st.tabs(["💻 Terminal / CLI", "📁 File Manager", "📊 System Resources"])

    # TAB 1: Terminal / CLI
    with tab_cli:
        welcome_banner = """
  Welcome to Termux Web Workspace!
  --------------------------------------------------
  * Working Directory: """ + st.session_state.cwd + """
  * Package Management: `pkg install <package>`
  * Termux API: `termux-info`, `termux-battery-status`
  --------------------------------------------------
"""
        st.markdown(f'<div class="termux-banner">{welcome_banner}</div>', unsafe_allow_html=True)

        # Render Terminal History
        for item in st.session_state.history:
            st.markdown(f"**`{item['prompt_str']}`**")
            if item.get("stdout"):
                st.code(item["stdout"], language="bash")
            if item.get("stderr"):
                st.error(item["stderr"])

        # Extra Keys Bar (2 rows) - Avoid nesting st.columns inside st.columns!
        st.markdown("**Virtual Extra Keys:**")
        row1_keys = ["ESC", "TAB", "CTRL", "ALT", "-", "/", "|", "~", "CLEAR"]
        row2_keys = ["UP", "DOWN", "LEFT", "RIGHT", "HOME", "END", "PGUP", "PGDN"]

        cols_r1 = st.columns(len(row1_keys))
        for idx, k in enumerate(row1_keys):
            if cols_r1[idx].button(k, key=f"btn_r1_{k}"):
                if k == "CLEAR":
                    st.session_state.history = []
                    st.rerun()
                else:
                    st.session_state.cli_input_val += f" {k} "
                    st.rerun()

        cols_r2 = st.columns(len(row2_keys))
        for idx, k in enumerate(row2_keys):
            if cols_r2[idx].button(k, key=f"btn_r2_{k}"):
                st.session_state.cli_input_val += f" {k} "
                st.rerun()

        # Command Processing Helper
        def handle_command_submit():
            cmd = st.session_state.cli_input_val.strip()
            if cmd:
                prompt_str = f"session@termux:{st.session_state.cwd}$ {cmd}"
                stdout, stderr, code, new_cwd = execute_shell_command(cmd, st.session_state.cwd)
                st.session_state.cwd = new_cwd
                st.session_state.history.append({
                    "prompt_str": prompt_str,
                    "stdout": stdout,
                    "stderr": stderr,
                    "code": code
                })
            st.session_state.cli_input_val = ""

        # Check for pending sidebar quick command
        if "pending_cmd" in st.session_state and st.session_state.pending_cmd:
            cmd = st.session_state.pending_cmd
            st.session_state.pending_cmd = None
            prompt_str = f"session@termux:{st.session_state.cwd}$ {cmd}"
            stdout, stderr, code, new_cwd = execute_shell_command(cmd, st.session_state.cwd)
            st.session_state.cwd = new_cwd
            st.session_state.history.append({
                "prompt_str": prompt_str,
                "stdout": stdout,
                "stderr": stderr,
                "code": code
            })
            st.rerun()

        prompt_prefix = f"session@termux:{st.session_state.cwd}$"
        st.text_input(
            label=prompt_prefix,
            key="cli_input_val",
            on_change=handle_command_submit
        )

    # TAB 2: File Manager & Code Editor
    with tab_fm:
        st.subheader("📁 File Manager")
        fm_dir = st.text_input("Current Directory", value=st.session_state.cwd)
        if os.path.exists(fm_dir) and os.path.isdir(fm_dir):
            st.session_state.cwd = os.path.abspath(fm_dir)
            entries = os.listdir(st.session_state.cwd)
            files_data = []
            for e in sorted(entries):
                full_p = os.path.join(st.session_state.cwd, e)
                is_dir = os.path.isdir(full_p)
                # Formatted size as string to prevent PyArrow serialization issues
                size_str = "-" if is_dir else str(os.path.getsize(full_p))
                files_data.append({
                    "Name": e,
                    "Type": "Directory" if is_dir else "File",
                    "Size (bytes)": size_str
                })
            if files_data:
                df = pd.DataFrame(files_data)
                st.dataframe(df, use_container_width=True)

            st.divider()
            st.subheader("📝 File Editor / Viewer")
            selected_file = st.selectbox("Select File to View/Edit", options=["None"] + [e for e in sorted(entries) if os.path.isfile(os.path.join(st.session_state.cwd, e))])
            if selected_file and selected_file != "None":
                filepath = os.path.join(st.session_state.cwd, selected_file)
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        file_content = f.read()
                    edited_content = st.text_area(f"Editing: {selected_file}", value=file_content, height=300)
                    if st.button("Save Changes"):
                        with open(filepath, "w", encoding="utf-8") as f:
                            f.write(edited_content)
                        st.success(f"Successfully saved {selected_file}")
                except Exception as ex:
                    st.error(f"Error reading file: {ex}")
        else:
            st.error("Invalid directory path.")

    # TAB 3: System Resources
    with tab_sys:
        st.subheader("📊 System Performance Dashboard")
        if st.button("Refresh Metrics"):
            st.rerun()

        metrics = get_system_metrics()
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("CPU Usage", f"{metrics['cpu_percent']}%")
        col_m2.metric("Memory Usage", f"{metrics['memory_percent']}%", f"{metrics['memory_used_gb']} GB / {metrics['memory_total_gb']} GB")
        col_m3.metric("Disk Usage", f"{metrics['disk_percent']}%", f"{metrics['disk_used_gb']} GB / {metrics['disk_total_gb']} GB")

        st.subheader("⚙️ Running Processes")
        procs = get_process_list()
        if procs:
            df_procs = pd.DataFrame(procs)
            st.dataframe(df_procs, use_container_width=True)

if __name__ == "__main__":
    main()
