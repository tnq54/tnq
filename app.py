import streamlit as st
import os
import sys
import subprocess
import shutil
import platform
import time
import json
import psutil
import pandas as pd

# Theme Definitions
THEMES = {
    "Classic Green": {
        "bg": "#000000",
        "text": "#00ff66",
        "container_bg": "#0a0a0a",
        "border": "#00ff66",
        "prompt": "#00ff66"
    },
    "Monokai": {
        "bg": "#272822",
        "text": "#f8f8f2",
        "container_bg": "#1e1e1e",
        "border": "#a6e22e",
        "prompt": "#a6e22e"
    },
    "Dracula": {
        "bg": "#282a36",
        "text": "#f8f8f2",
        "container_bg": "#1e1f29",
        "border": "#ff79c6",
        "prompt": "#50fa7b"
    },
    "Solarized Dark": {
        "bg": "#002b36",
        "text": "#839496",
        "container_bg": "#073642",
        "border": "#2aa198",
        "prompt": "#b58900"
    },
    "Cyan Cyberpunk": {
        "bg": "#050814",
        "text": "#00f0ff",
        "container_bg": "#0a1128",
        "border": "#ff0055",
        "prompt": "#00f0ff"
    }
}


# Function to execute commands
def execute_shell_command(command: str, cwd: str = None) -> str:
    if cwd is None:
        if hasattr(st, "session_state") and "cwd" in st.session_state:
            cwd = st.session_state.cwd
        else:
            cwd = os.getcwd()

    command = command.strip()
    if not command:
        return ""

    # Handle 'cd' directory navigation
    if command == "cd" or command.startswith("cd "):
        parts = command.split(maxsplit=1)
        target_dir = parts[1].strip() if len(parts) > 1 else os.path.expanduser("~")
        if target_dir == "~":
            target_dir = os.path.expanduser("~")
        elif not os.path.isabs(target_dir):
            target_dir = os.path.abspath(os.path.join(cwd, target_dir))

        if os.path.exists(target_dir) and os.path.isdir(target_dir):
            if hasattr(st, "session_state") and "cwd" in st.session_state:
                st.session_state.cwd = target_dir
            return f"Changed directory to: {target_dir}"
        else:
            return f"cd: no such file or directory: {parts[1] if len(parts) > 1 else '~'}"

    # Handle Termux 'pkg' emulation
    if command == "pkg" or command.startswith("pkg "):
        args = command.split()[1:]
        if not args or args[0] in ["help", "-h", "--help"]:
            return (
                "Termux Package Manager (pkg Emulator)\n\n"
                "Usage: pkg <command> [args]\n\n"
                "Commands:\n"
                "  install <pkg>   Install Python package via pip\n"
                "  uninstall <pkg> Uninstall Python package via pip\n"
                "  update / upgrade Update/List packages\n"
                "  search <pkg>    Search installed packages\n"
                "  list-installed  List all installed packages\n"
                "  help            Show this help text"
            )
        subcmd = args[0]
        if subcmd == "install" and len(args) > 1:
            pkg_name = " ".join(args[1:])
            res = subprocess.run([sys.executable, "-m", "pip", "install", pkg_name], capture_output=True, text=True, timeout=60)
            return res.stdout + ("\n" + res.stderr if res.stderr else "")
        elif subcmd == "uninstall" and len(args) > 1:
            pkg_name = " ".join(args[1:])
            res = subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", pkg_name], capture_output=True, text=True, timeout=60)
            return res.stdout + ("\n" + res.stderr if res.stderr else "")
        elif subcmd in ["update", "upgrade"]:
            res = subprocess.run([sys.executable, "-m", "pip", "list", "--outdated"], capture_output=True, text=True, timeout=30)
            return "Checking for outdated packages:\n" + res.stdout
        elif subcmd == "search" and len(args) > 1:
            query = args[1]
            res = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True, timeout=30)
            matches = [line for line in res.stdout.splitlines() if query.lower() in line.lower()]
            return "\n".join(matches) if matches else f"No packages matching '{query}' found."
        elif subcmd == "list-installed":
            res = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True, timeout=30)
            return res.stdout
        else:
            return f"pkg: unknown command '{subcmd}'. Type 'pkg help' for usage."

    # Handle termux-* API commands
    if command.startswith("termux-"):
        if command == "termux-info":
            info = {
                "Platform": platform.platform(),
                "System": platform.system(),
                "Release": platform.release(),
                "Architecture": platform.machine(),
                "Python Version": platform.python_version(),
                "Termux Workspace": "Active (Hugging Face Spaces)",
                "Host Node": platform.node()
            }
            return json.dumps(info, indent=2)
        elif command == "termux-setup-storage":
            storage_dir = os.path.expanduser("~/storage")
            subdirs = ["dcim", "downloads", "movies", "music", "pictures", "shared"]
            os.makedirs(storage_dir, exist_ok=True)
            for sd in subdirs:
                os.makedirs(os.path.join(storage_dir, sd), exist_ok=True)
            return f"Simulating storage setup: Symlinks created in {storage_dir} ({', '.join(subdirs)})."
        elif command == "termux-battery-status":
            battery = psutil.sensors_battery()
            if battery:
                return json.dumps({
                    "health": "GOOD",
                    "percentage": int(battery.percent),
                    "plugged": "PLUGGED_AC" if battery.power_plugged else "UNPLUGGED",
                    "status": "CHARGING" if battery.power_plugged else "DISCHARGING"
                }, indent=2)
            else:
                return json.dumps({
                    "health": "GOOD",
                    "percentage": 100,
                    "plugged": "PLUGGED_AC",
                    "status": "CHARGING"
                }, indent=2)
        elif command.startswith("termux-toast"):
            msg = command[12:].strip() or "Notification Toast"
            return f"🔔 [Toast Notification]: {msg}"
        elif command == "termux-clipboard-get":
            clip = st.session_state.clipboard if hasattr(st, "session_state") and "clipboard" in st.session_state else ""
            return clip if clip else "(clipboard empty)"
        elif command.startswith("termux-clipboard-set"):
            new_clip = command[20:].strip()
            if hasattr(st, "session_state"):
                st.session_state.clipboard = new_clip
            return "Clipboard updated."
        else:
            return f"{command}: termux API simulated."

    # Handle general shell commands
    try:
        res = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True, timeout=30)
        output = res.stdout
        if res.stderr:
            output += ("\n" if output else "") + res.stderr
        if not output and res.returncode == 0:
            output = "(Command executed successfully with no output)"
        return output
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds."
    except Exception as e:
        return f"Error executing command: {e}"


# System Resource Helper Functions
def get_system_metrics():
    cpu_pct = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    boot_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(psutil.boot_time()))
    return {
        "cpu_percent": cpu_pct,
        "memory_percent": mem.percent,
        "memory_used_gb": round(mem.used / (1024**3), 2),
        "memory_total_gb": round(mem.total / (1024**3), 2),
        "disk_percent": disk.percent,
        "disk_used_gb": round(disk.used / (1024**3), 2),
        "disk_total_gb": round(disk.total / (1024**3), 2),
        "boot_time": boot_time
    }


def get_process_list():
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 'memory_percent', 'status']):
        try:
            pinfo = proc.info
            processes.append({
                "PID": pinfo['pid'],
                "Name": pinfo['name'],
                "User": pinfo['username'],
                "CPU %": round(pinfo['cpu_percent'] or 0.0, 1),
                "Mem %": round(pinfo['memory_percent'] or 0.0, 1),
                "Status": pinfo['status']
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return sorted(processes, key=lambda x: x['CPU %'], reverse=True)[:25]


def main():
    # Page Config MUST be the very first Streamlit call
    st.set_page_config(
        page_title="Termux Web Workspace",
        page_icon="📱",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize Session State
    if "cwd" not in st.session_state:
        st.session_state.cwd = os.getcwd()

    if "command_history" not in st.session_state:
        st.session_state.command_history = [
            ("system", "Welcome to Termux Web Workspace on Hugging Face Spaces!\nType 'help' or 'pkg help' for available commands.")
        ]

    if "clipboard" not in st.session_state:
        st.session_state.clipboard = ""

    if "command_input" not in st.session_state:
        st.session_state.command_input = ""

    if "theme" not in st.session_state:
        st.session_state.theme = "Classic Green"

    current_theme = THEMES.get(st.session_state.theme, THEMES["Classic Green"])

    # Custom CSS for Termux styling
    st.markdown(f"""
    <style>
        .stApp {{
            background-color: {current_theme['bg']};
            color: {current_theme['text']};
        }}
        .termux-banner {{
            font-family: 'Courier New', Courier, monospace;
            color: {current_theme['prompt']};
            white-space: pre;
            font-weight: bold;
            line-height: 1.2;
        }}
        .terminal-box {{
            background-color: {current_theme['container_bg']};
            border: 1px solid {current_theme['border']};
            border-radius: 5px;
            padding: 15px;
            font-family: 'Courier New', Courier, monospace;
            color: {current_theme['text']};
            height: 400px;
            overflow-y: auto;
            margin-bottom: 15px;
        }}
        .terminal-prompt {{
            color: {current_theme['prompt']};
            font-weight: bold;
        }}
        .extra-key-btn button {{
            width: 100%;
            background-color: {current_theme['container_bg']};
            color: {current_theme['text']};
            border: 1px solid {current_theme['border']};
            font-family: monospace;
            margin-bottom: 5px;
        }}
    </style>
    """, unsafe_allow_html=True)

    # Sidebar Configuration
    with st.sidebar:
        st.title("⚙️ Workspace Config")

        selected_theme = st.selectbox("Color Theme", list(THEMES.keys()), index=list(THEMES.keys()).index(st.session_state.theme))
        if selected_theme != st.session_state.theme:
            st.session_state.theme = selected_theme
            st.rerun()

        st.markdown("---")
        st.markdown(f"**Current CWD:**\n`{st.session_state.cwd}`")

        if st.button("📁 Reset to Root Directory"):
            st.session_state.cwd = os.getcwd()
            st.rerun()

        st.markdown("---")
        st.subheader("⚡ Termux Shortcuts")
        if st.button("🚀 termux-info"):
            out = execute_shell_command("termux-info")
            st.session_state.command_history.append(("termux-info", out))
            st.rerun()
        if st.button("💾 termux-setup-storage"):
            out = execute_shell_command("termux-setup-storage")
            st.session_state.command_history.append(("termux-setup-storage", out))
            st.rerun()
        if st.button("🔋 termux-battery-status"):
            out = execute_shell_command("termux-battery-status")
            st.session_state.command_history.append(("termux-battery-status", out))
            st.rerun()

        if st.button("🧹 Clear Terminal Output"):
            st.session_state.command_history = []
            st.rerun()

    # Welcome Banner
    st.markdown(f"""
    <div class="termux-banner">
      #####  ######  #####  # # #   # #     # # #
        #    #      #     # # # #   # #     # # #
        #    #####  ######  # #  # #  #  #  # # #
        #    #      #   #   # #   #   # # # # # #
        #    ###### #    #  # #       ##   ## # #
    </div>
    """, unsafe_allow_html=True)
    st.caption("Termux Web Workspace • Linux Environment on Hugging Face Spaces")

    # Main Interface Tabs
    tab_terminal, tab_files, tab_system = st.tabs(["💻 Terminal / CLI", "📁 File Manager & Editor", "📊 System Resources"])

    # --- TAB 1: TERMINAL / CLI ---
    with tab_terminal:
        st.subheader("Interactive Terminal")

        # Display Terminal History
        terminal_content = ""
        for cmd, out in st.session_state.command_history:
            if cmd == "system":
                terminal_content += f"{out}\n\n"
            else:
                prompt_str = f"session@termux:{st.session_state.cwd}$ {cmd}"
                terminal_content += f"{prompt_str}\n{out}\n\n"

        st.code(terminal_content if terminal_content else "Terminal ready...", language="bash")

        # Virtual Extra Keys Bar (2 Rows)
        st.markdown("**Extra Keys:**")

        # Row 1: ESC, TAB, CTRL, ALT, -, /, |, ~
        r1_col1, r1_col2, r1_col3, r1_col4, r1_col5, r1_col6, r1_col7, r1_col8 = st.columns(8)
        if r1_col1.button("ESC", key="k_esc"):
            st.session_state.command_input += ""
        if r1_col2.button("TAB", key="k_tab"):
            st.session_state.command_input += "  "
        if r1_col3.button("CTRL", key="k_ctrl"):
            st.toast("CTRL key pressed")
        if r1_col4.button("ALT", key="k_alt"):
            st.toast("ALT key pressed")
        if r1_col5.button("-", key="k_dash"):
            st.session_state.command_input += "-"
        if r1_col6.button("/", key="k_slash"):
            st.session_state.command_input += "/"
        if r1_col7.button("|", key="k_pipe"):
            st.session_state.command_input += "|"
        if r1_col8.button("~", key="k_tilde"):
            st.session_state.command_input += "~"

        # Row 2: CLEAR, UP, DOWN, LEFT, RIGHT, HOME, END, PGUP
        r2_col1, r2_col2, r2_col3, r2_col4, r2_col5, r2_col6, r2_col7, r2_col8 = st.columns(8)
        if r2_col1.button("CLEAR", key="k_clear"):
            st.session_state.command_history = []
            st.rerun()
        if r2_col2.button("UP", key="k_up"):
            if st.session_state.command_history:
                last_cmd = [c for c, _ in st.session_state.command_history if c != "system"]
                if last_cmd:
                    st.session_state.command_input = last_cmd[-1]
        if r2_col3.button("DOWN", key="k_down"):
            st.session_state.command_input = ""
        if r2_col4.button("LEFT", key="k_left"):
            pass
        if r2_col5.button("RIGHT", key="k_right"):
            pass
        if r2_col6.button("HOME", key="k_home"):
            st.session_state.command_input = "cd ~"
        if r2_col7.button("END", key="k_end"):
            pass
        if r2_col8.button("PGUP", key="k_pgup"):
            pass

        # Command Input Form
        with st.form(key="terminal_form", clear_on_submit=True):
            user_command = st.text_input(
                f"session@termux:{st.session_state.cwd}$",
                value=st.session_state.command_input,
                key="input_box"
            )
            submit_btn = st.form_submit_button("Run Command ↵")

            if submit_btn and user_command.strip():
                output = execute_shell_command(user_command)
                st.session_state.command_history.append((user_command, output))
                st.session_state.command_input = ""
                st.rerun()

    # --- TAB 2: FILE MANAGER & CODE EDITOR ---
    with tab_files:
        st.subheader("File Manager & Code Editor")

        col_nav, col_actions = st.columns([3, 1])
        with col_nav:
            st.markdown(f"**Current Directory:** `{st.session_state.cwd}`")
        with col_actions:
            if st.button("⬆️ Go Up Directory"):
                parent_dir = os.path.dirname(st.session_state.cwd)
                if parent_dir and os.path.exists(parent_dir):
                    st.session_state.cwd = parent_dir
                    st.rerun()

        # List items in directory
        try:
            items = os.listdir(st.session_state.cwd)
            file_data = []
            for item in sorted(items):
                item_path = os.path.join(st.session_state.cwd, item)
                is_dir = os.path.isdir(item_path)
                try:
                    stat = os.stat(item_path)
                    # Convert size to string to prevent PyArrow serialization errors in Streamlit
                    size_str = "<DIR>" if is_dir else f"{stat.st_size} B"
                    mod_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime))
                except Exception:
                    size_str = "<UNKNOWN>"
                    mod_time = "<UNKNOWN>"

                file_data.append({
                    "Name": ("📁 " if is_dir else "📄 ") + item,
                    "Type": "Directory" if is_dir else "File",
                    "Size": size_str,
                    "Modified": mod_time
                })

            df_files = pd.DataFrame(file_data)
            st.dataframe(df_files, use_container_width=True)

            # File Selection & Navigation / Editing
            file_list = [item for item in items if os.path.isfile(os.path.join(st.session_state.cwd, item))]
            dir_list = [item for item in items if os.path.isdir(os.path.join(st.session_state.cwd, item))]

            col_dir_select, col_file_select = st.columns(2)
            with col_dir_select:
                selected_dir = st.selectbox("Open Subdirectory", ["-- Select Directory --"] + sorted(dir_list))
                if selected_dir != "-- Select Directory --":
                    st.session_state.cwd = os.path.join(st.session_state.cwd, selected_dir)
                    st.rerun()

            with col_file_select:
                selected_file = st.selectbox("Select File to View / Edit", ["-- Select File --"] + sorted(file_list))

            if selected_file and selected_file != "-- Select File --":
                target_file_path = os.path.join(st.session_state.cwd, selected_file)
                st.markdown(f"### Editing: `{selected_file}`")
                try:
                    with open(target_file_path, "r", encoding="utf-8", errors="replace") as f:
                        file_content = f.read()

                    edited_content = st.text_area("File Content", value=file_content, height=350)
                    if st.button("💾 Save Changes"):
                        with open(target_file_path, "w", encoding="utf-8") as f:
                            f.write(edited_content)
                        st.success(f"File `{selected_file}` saved successfully.")
                except Exception as e:
                    st.error(f"Error reading file: {e}")

            # File Creation Controls
            st.markdown("---")
            st.subheader("Create File / Directory")
            c_name, c_type, c_btn = st.columns([3, 2, 1])
            new_item_name = c_name.text_input("Name", key="new_item_name")
            new_item_type = c_type.selectbox("Type", ["File", "Directory"])
            if c_btn.button("Create"):
                if new_item_name:
                    target_path = os.path.join(st.session_state.cwd, new_item_name)
                    try:
                        if new_item_type == "Directory":
                            os.makedirs(target_path, exist_ok=True)
                            st.success(f"Directory `{new_item_name}` created.")
                        else:
                            with open(target_path, "w") as f:
                                f.write("")
                            st.success(f"File `{new_item_name}` created.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error creating {new_item_type.lower()}: {e}")

        except Exception as e:
            st.error(f"Error listing directory contents: {e}")

    # --- TAB 3: SYSTEM RESOURCES ---
    with tab_system:
        st.subheader("System Metrics & Resource Dashboard")

        metrics = get_system_metrics()

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("CPU Usage", f"{metrics['cpu_percent']}%")
        m_col2.metric("Memory Usage", f"{metrics['memory_percent']}%", f"{metrics['memory_used_gb']} / {metrics['memory_total_gb']} GB")
        m_col3.metric("Disk Usage", f"{metrics['disk_percent']}%", f"{metrics['disk_used_gb']} / {metrics['disk_total_gb']} GB")
        m_col4.metric("System Boot", metrics['boot_time'].split()[1])

        st.markdown("---")
        st.subheader("Active System Processes (Top CPU)")
        processes = get_process_list()
        df_proc = pd.DataFrame(processes)
        st.dataframe(df_proc, use_container_width=True)


if __name__ == "__main__":
    main()
