import os
import sys
import socket


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


if __name__ == "__main__":
    # Get port from environment or default to 7860 (Hugging Face default)
    port_str = os.environ.get("PORT", "7860")
    try:
        port = int(port_str)
    except ValueError:
        port = 7860

    if is_port_in_use(port):
        print(f"Port {port} is already in use. Streamlit app is already running.")
        sys.exit(0)

    # Launch Streamlit pointing to app.py
    cmd = f"{sys.executable} -m streamlit run app.py --server.port {port} --server.address 0.0.0.0"

    print(f"Launching Streamlit: {cmd}")
    exit_code = os.system(cmd)
    sys.exit(exit_code)
