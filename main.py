import os
import sys
import socket

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

if __name__ == "__main__":
    port_str = os.environ.get("PORT", "7860")
    try:
        port = int(port_str)
    except ValueError:
        port = 7860

    if is_port_in_use(port):
        print(f"Port {port} is already in use. Exiting gracefully.")
        sys.exit(0)

    cmd = f"{sys.executable} -m streamlit run app.py --server.port {port} --server.address 0.0.0.0"
    print(f"Launching Streamlit: {cmd}")
    exit_code = os.system(cmd)
    sys.exit(exit_code)
