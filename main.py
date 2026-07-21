import os
import sys

if __name__ == "__main__":
    # Get port from environment or default to 7860
    port = os.environ.get("PORT", "7860")

    # Launch Streamlit pointing to app.py
    # address=0.0.0.0 is crucial for Render
    cmd = f"{sys.executable} -m streamlit run app.py --server.port {port} --server.address 0.0.0.0"

    print(f"Launching Streamlit: {cmd}")
    exit_code = os.system(cmd)
    sys.exit(exit_code)
