import threading
import os
from fastapi import FastAPI # Hoặc Flask tùy bạn dùng cái nào

app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "ok"}

def run_bot():
    # Giữ nguyên lệnh chạy bot cũ của bạn ở đây
    bot.polling(none_stop=True)

if __name__ == "__main__":
    # Chạy bot ở một luồng riêng để không làm nghẽn Web Server
    threading.Thread(target=run_bot).start()
    
    # Chạy Web Server để trả lời Render
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
    
