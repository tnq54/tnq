from fastapi import FastAPI
from pydantic import BaseModel
import os

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Brain is Online", "model": "Gemini 2.5 Flash"}

@app.post("/chat")
def chat(query: str):
    # Logic kết nối Gemini 2.5 Flash sẽ nằm ở đây
    return {"thinking": "Logic GRPO đang được nạp...", "response": "Chào sếp!"}
