import os
from fastapi import FastAPI
import google.generativeai as genai

app = FastAPI()

# Kết nối API Key sếp đã nạp trên Render
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash-exp') # Bản Flash 2026 siêu tốc

@app.get("/")
def health():
    return {"status": "Live", "brain": "Gemini 2.5 Flash Online"}

@app.get("/ask")
def ask_agent(prompt: str):
    # Ép não tư duy theo phong cách sếp
    response = model.generate_content(f"Hành xử như Agent tnq. Suy nghĩ: {prompt}")
    return {"answer": response.text}
    
