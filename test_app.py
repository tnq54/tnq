import pytest
import io
import os
from PIL import Image
from app import extract_pdf_text, summarize_with_gemini, analyze_photo_with_gemini

def test_extract_pdf_text_invalid():
    # Test with invalid bytes
    result = extract_pdf_text(b"invalid pdf content")
    assert result is None

def test_summarize_with_gemini_without_key():
    # Test fallback when key is not configured or in testing environment
    result = summarize_with_gemini("Test content")
    assert isinstance(result, str)

def test_analyze_photo_with_gemini_fallback():
    # Create a small sample image
    img = Image.new('RGB', (100, 100), color='red')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    img_bytes = buf.getvalue()

    result = analyze_photo_with_gemini(img_bytes)
    assert isinstance(result, str)
