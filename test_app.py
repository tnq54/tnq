import pytest
import os
import shutil
from app import execute_shell_command, extract_pdf_text, summarize_with_gemini

def test_execute_shell_command_basic():
    output = execute_shell_command("echo Hello HuggingFace")
    assert "Hello HuggingFace" in output

def test_execute_shell_command_clear():
    assert execute_shell_command("clear") == "__CLEAR__"
    assert execute_shell_command("cls") == "__CLEAR__"

def test_execute_shell_command_restricted():
    output = execute_shell_command("rm -rf /")
    assert "restricted for security" in output

def test_execute_shell_command_chaining_restricted():
    output = execute_shell_command("echo hello && whoami")
    assert "restricted for security" in output

def test_execute_shell_command_mkdir():
    test_dir = "test_lab_dir"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    res = execute_shell_command(f"mkdir {test_dir}")
    assert "Directory created" in res
    assert os.path.exists(test_dir)

    os.rmdir(test_dir)

def test_extract_pdf_text_invalid():
    res = extract_pdf_text(b"invalid pdf data")
    assert res is None

def test_summarize_with_gemini_missing_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    import app
    app.GOOGLE_API_KEY = None
    res = summarize_with_gemini("Sample text")
    assert "Error: GOOGLE_API_KEY not found." in res
