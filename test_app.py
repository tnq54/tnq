import os
import pytest
from app import execute_shell_command, extract_pdf_text, summarize_with_gemini

def test_execute_shell_command(tmp_path):
    test_dir = str(tmp_path / "workspace")
    os.makedirs(test_dir, exist_ok=True)

    # Test simple echo
    res = execute_shell_command("echo Hello Pi", workspace_dir=test_dir)
    assert res == "Hello Pi"

    # Test clear signal
    res_clear = execute_shell_command("clear", workspace_dir=test_dir)
    assert res_clear == "CLEAR_SIGNAL"

    # Test directory creation and listing
    execute_shell_command("mkdir test_folder", workspace_dir=test_dir)
    res_ls = execute_shell_command("ls", workspace_dir=test_dir)
    assert "test_folder" in res_ls

def test_extract_pdf_text_empty():
    res = extract_pdf_text(b"invalid_pdf_data")
    assert res is None

def test_summarize_with_gemini_no_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    # Re-import or set app module level check
    import app
    app.GOOGLE_API_KEY = None
    res = summarize_with_gemini("test text")
    assert "Error: GOOGLE_API_KEY not found." in res
