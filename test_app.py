import os
import pytest
from app import execute_shell_command, get_docker_status, run_docker_command, get_system_info, extract_pdf_text, summarize_with_gemini

def test_execute_shell_command_basic():
    res = execute_shell_command("echo 'Hello Linux CLI'")
    assert res["returncode"] == 0
    assert "Hello Linux CLI" in res["stdout"]
    assert res["stderr"] == ""

def test_execute_shell_command_cd(tmp_path):
    target_dir = str(tmp_path)
    res = execute_shell_command(f"cd {target_dir}")
    assert res["returncode"] == 0
    assert res["cwd"] == target_dir
    assert "Changed directory to" in res["stdout"]

def test_execute_shell_command_cd_invalid():
    res = execute_shell_command("cd /non_existent_folder_12345")
    assert res["returncode"] == 1
    assert "No such file or directory" in res["stderr"]

def test_execute_shell_command_timeout():
    res = execute_shell_command("sleep 5", timeout=1)
    assert res["returncode"] == 124
    assert "timed out" in res["stderr"]

def test_get_docker_status():
    status = get_docker_status()
    assert "installed" in status
    assert "version" in status
    assert "daemon_ready" in status
    assert status["installed"] is True

def test_run_docker_command():
    res = run_docker_command("docker --version")
    assert res["returncode"] == 0
    assert "Docker version" in res["stdout"]

def test_get_system_info():
    info = get_system_info()
    assert "os" in info
    assert "python_version" in info
    assert "user" in info
    assert "disk_free_gb" in info

def test_summarize_with_gemini_missing_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    import app
    app.GOOGLE_API_KEY = None
    res = summarize_with_gemini("Test text")
    assert "Error: GOOGLE_API_KEY not found" in res
