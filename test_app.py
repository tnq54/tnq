import os
import pytest
import pandas as pd
from app import execute_shell_command, get_system_metrics, get_process_list

def test_execute_shell_command_basic():
    stdout, stderr, code = execute_shell_command("echo 'Hello Termux'", cwd=os.getcwd())
    assert code == 0
    assert "Hello Termux" in stdout
    assert stderr == ""

def test_execute_shell_command_termux_info():
    stdout, stderr, code = execute_shell_command("termux-info", cwd=os.getcwd())
    assert code == 0
    assert "TERMUX_VERSION" in stdout

def test_execute_shell_command_termux_battery():
    stdout, stderr, code = execute_shell_command("termux-battery-status", cwd=os.getcwd())
    assert code == 0
    assert "percentage" in stdout

def test_execute_shell_command_termux_toast():
    stdout, stderr, code = execute_shell_command("termux-toast Test Notification", cwd=os.getcwd())
    assert code == 0
    assert "Toast Notification" in stdout

def test_execute_shell_command_termux_clipboard():
    execute_shell_command("termux-clipboard-set my_secret_key", cwd=os.getcwd())
    stdout, stderr, code = execute_shell_command("termux-clipboard-get", cwd=os.getcwd())
    assert code == 0
    assert "my_secret_key" in stdout

def test_execute_shell_command_termux_storage(tmp_path):
    stdout, stderr, code = execute_shell_command("termux-setup-storage", cwd=str(tmp_path))
    assert code == 0
    assert "Creating storage directory" in stdout
    assert (tmp_path / "storage").exists()

def test_execute_shell_command_pkg_help():
    stdout, stderr, code = execute_shell_command("pkg help", cwd=os.getcwd())
    assert code == 0
    assert "Termux package manager" in stdout

def test_execute_shell_command_invalid():
    stdout, stderr, code = execute_shell_command("invalid_command_xyz_123", cwd=os.getcwd())
    assert code != 0
    assert stderr != ""

def test_execute_shell_command_cd(tmp_path):
    sub_dir = tmp_path / "test_dir"
    sub_dir.mkdir()

    stdout, stderr, code = execute_shell_command(f"cd {sub_dir}", cwd=str(tmp_path))
    assert code == 0
    assert "Changed directory to" in stdout

def test_get_system_metrics():
    metrics = get_system_metrics()
    assert isinstance(metrics, dict)
    assert "cpu_percent" in metrics
    assert "mem_percent" in metrics
    assert "disk_percent" in metrics
    assert metrics["mem_total_gb"] > 0

def test_get_process_list():
    df = get_process_list()
    assert isinstance(df, pd.DataFrame)
