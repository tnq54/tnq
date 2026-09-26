import os
import json
import pytest
import streamlit as st
from unittest.mock import MagicMock

# Set up mock session state before importing functions if needed
if "cwd" not in st.session_state:
    st.session_state["cwd"] = os.getcwd()
if "clipboard" not in st.session_state:
    st.session_state["clipboard"] = ""

from app import execute_shell_command, get_system_metrics, get_process_list


def test_execute_shell_command_basic():
    output = execute_shell_command("echo Hello World", cwd=os.getcwd())
    assert "Hello World" in output


def test_execute_shell_command_cd():
    initial_cwd = st.session_state.cwd
    # Test cd to parent directory
    parent_dir = os.path.dirname(initial_cwd)
    output = execute_shell_command(f"cd {parent_dir}")
    assert "Changed directory to:" in output
    assert st.session_state.cwd == parent_dir

    # Restore directory
    execute_shell_command(f"cd {initial_cwd}")
    assert st.session_state.cwd == initial_cwd


def test_execute_shell_command_pkg_help():
    output = execute_shell_command("pkg help")
    assert "Termux Package Manager" in output
    assert "Usage: pkg" in output


def test_execute_shell_command_pkg_search():
    output = execute_shell_command("pkg search pytest")
    assert "pytest" in output.lower()


def test_execute_shell_command_pkg_install_with_flags():
    output = execute_shell_command("pkg install pytest -y")
    assert "Requirement already satisfied" in output or "Successfully installed" in output or "pytest" in output


def test_execute_shell_command_background():
    output = execute_shell_command("sleep 5 &")
    assert "Started background process PID" in output


def test_execute_shell_command_termux_info():
    output = execute_shell_command("termux-info")
    info = json.loads(output)
    assert "Platform" in info
    assert "Termux Workspace" in info


def test_execute_shell_command_termux_battery():
    output = execute_shell_command("termux-battery-status")
    battery = json.loads(output)
    assert "health" in battery
    assert "percentage" in battery


def test_execute_shell_command_termux_clipboard():
    set_output = execute_shell_command("termux-clipboard-set hello_termux")
    assert set_output == "Clipboard updated."

    get_output = execute_shell_command("termux-clipboard-get")
    assert get_output == "hello_termux"


def test_get_system_metrics():
    metrics = get_system_metrics()
    assert "cpu_percent" in metrics
    assert "memory_percent" in metrics
    assert "disk_percent" in metrics
    assert "boot_time" in metrics


def test_get_process_list():
    procs = get_process_list()
    assert isinstance(procs, list)
    if len(procs) > 0:
        assert "PID" in procs[0]
        assert "Name" in procs[0]
