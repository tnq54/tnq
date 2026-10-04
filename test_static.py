import os
import re

def test_index_html_exists():
    assert os.path.exists("index.html"), "index.html file must exist"

def test_index_html_structure():
    with open("index.html", "r", encoding="utf-8") as f:
        content = f.read()

    # Check key DOM elements
    assert "id=\"terminal-container\"" in content
    assert "id=\"output-area\"" in content
    assert "id=\"cmd-input\"" in content
    assert "id=\"extra-keys\"" in content
    assert "id=\"matrix-canvas\"" in content

    # Check key JS features & commands
    assert "handleCommand" in content
    assert "handlePkg" in content
    assert "neofetch" in content
    assert "runMatrix" in content
    assert "handleTheme" in content

def test_readme_static_config():
    assert os.path.exists("README.md"), "README.md file must exist"
    with open("README.md", "r", encoding="utf-8") as f:
        content = f.read()

    assert "sdk: static" in content
    assert "title: Termux Static" in content
