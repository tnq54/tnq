import os
import sys
import unittest
import tempfile
import shutil
from unittest.mock import MagicMock, patch

# Import backend helper functions from app.py
from app import (
    execute_shell_command,
    parse_pkg_command,
    get_system_metrics,
    get_process_list,
    is_port_in_use
)

class TestAppBackend(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_execute_shell_command_basic(self):
        stdout, stderr, code, cwd = execute_shell_command("echo Hello Termux", cwd=self.temp_dir)
        self.assertEqual(code, 0)
        self.assertIn("Hello Termux", stdout)
        self.assertEqual(stderr, "")

    def test_execute_shell_command_cd(self):
        sub_dir = os.path.join(self.temp_dir, "subdir")
        os.makedirs(sub_dir)
        stdout, stderr, code, new_cwd = execute_shell_command(f"cd {sub_dir}", cwd=self.temp_dir)
        self.assertEqual(code, 0)
        self.assertEqual(new_cwd, sub_dir)

    def test_execute_shell_command_invalid_cd(self):
        stdout, stderr, code, new_cwd = execute_shell_command("cd /non_existent_folder_12345", cwd=self.temp_dir)
        self.assertEqual(code, 1)
        self.assertIn("no such file or directory", stderr)

    def test_parse_pkg_command(self):
        cmd1 = parse_pkg_command("pkg install pandas -y")
        self.assertIn("pip install pandas", cmd1)

        cmd2 = parse_pkg_command("pkg uninstall numpy --yes")
        self.assertIn("pip uninstall -y numpy", cmd2)

        cmd3 = parse_pkg_command("pkg update")
        self.assertIn("pip list --outdated", cmd3)

        cmd4 = parse_pkg_command("pkg search requests")
        self.assertIn("pip list | grep -i 'requests'", cmd4)

    def test_termux_api_emulation(self):
        stdout, stderr, code, cwd = execute_shell_command("termux-info", cwd=self.temp_dir)
        self.assertEqual(code, 0)
        self.assertIn("Termux Emulation Environment", stdout)

        stdout, stderr, code, cwd = execute_shell_command("termux-battery-status", cwd=self.temp_dir)
        self.assertEqual(code, 0)
        self.assertIn("CHARGING", stdout)

        stdout, stderr, code, cwd = execute_shell_command("termux-toast 'Hello World'", cwd=self.temp_dir)
        self.assertEqual(code, 0)
        self.assertIn("[TOAST]: Hello World", stdout)

    def test_background_command(self):
        stdout, stderr, code, cwd = execute_shell_command("sleep 1 &", cwd=self.temp_dir)
        self.assertEqual(code, 0)
        self.assertIn("Background Process Started", stdout)

    def test_get_system_metrics(self):
        metrics = get_system_metrics()
        self.assertIn("cpu_percent", metrics)
        self.assertIn("memory_percent", metrics)
        self.assertIn("disk_percent", metrics)

    def test_get_process_list(self):
        procs = get_process_list()
        self.assertIsInstance(procs, list)
        if len(procs) > 0:
            self.assertIn("PID", procs[0])
            self.assertIn("Name", procs[0])

    def test_is_port_in_use(self):
        # Unused port test
        res = is_port_in_use(59999)
        self.assertIsInstance(res, bool)

if __name__ == "__main__":
    unittest.main()
