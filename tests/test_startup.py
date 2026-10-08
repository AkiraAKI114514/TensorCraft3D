"""Smoke the actual launcher against an already-built frontend, without GPU probes."""
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import contextmanager
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent.parent
WINDOWS = sys.platform == "win32"


class StartupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        index = ROOT / "dist" / "index.html"
        if not index.is_file():
            raise AssertionError("Frontend build missing. Run npm run build before startup tests.")
        if re.search(r'(?:src|href)="(/assets/[^\"]+)"', index.read_text(encoding="utf-8")) is None:
            raise AssertionError("Built index.html does not reference a frontend asset.")

    def require_api(self):
        for package in ("fastapi", "uvicorn"):
            self.assertIsNotNone(importlib.util.find_spec(package), f"Missing startup test dependency: {package}")

    def free_port(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    @contextmanager
    def server(self, command, mode, port, timeout=90):
        environment = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "CUDA_VISIBLE_DEVICES": "-1"}
        with tempfile.TemporaryDirectory(prefix="tensorcraft-startup-") as directory:
            log_path = Path(directory) / "server.log"
            with log_path.open("wb") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
                try:
                    deadline = time.monotonic() + timeout
                    base_url = None
                    while time.monotonic() < deadline:
                        output = log_path.read_text(encoding="utf-8", errors="replace")
                        match = re.search(r"TensorCraft3D .*: (http://127\.0\.0\.1:\d+)", output)
                        if match and mode in output:
                            base_url = match.group(1)
                            try:
                                with urlopen(base_url, timeout=2) as response:
                                    self.assertEqual(response.status, 200)
                                    index = (ROOT / "dist" / "index.html").read_bytes()
                                    self.assertEqual(response.read(), index)
                                break
                            except (URLError, TimeoutError):
                                pass
                        if process.poll() is not None:
                            self.fail(f"Launcher exited before readiness (code {process.returncode}).")
                        time.sleep(0.1)
                    else:
                        self.fail(f"Launcher was not ready within {timeout}s.")
                    if port is not None:
                        self.assertEqual(int(base_url.rsplit(":", 1)[1]), port)
                    match = re.search(r'(?:src|href)="(/assets/[^\"]+)"', index.decode("utf-8"))
                    self.assertIsNotNone(match, "Built index.html does not reference a frontend asset.")
                    asset_path = match.group(1)
                    asset = (ROOT / "dist" / asset_path.lstrip("/")).read_bytes()
                    with urlopen(base_url + asset_path, timeout=10) as response:
                        self.assertEqual(response.status, 200)
                        self.assertEqual(response.read(), asset)
                    yield base_url
                except Exception as error:
                    output = log_path.read_text(encoding="utf-8", errors="replace")
                    raise AssertionError(f"{error}\nLauncher output:\n{output}") from error
                finally:
                    if process.poll() is None:
                        if WINDOWS:
                            result = subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, text=True, timeout=30)
                            if result.returncode and process.poll() is None:
                                raise AssertionError(f"Could not stop launcher tree {process.pid}: {result.stdout}\n{result.stderr}")
                        else:
                            process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)

    def check_health(self, base_url):
        with urlopen(base_url + "/api/health", timeout=30) as response:
            self.assertEqual(response.status, 200)
            health = json.load(response)
        self.assertTrue(health["pytorchImport"])
        self.assertTrue(health["tensorInference"])
        self.assertFalse(health["cuda"])

    def test_stdlib_launcher_serves_built_frontend(self):
        port = self.free_port()
        with self.server([sys.executable, "-S", "run.py", "--no-browser", "--port", str(port)], "Browser-only mode", port) as base_url:
            with self.assertRaises(HTTPError) as error:
                urlopen(base_url + "/api/health", timeout=10)
            self.assertEqual(error.exception.code, 404)

    @unittest.skipUnless(WINDOWS, "PowerShell startup is Windows-only")
    def test_powershell_launcher_serves_api_and_frontend(self):
        self.require_api()
        port = self.free_port()
        command = ["powershell.exe", "-NoProfile", "-NonInteractive", "-File", str(ROOT / "start.ps1"), "-NoBrowser", "-Port", str(port)]
        with self.server(command, "Python API enabled", port, timeout=180) as base_url:
            self.check_health(base_url)

    @unittest.skipUnless(WINDOWS, "CMD startup is Windows-only")
    def test_cmd_launcher_serves_api_and_frontend(self):
        self.require_api()
        port = self.free_port()
        command = ["cmd.exe", "/d", "/c", ".\\start.cmd", "--no-browser", "--port", str(port)]
        with self.server(command, "Python API enabled", port) as base_url:
            self.check_health(base_url)

    def test_api_launcher_falls_back_from_occupied_port(self):
        self.require_api()
        for _ in range(10):
            with socket.socket() as occupied:
                occupied.bind(("127.0.0.1", 0))
                port = occupied.getsockname()[1]
                if port > 65515:
                    continue
                occupied.listen()
                with self.server([sys.executable, "run.py", "--no-browser", "--port", str(port)], "Python API enabled", None) as base_url:
                    actual_port = int(base_url.rsplit(":", 1)[1])
                    self.assertGreater(actual_port, port)
                    self.assertLess(actual_port, port + 20)
                    self.check_health(base_url)
                return
        self.fail("Could not reserve an ephemeral port with room for launcher fallback.")


if __name__ == "__main__":
    unittest.main()
