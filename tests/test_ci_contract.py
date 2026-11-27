"""Exercise the actual report/pin validation entry points, even under python -O."""
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as E
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools/verify-model-free-report.py"
PIN = ROOT / "tools/ci-engine-pin.py"

class Contract(unittest.TestCase):
    def test_report_rejects_incomplete_duplicate_and_failed_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.xml"
            required = (ROOT / "tests/model-free-required.txt").read_text().splitlines()
            def run(names, status=None):
                suite = E.Element("testsuite")
                for name in names:
                    case = E.SubElement(suite, "testcase", name=name, status="run")
                    if status and name == required[0]:
                        if status == "notrun":
                            case.set("status", status)
                        else:
                            E.SubElement(case, status)
                E.ElementTree(suite).write(report)
                return subprocess.run([sys.executable, "-O", str(VALIDATOR), str(report), "44"],
                                      capture_output=True, text=True)
            self.assertEqual(run(required).returncode, 0)
            self.assertNotEqual(run(required[:-1]+["irrelevant"]).returncode, 0)
            self.assertNotEqual(run(required+[required[0]]).returncode, 0)
            for outcome in ("failure", "error", "skipped", "notrun"):
                self.assertNotEqual(run(required, outcome).returncode, 0)
    def test_modern_layout_preserves_runtime_and_rejects_collision(self):
        script = ROOT / "tools/stage-ci-layout.py"
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "module"
            bin_dir = package / "bin/64bit"
            bin_dir.mkdir(parents=True)
            for name in ("live-latent-lab-sight.so", "libonnxruntime.so.1.17.1"):
                (bin_dir / name).write_bytes(name.encode())
            data = package / "data"
            data.mkdir()
            (data / "locale.ini").write_bytes(b"data")
            cmd = [sys.executable, str(script), str(package), "modern"]
            (package / "live-latent-lab-sight.so").write_bytes(b"collision")
            result = subprocess.run(cmd, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue((bin_dir / "live-latent-lab-sight.so").exists())
            (package / "live-latent-lab-sight.so").unlink()
            result = subprocess.run(cmd, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((package / "bin").exists())
            self.assertEqual((package / "live-latent-lab-sight.so").read_bytes(), b"live-latent-lab-sight.so")
            self.assertEqual((package / "libonnxruntime.so.1.17.1").read_bytes(), b"libonnxruntime.so.1.17.1")
            self.assertEqual((data / "locale.ini").read_bytes(), b"data")
    def test_engine_pin_from_workflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            workflow = Path(tmp) / "workflow.yml"
            revision = "a"*40
            line = "    uses: myrqyry/obs-plugin-ci/.github/workflows/build.yml@" + revision
            def run(content):
                workflow.write_text(content)
                return subprocess.run([sys.executable, "-O", str(PIN), str(workflow)],
                                      capture_output=True, text=True)
            result = run(line+"\n")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), revision)
            for bad in (line+"\n"+line+"\n", line.replace(revision,"main"), "# "+line, line+"suffix"):
                self.assertNotEqual(run(bad).returncode, 0)
if __name__ == "__main__":
    unittest.main()
