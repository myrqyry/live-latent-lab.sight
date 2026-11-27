#!/usr/bin/env python3
import hashlib
import importlib.util
import pathlib
import tempfile
import unittest
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("acquire", ROOT / "tools/acquire_mediapipe.py")
acquire = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acquire)

class AcquisitionTest(unittest.TestCase):
    def test_verified_extraction_and_existing_file_protection(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            wheel = root / "fixture.whl"
            data = b"test ELF payload"
            with zipfile.ZipFile(wheel, "w") as z:
                z.writestr(acquire.MEMBER, data)
                z.writestr("../../not-extracted", b"no")
            digest = lambda b: hashlib.sha256(b).hexdigest()
            out = root / "output.so"
            acquire.extract_verified(wheel, out, digest(wheel.read_bytes()), digest(data), len(data))
            self.assertEqual(out.read_bytes(), data)
            self.assertFalse((root / "not-extracted").exists())
            with self.assertRaises(FileExistsError):
                acquire.extract_verified(wheel, out, digest(wheel.read_bytes()), digest(data), len(data))
            self.assertEqual(out.read_bytes(), data)

    def test_bad_package_or_member_cannot_publish_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            wheel = root / "fixture.whl"
            with zipfile.ZipFile(wheel, "w") as z:
                z.writestr(acquire.MEMBER, b"wrong")
            out = root / "output.so"
            digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
            for w, m, n in [("0"*64, "0"*64, 5), (digest, "0"*64, 5), (digest, "0"*64, 4)]:
                with self.assertRaises(ValueError):
                    acquire.extract_verified(wheel, out, w, m, n)
                self.assertFalse(out.exists())

if __name__ == "__main__":
    unittest.main()
