import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"tools/acquire-native-ort.py"
class Test(unittest.TestCase):
    def test_exact_pinned_members_and_no_overwrite(self):
        # Fixture exercises the same extraction function with an independent
        # synthetic receipt. It is not a native library or inference test.
        import importlib.util
        spec=importlib.util.spec_from_file_location("acquire",SCRIPT)
        mod=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive=root/"fixture.zip";dest=root/"sdk"
            with zipfile.ZipFile(archive,"w") as z:
                z.writestr("vendor/lib",b"binary")
                z.writestr("vendor/LICENSE",b"notice")
                z.writestr("unselected/weights.onnx",b"weights")
            receipt={"archive_sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),
                     "targets":{"windows-x64":{"lib/onnxruntime.dll":{"member":"vendor/lib","sha256":hashlib.sha256(b"binary").hexdigest()},
                     "LICENSE":{"member":"vendor/LICENSE","sha256":hashlib.sha256(b"notice").hexdigest()}}}}
            mod.extract(archive,dest,"windows-x64",receipt)
            self.assertEqual((dest/"lib/onnxruntime.dll").read_bytes(),b"binary")
            self.assertEqual({p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()},{"lib/onnxruntime.dll","LICENSE"})
            with self.assertRaises(ValueError):mod.extract(archive,dest,"windows-x64",receipt)
            receipt["targets"]["windows-x64"]["lib/onnxruntime.dll"]["sha256"]="0"*64
            with self.assertRaises(ValueError):mod.extract(archive,root/"bad","windows-x64",receipt)
            self.assertFalse((root/"bad").exists())
            receipt["archive_sha256"]="0"*64
            with self.assertRaises(ValueError):mod.extract(archive,root/"bad-archive","windows-x64",receipt)
            self.assertFalse((root/"bad-archive").exists())
if __name__=="__main__":unittest.main()
