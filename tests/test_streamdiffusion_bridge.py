import base64
import importlib.util
import io
import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from PIL import Image

MODULE = Path(__file__).resolve().parents[1] / "tools" / "sight_streamdiffusion_bridge.py"
SPEC = importlib.util.spec_from_file_location("sight_streamdiffusion_bridge", MODULE)
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


class BridgeTests(unittest.TestCase):
    def test_graph_is_actual_img2img_not_txt2img(self):
        nodes = bridge.build_img2img_graph(
            "image.png", "3DV6TurboLCM_v6.safetensors",
            "watercolor", 13, 4, 0.65, 1.2, "lcm"
        )
        self.assertEqual(nodes["3"]["class_type"], "VAEEncode")
        self.assertEqual(nodes["6"]["inputs"]["latent_image"], ["3", 0])
        self.assertEqual(nodes["6"]["inputs"]["denoise"], 0.65)
        self.assertEqual(nodes["6"]["inputs"]["seed"], 13)
        self.assertEqual(nodes["8"]["class_type"], "PreviewImage")

    def test_default_size_follows_input_aspect_ratio(self):
        self.assertEqual(bridge.target_dimensions((1920, 1080), None, 288), (512, 288))
        self.assertEqual(bridge.target_dimensions((640, 480), None, 288), (384, 288))
        self.assertEqual(bridge.target_dimensions((1280, 720), 256, 256), (256, 256))

    def test_opt_in_1080p_preserves_full_16_9_width(self):
        self.assertEqual(bridge.target_dimensions((1920,1080),None,1080),
                         (1920,1080))
        self.assertEqual(bridge.target_dimensions((3840,2160),None,1152),
                         (2048,1152))

    def test_explicit_high_resolution_rejected_beyond_budget(self):
        with self.assertRaisesRegex(ValueError,"resolution|budget|dimensions"):
            bridge.target_dimensions((1920,1080),None,1280)
        with self.assertRaisesRegex(ValueError,"resolution|budget|dimensions"):
            bridge.target_dimensions((1920,1080),2048,1536)

    def test_center_crop_keeps_original_proportions_in_square_mode(self):
        from PIL import ImageDraw
        frame = Image.new("RGB", (600, 300), "red")
        ImageDraw.Draw(frame).rectangle((150, 0, 449, 299), fill="blue")
        buf = io.BytesIO()
        frame.save(buf, "PNG")
        cropped = Image.open(io.BytesIO(bridge.normalized_png(buf.getvalue(), 256, 256)))
        self.assertEqual(cropped.size, (256, 256))
        self.assertEqual(cropped.getpixel((128, 128)), (0, 0, 255))

    def test_height_only_obs_screenshot_preserves_source_aspect(self):
        obs = bridge.ObsScreenshots.__new__(bridge.ObsScreenshots)
        obs.source = "Camera"
        image = Image.new("RGB", (512, 288))
        out = io.BytesIO()
        image.save(out, format="PNG")
        obs.ensure_live_source = MagicMock()
        obs.request = MagicMock(return_value={
            "imageData": "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()
        })
        png = obs.frame(None, 288)
        obs.request.assert_called_once_with(
            "GetSourceScreenshot", {
                "sourceName": "Camera", "imageFormat": "png", "imageHeight": 288
            })
        self.assertEqual(Image.open(io.BytesIO(png)).size, (512, 288))

    def test_preview_layout_fills_canvas_without_stretching(self):
        obs = bridge.ObsScreenshots.__new__(bridge.ObsScreenshots)
        obs.request = MagicMock(side_effect=[
            {"baseWidth": 1920, "baseHeight": 1080},
            {"sceneItems": [
                {"sourceName": "Sight Diffusion (Live Test)", "sceneItemId": 21}
            ]},
            {},
        ])
        obs.size_preview("Sight Diffusion Preview", "Sight Diffusion (Live Test)", (512, 288))
        self.assertEqual(obs.request.call_args.args[0], "SetSceneItemTransform")
        params = obs.request.call_args.args[1]
        values = params["sceneItemTransform"]
        self.assertEqual(values["positionX"], 0)
        self.assertEqual(values["positionY"], 0)
        self.assertAlmostEqual(values["scaleX"], 3.75)
        self.assertAlmostEqual(values["scaleY"], 3.75)

    def test_preview_layout_centers_square_crop_by_height(self):
        obs = bridge.ObsScreenshots.__new__(bridge.ObsScreenshots)
        obs.request = MagicMock(side_effect=[
            {"baseWidth": 1920, "baseHeight": 1080},
            {"sceneItems": [{"sourceName": "diffusion", "sceneItemId": 4}]},
            {},
        ])
        obs.size_preview("Sight Diffusion Preview", "diffusion", (256, 256))
        values = obs.request.call_args.args[1]["sceneItemTransform"]
        self.assertEqual(values["positionX"], 420)
        self.assertEqual(values["positionY"], 0)
        self.assertEqual(values["scaleX"], values["scaleY"])
        self.assertAlmostEqual(values["scaleY"], 1080 / 256)

    def test_decode_obs_screenshot_requires_real_png(self):
        image = Image.new("RGB", (128, 128), (12, 34, 56))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        self.assertEqual(bridge.decode_obs_screenshot(url), buffer.getvalue())
        with self.assertRaises(ValueError):
            bridge.decode_obs_screenshot("data:image/png;base64,broken")

    def test_output_is_atomically_replaced_only_after_valid_png(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as folder:
            dest = Path(folder) / "live.png"
            dest.write_bytes(b"existing")
            with self.assertRaises(ValueError):
                bridge.store_png(b"not png", dest)
            self.assertEqual(dest.read_bytes(), b"existing")
            img = Image.new("RGB", (16, 16), (1, 2, 3))
            buffer = io.BytesIO(); img.save(buffer, format="PNG")
            bridge.store_png(buffer.getvalue(), dest)
            self.assertEqual(Image.open(dest).getpixel((1, 1)), (1, 2, 3))
            self.assertFalse(list(Path(folder).glob("*.tmp")))

    def test_sequential_frames_reuse_one_upload_filename(self):
        client = bridge.ComfyClient.__new__(bridge.ComfyClient)
        client.upload = MagicMock(return_value="input.png")
        client.run_graph = MagicMock(return_value=b"generated")
        kwargs = dict(model="test.safetensors", prompt="test", seed=1,
                      steps=4, denoise=.7, cfg=1.2, sampler="lcm",
                      timeout=20)
        client.generate(b"first", count=0, **kwargs)
        client.generate(b"second", count=1, **kwargs)
        calls = client.upload.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].args[1], calls[1].args[1])
        self.assertNotEqual(calls[0].args[0], calls[1].args[0])

    def test_output_refresh_only_targets_preconfigured_image_source(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as folder:
            output = Path(folder) / "live.png"
            obs = bridge.ObsScreenshots.__new__(bridge.ObsScreenshots)
            obs.request = MagicMock(return_value={
                "inputKind": "image_source",
                "inputSettings": {"file": str(output)}
            })
            obs.prepare_output("Sight Diffusion (Live Test)", output)
            obs.request.reset_mock()
            obs.refresh_output("Sight Diffusion (Live Test)", output)
            obs.request.assert_called_once_with(
                "SetInputSettings", {
                    "inputName": "Sight Diffusion (Live Test)",
                    "inputSettings": {"file": str(output)},
                    "overlay": True
                })

    def test_output_refresh_rejects_wrong_source_or_file(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as folder:
            output = Path(folder) / "live.png"
            obs = bridge.ObsScreenshots.__new__(bridge.ObsScreenshots)
            obs.request = MagicMock(return_value={
                "inputKind": "v4l2_input", "inputSettings": {"file": str(output)}
            })
            with self.assertRaises(ValueError):
                obs.prepare_output("webcam", output)
            obs.request.return_value = {
                "inputKind": "image_source", "inputSettings": {"file": "/tmp/other.png"}
            }
            with self.assertRaises(ValueError):
                obs.prepare_output("other input", output)

    def test_obs_connection_refused_is_actionable(self):
        from unittest.mock import patch
        import websocket
        with patch.object(websocket, "create_connection",
                          side_effect=ConnectionRefusedError("Connection refused")):
            with self.assertRaisesRegex(RuntimeError, "Open OBS"):
                bridge.ObsScreenshots("Camera")

    def test_live_source_must_be_active_before_processing(self):
        obs = bridge.ObsScreenshots.__new__(bridge.ObsScreenshots)
        obs.source = "Camera"
        obs.request = MagicMock(return_value={"videoActive": False, "videoShowing": False})
        with self.assertRaisesRegex(RuntimeError, "inactive"):
            obs.ensure_live_source()
        obs.request.return_value = {"videoActive": True, "videoShowing": True}
        obs.ensure_live_source()
        obs.request.assert_called_with("GetSourceActive", {"sourceName": "Camera"})

    def test_image_input_with_obs_preview_does_not_capture_obs(self):
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        import sys
        with TemporaryDirectory() as folder:
            input_path = Path(folder) / "input.png"
            output_path = Path(folder) / "out.png"
            image = Image.new("RGB", (64, 64), (30, 60, 90))
            image.save(input_path)
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            argv = ["bridge", "--image", str(input_path),
                    "--output", str(output_path),
                    "--obs-output-source", "Sight Diffusion (Live Test)"]
            with patch.object(sys, "argv", argv), \
                    patch.object(bridge, "ComfyClient") as client, \
                    patch.object(bridge, "ObsScreenshots") as obs_type:
                client.return_value.generate.return_value = buf.getvalue()
                bridge.main()
                obs_type.return_value.frame.assert_not_called()
                obs_type.return_value.prepare_output.assert_called_once()
                obs_type.return_value.refresh_output.assert_called_once()
                self.assertTrue(output_path.is_file())

    def test_does_not_queue_extra_jobs_until_previous_finishes(self):
        client = bridge.ComfyClient.__new__(bridge.ComfyClient)
        client.session = MagicMock()
        client.root = "http://127.0.0.1:8188"
        created = MagicMock()
        created.status_code = 200
        created.json.return_value = {"prompt_id": "abc"}
        created.raise_for_status.return_value = None
        history1 = MagicMock()
        history1.json.return_value = {}
        history2 = MagicMock()
        history2.json.return_value = {"abc": {"status": {"completed": True, "status_str": "success"},
                                             "outputs": {"8": {"images": [{"filename":"abc.png","type":"temp","subfolder":""}]}}}}
        fetched = MagicMock()
        fetched.content = b"some png"; fetched.raise_for_status.return_value = None
        client.session.post.return_value = created
        client.session.get.side_effect = [history1, history2, fetched]
        with unittest.mock.patch.object(bridge.time, "sleep", return_value=None):
            result = client.run_graph({"1": {}}, timeout=10)
        self.assertEqual(result, b"some png")
        self.assertEqual(client.session.post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
