#!/usr/bin/env python3
"""Opt-in OBS/image/video -> local ComfyUI img2img -> atomic PNG bridge.

Nothing captures OBS unless --obs-source is explicitly supplied. This prototype
runs one real GPU job at a time; it never queues stale intermediate video frames.
"""
from __future__ import annotations
import argparse
import base64
import binascii
import io
import json
import os
from pathlib import Path
import tempfile
import time
import uuid


COMFY_ROOT = "http://127.0.0.1:8188"
OBS_URL = "ws://127.0.0.1:4455/"

# Deliberate high-resolution requests are permitted up to this *output*
# budget. Native diffusion computation can still exceed GPU memory.
MAX_OUTPUT_EDGE = 2048
MAX_OUTPUT_PIXELS = 2048 * 1152


def build_img2img_graph(image_name: str, model: str, prompt: str, seed: int,
                        steps: int, denoise: float, cfg: float, sampler: str) -> dict:
    """Always VAE-encode the uploaded IMAGE; no txt2img substitution."""
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": model}},
        "2": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "3": {"class_type": "VAEEncode", "inputs": {"pixels": ["2", 0], "vae": ["1", 2]}},
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["1", 1]}},
        "5": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "", "clip": ["1", 1]}},
        "6": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["4", 0],
                         "negative": ["5", 0], "latent_image": ["3", 0],
                         "seed": seed, "steps": steps, "cfg": cfg,
                         "sampler_name": sampler, "scheduler": "normal",
                         "denoise": denoise}},
        "7": {"class_type": "VAEDecode",
              "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "PreviewImage", "inputs": {"images": ["7", 0]}},
    }


def decode_obs_screenshot(value: str) -> bytes:
    if not value.startswith("data:image/png;base64,"):
        raise ValueError("OBS screenshot was not a PNG data URI")
    try:
        blob = base64.b64decode(value.split(",", 1)[1], validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("Invalid OBS screenshot base64") from error
    if not blob.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Invalid OBS screenshot PNG")
    return blob


def target_dimensions(source: tuple[int, int], width: int | None,
                      height: int) -> tuple[int, int]:
    """Keep source ratio by default, constrained to 8-pixel latent units.

    Explicit --width selects a deliberate crop to those dimensions.
    """
    src_width, src_height = source
    if src_width <= 0 or src_height <= 0:
        raise ValueError("Invalid source image geometry")
    if width is None:
        width = max(64, 8 * round(src_width * height / src_height / 8))
    validate_output_dimensions(width, height)
    return width, height


def validate_output_dimensions(width: int, height: int) -> None:
    if (width < 64 or height < 64 or width > MAX_OUTPUT_EDGE or
            height > MAX_OUTPUT_EDGE or width % 8 or height % 8 or
            width * height > MAX_OUTPUT_PIXELS):
        raise ValueError(
            f"Output dimensions must be multiples of 8, each 64 to "
            f"{MAX_OUTPUT_EDGE}, within {MAX_OUTPUT_PIXELS:,} pixels; "
            "large models may still run out of VRAM"
        )


def normalized_png(data: bytes, width: int, height: int) -> bytes:
    from PIL import Image, ImageOps
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as error:
        raise ValueError("Input image cannot be decoded") from error
    fitted = ImageOps.fit(image.convert("RGB"), (width, height))
    buffer = io.BytesIO()
    fitted.save(buffer, format="PNG")
    return buffer.getvalue()


def store_png(blob: bytes, target: Path) -> None:
    from PIL import Image
    if not blob.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Refusing non-PNG output; previous frame preserved")
    try:
        with Image.open(io.BytesIO(blob)) as image:
            image.verify()
    except Exception as error:
        raise ValueError("Refusing corrupt PNG output; previous frame preserved") from error
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix="sight-diffusion-", suffix=".tmp",
                                     dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(blob)
        os.replace(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


class ComfyClient:
    def __init__(self, root: str = COMFY_ROOT):
        import requests
        if root != COMFY_ROOT:
            raise ValueError("Only local ComfyUI at 127.0.0.1:8188 is supported")
        self.root = root
        self.session = requests.Session()
        self.session.trust_env = False

    def upload(self, blob: bytes, filename: str) -> str:
        result = self.session.post(
            self.root + "/upload/image",
            files={"image": (filename, blob, "image/png")},
            data={"type": "input", "overwrite": "true"}, timeout=30)
        result.raise_for_status()
        name = result.json().get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("ComfyUI image upload did not return a filename")
        return name

    def run_graph(self, graph: dict, timeout: float = 240) -> bytes:
        response = self.session.post(
            self.root + "/prompt",
            json={"prompt": graph, "client_id": "sight-streamdiffusion-bridge"},
            timeout=20)
        response.raise_for_status()
        data = response.json()
        if "error" in data or data.get("node_errors"):
            raise RuntimeError("ComfyUI rejected diffusion graph: " + str(data)[:500])
        job_id = data.get("prompt_id")
        if not job_id:
            raise RuntimeError("No ComfyUI prompt ID returned")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            history = self.session.get(
                self.root + "/history/" + str(job_id), timeout=15)
            history.raise_for_status()
            job = history.json().get(str(job_id))
            if job is not None:
                status = job.get("status", {})
                if not status.get("completed") or status.get("status_str") != "success":
                    raise RuntimeError("Diffusion job failed: " + str(status)[:350])
                images = job.get("outputs", {}).get("8", {}).get("images", [])
                if not images:
                    raise RuntimeError("Completed diffusion job produced no images")
                output = images[0]
                result = self.session.get(
                    self.root + "/view",
                    params={"filename": output["filename"],
                            "subfolder": output.get("subfolder", ""),
                            "type": output.get("type", "temp")},
                    timeout=40)
                result.raise_for_status()
                return result.content
            time.sleep(0.8)
        raise TimeoutError("ComfyUI did not finish within the configured timeout")

    def generate(self, input_png: bytes, *, model: str, prompt: str,
                 seed: int, steps: int, denoise: float, cfg: float, sampler: str,
                 count: int, timeout: float) -> bytes:
        filename = f"sight_bridge_{os.getpid()}.png"
        name = self.upload(input_png, filename)
        return self.run_graph(build_img2img_graph(name, model, prompt, seed,
                                                  steps, denoise, cfg, sampler),
                              timeout=timeout)


class ObsScreenshots:
    def __init__(self, source: str):
        import websocket
        self.source = source
        try:
            self.ws = websocket.create_connection(OBS_URL, timeout=4)
        except (OSError, websocket.WebSocketException) as error:
            raise RuntimeError(
                "Cannot connect to OBS. Open OBS Studio, enable its local "
                "WebSocket server on port 4455, and select 'Sight Diffusion "
                "Preview' before starting the bridge.") from error
        welcome = json.loads(self.ws.recv())
        if welcome.get("op") != 0:
            raise RuntimeError("OBS WebSocket did not send Hello")
        if "authentication" in welcome.get("d", {}):
            raise RuntimeError("OBS WebSocket requires authentication; no credentials were read")
        self.ws.send(json.dumps(
            {"op": 1, "d": {"rpcVersion": 1, "eventSubscriptions": 0}}))
        identified = json.loads(self.ws.recv())
        if identified.get("op") != 2:
            raise RuntimeError("OBS WebSocket identification failed")

    def request(self, request_type: str, request_data: dict) -> dict:
        ident = uuid.uuid4().hex
        self.ws.send(json.dumps({"op": 6, "d": {
            "requestType": request_type, "requestId": ident,
            "requestData": request_data}}))
        answer = json.loads(self.ws.recv())
        if answer.get("op") != 7 or answer.get("d", {}).get("requestId") != ident:
            raise RuntimeError("Unexpected OBS WebSocket response")
        response = answer["d"]
        if not response.get("requestStatus", {}).get("result"):
            raise RuntimeError("OBS " + request_type + " failed: " +
                               str(response.get("requestStatus", {}))[:240])
        return response.get("responseData", {})

    def ensure_live_source(self) -> None:
        active = self.request("GetSourceActive", {"sourceName": self.source})
        if not active.get("videoActive") or not active.get("videoShowing"):
            raise RuntimeError(
                "OBS live source inactive: make its scene active before "
                "starting diffusion; refusing to process a frozen screenshot")

    def prepare_output(self, source: str, path: Path) -> None:
        """Only touch an explicit Image Source already pointing to this path."""
        settings = self.request("GetInputSettings", {"inputName": source})
        if settings.get("inputKind") != "image_source":
            raise ValueError("OBS output must be an existing Image Source")
        actual = settings.get("inputSettings", {}).get("file")
        if actual != str(path):
            raise ValueError("OBS Image Source is configured for a different file")

    def size_preview(self, scene: str, output_source: str,
                     size: tuple[int, int]) -> None:
        """Set only the explicitly selected preview item, using uniform scale.

        Scale by output height: widescreen matches the canvas while square
        outputs remain centered, with the camera visible to either side.
        """
        if getattr(self, "_preview_geometry", None) == (scene, output_source, size):
            return
        canvas = self.request("GetVideoSettings", {})
        canvas_w = canvas["baseWidth"]
        canvas_h = canvas["baseHeight"]
        items = self.request("GetSceneItemList", {"sceneName": scene})["sceneItems"]
        matches = [item for item in items if item["sourceName"] == output_source]
        if len(matches) != 1:
            raise ValueError("Preview scene must contain exactly one matching Image Source")
        width, height = size
        scale = canvas_h / height
        self.request("SetSceneItemTransform", {
            "sceneName": scene,
            "sceneItemId": matches[0]["sceneItemId"],
            "sceneItemTransform": {
                "positionX": (canvas_w - width * scale) / 2,
                "positionY": 0,
                "scaleX": scale,
                "scaleY": scale,
            },
        })
        self._preview_geometry = (scene, output_source, size)

    def refresh_output(self, source: str, path: Path) -> None:
        # OBS's Image Source may not notice an atomic file replacement;
        # explicitly re-applying the same path forces texture reloading.
        self.request("SetInputSettings", {
            "inputName": source, "inputSettings": {"file": str(path)},
            "overlay": True
        })

    def frame(self, width: int, height: int) -> bytes:
        self.ensure_live_source()
        request_data = {"sourceName": self.source, "imageFormat": "png",
                        "imageHeight": height}
        if width is not None:
            request_data["imageWidth"] = width
        data = self.request("GetSourceScreenshot", request_data)
        return decode_obs_screenshot(data["imageData"])

    def close(self):
        self.ws.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Repeated local image input")
    source.add_argument("--video", type=Path, help="Frames of a local video file")
    source.add_argument("--obs-source", help="Explicit OBS source name to capture")
    parser.add_argument("--obs-output-source", help="Existing OBS Image Source with matching --output file; refreshes it after each frame")
    parser.add_argument("--obs-preview-scene", help="Explicit OBS scene in which to size the output Image Source by height")
    parser.add_argument("--output", type=Path,
                        default=Path("/tmp/sight-streamdiffusion-live.png"))
    parser.add_argument("--model", default="3DV6TurboLCM_v6.safetensors")
    parser.add_argument("--prompt", default="soft watercolor illustration, detailed brush strokes")
    parser.add_argument("--width", type=int, default=None,
                        help="Output width (multiple of 8); default matches the source aspect ratio")
    parser.add_argument("--height", type=int, default=288,
                        help="Output height (default 288; explicit high-res up to 2048 with memory guard)")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--strength", type=float, default=0.7)
    parser.add_argument("--cfg", type=float, default=1.2)
    parser.add_argument("--sampler", default="lcm")
    parser.add_argument("--max-frames", type=int, default=1,
                        help="0 for continuous; defaults to single safe proof frame")
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()
    if not 0 < args.strength <= 1 or args.steps not in range(1, 33) or not 0 < args.cfg <= 20:
        parser.error("Invalid diffusion sampler values")
    try:
        if args.width is None:
            validate_output_dimensions(64, args.height)
        else:
            validate_output_dimensions(args.width, args.height)
    except ValueError as error:
        parser.error(str(error))
    if args.obs_preview_scene and not args.obs_output_source:
        parser.error("--obs-preview-scene requires --obs-output-source")
    if args.max_frames < 0:
        parser.error("--max-frames must be nonnegative")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    client = ComfyClient()
    obs = ObsScreenshots(args.obs_source or "") if (
        args.obs_source or args.obs_output_source) else None
    if args.obs_output_source:
        obs.prepare_output(args.obs_output_source, args.output)
    video = None
    if args.video:
        import cv2
        video = cv2.VideoCapture(str(args.video))
        if not video.isOpened():
            raise RuntimeError("Cannot open local video file")
    print(f"LOCAL_DIFFUSION_SOURCE image={args.image} video={args.video} obs={args.obs_source}", flush=True)
    print(f"MODEL {args.model} sampler={args.sampler} steps={args.steps} "
          f"denoise={args.strength} cfg={args.cfg} "
          f"output_width={'auto' if args.width is None else args.width} "
          f"output_height={args.height}",
          flush=True)
    count = 0
    try:
        while not args.max_frames or count < args.max_frames:
            if args.obs_source:
                frame = obs.frame(None, args.height)
            elif video:
                import cv2
                success, bgr = video.read()
                if not success:
                    print("VIDEO_EOF", flush=True)
                    break
                success, encoded = cv2.imencode(".png", bgr)
                if not success:
                    raise RuntimeError("Could not encode a video frame")
                frame = encoded.tobytes()
            else:
                frame = args.image.read_bytes()
            from PIL import Image
            with Image.open(io.BytesIO(frame)) as captured:
                size = target_dimensions(captured.size, args.width, args.height)
            png = normalized_png(frame, *size)
            started = time.monotonic()
            generated = client.generate(
                png, model=args.model, prompt=args.prompt, seed=20300828,
                steps=args.steps, denoise=args.strength, cfg=args.cfg,
                sampler=args.sampler, count=count, timeout=args.timeout)
            store_png(generated, args.output)
            if args.obs_output_source:
                obs.refresh_output(args.obs_output_source, args.output)
                if args.obs_preview_scene:
                    obs.size_preview(args.obs_preview_scene,
                                     args.obs_output_source, size)
            count += 1
            print(f"REAL_DIFFUSION_FRAME {count}: {args.output} "
                  f"{size[0]}x{size[1]} ({time.monotonic()-started:.2f}s)",
                  flush=True)
    except KeyboardInterrupt:
        print("Stopped at user request", flush=True)
    finally:
        if obs:
            obs.close()
        if video:
            video.release()


if __name__ == "__main__":
    main()
