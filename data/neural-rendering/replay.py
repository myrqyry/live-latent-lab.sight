"""Fixed-shape CUDA graph runner, single owning stream, frozen inference."""
import torch


class FixedReplay:
    """Captures a tensor-only function, returns independent output tensors.

    Use on one CUDA stream from one worker. Shape/device/dtype are fixed.
    Model weights and options must remain immutable after capture.
    """
    @torch.inference_mode()
    def __init__(self, fn, *inputs):
        if not inputs or any(not x.is_cuda or x.device != inputs[0].device for x in inputs):
            raise ValueError("all inputs must share one CUDA device")
        self.device=inputs[0].device
        self.owner=torch.cuda.current_stream(self.device)
        self.inputs=tuple(x.clone().contiguous() for x in inputs)
        warm=torch.cuda.Stream(device=self.device)
        warm.wait_stream(self.owner)
        with torch.cuda.stream(warm):
            for _ in range(3):
                fn(*self.inputs)
        self.owner.wait_stream(warm)
        self.owner.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.device(self.device), torch.cuda.graph(self.graph):
            self.output=fn(*self.inputs)
        if not isinstance(self.output,torch.Tensor):
            raise TypeError("expected one tensor output")
        self.owner.wait_stream(torch.cuda.current_stream(self.device))

    @torch.inference_mode()
    def __call__(self,*inputs):
        if torch.cuda.current_stream(self.device)!=self.owner:
            raise RuntimeError("replay must use its owning stream")
        if len(inputs)!=len(self.inputs):
            raise ValueError("input count changed")
        for dst,src in zip(self.inputs,inputs):
            if (src.shape,src.dtype,src.device)!=(dst.shape,dst.dtype,dst.device):
                raise ValueError("fixed graph input shape, dtype or device changed")
        for dst,src in zip(self.inputs,inputs):
            dst.copy_(src)
        self.graph.replay()
        return self.output.clone()
