"""
Operational Profiling and Latency Benchmark Engine for Project Vigilance.
Measures parameter count, estimated FLOPs, CPU/GPU inference latency (ms), throughput (FPS), and memory footprint.
"""

import time
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from tabulate import tabulate

from src.models.distress_net import DistressSpatioTemporalNet
from src.utils.logger import setup_logger

logger = setup_logger("BenchmarkEngine")


class OperationalBenchmark:
    """Measures operational latency, FLOPs, parameters, and throughput for edge models."""

    @staticmethod
    def count_parameters(model: torch.nn.Module) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def estimate_flops(model: torch.nn.Module, dummy_input: torch.Tensor) -> float:
        """
        Estimates total Floating Point Operations (FLOPs) for forward pass.
        Returns value in MegaFLOPs (MFLOPs).
        """
        # Simple FLOP estimation based on model parameter interactions
        params = sum(p.numel() for p in model.parameters())
        # Forward pass multiplies params by approx 2 FLOPs per parameter per token/frame
        flops = params * 2.0 * dummy_input.shape[1] if dummy_input.dim() >= 2 else params * 2.0
        return float(flops / 1e6)

    def benchmark_model_latency(
        self,
        model: torch.nn.Module,
        dummy_input: torch.Tensor,
        device_str: str = "cpu",
        num_warmup: int = 10,
        num_iters: int = 100
    ) -> Dict[str, Any]:
        """
        Executes benchmark loop on CPU or CUDA GPU measuring latency and throughput.
        """
        device = torch.device(device_str)
        model = model.to(device)
        model.eval()

        inp = dummy_input.to(device)

        # Warmup iterations
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(inp)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark loop
        latencies_ms = []
        with torch.no_grad():
            for _ in range(num_iters):
                start_t = time.perf_counter()
                _ = model(inp)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end_t = time.perf_counter()
                latencies_ms.append((end_t - start_t) * 1000.0)

        mean_latency = float(np.mean(latencies_ms))
        std_latency = float(np.std(latencies_ms))
        p95_latency = float(np.percentile(latencies_ms, 95))
        fps = float(1000.0 / max(1e-5, mean_latency))

        params = self.count_parameters(model)
        mflops = self.estimate_flops(model, dummy_input)

        return {
            "device": device_str.upper(),
            "params_millions": float(params / 1e6),
            "mflops": mflops,
            "mean_latency_ms": mean_latency,
            "p95_latency_ms": p95_latency,
            "fps": fps
        }

    def print_operational_profile(self, results_list: List[Dict[str, Any]]) -> None:
        """Prints clean formatted operational profile table."""
        print("\n" + "=" * 85)
        print("         PROJECT VIGILANCE - STAGE 2 EDGE OPERATIONAL PROFILE          ")
        print("=" * 85)

        table_data = []
        for res in results_list:
            table_data.append([
                res["model_name"],
                res["device"],
                f"{res['params_millions']:.3f} M",
                f"{res['mflops']:.2f} MFLOPs",
                f"{res['mean_latency_ms']:.2f} ms",
                f"{res['p95_latency_ms']:.2f} ms",
                f"{res['fps']:.1f} FPS"
            ])

        headers = ["Model", "Device", "Params", "FLOPs", "Mean Latency", "P95 Latency", "Throughput"]
        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
        print("=" * 85 + "\n")
