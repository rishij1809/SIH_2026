#!/usr/bin/env python3
"""
Project Vigilance - Stage 2 Dual-Model Training & Edge Optimization Demonstrator
Executes:
1. Tier 2 Spatio-Temporal Distress Model Training (AMP + Focal Loss + Cosine Annealing)
2. Model Export to TorchScript (.pt) and ONNX (.onnx)
3. Operational Profiling & Latency Benchmark
"""

from pathlib import Path
import torch

from src.annotation.sequence_builder import SequenceDatasetBuilder
from src.export.exporter import ModelExporter
from src.models.distress_net import DistressSpatioTemporalNet
from src.training.distress_trainer import DistressModelTrainer, KeypointSequenceDataset
from src.utils.logger import setup_logger
from src.validation.benchmark import OperationalBenchmark

logger = setup_logger("Stage2DemoPipeline")


def main():
    base_dir = Path(__file__).parent.resolve()
    logger.info("Initializing Stage 2 Training & Edge Optimization Pipeline...")

    # Step 1: Prepare synthetic sequence data for fast demonstration if needed
    seq_dir = base_dir / "data" / "sequence_dataset"
    seq_dir.mkdir(parents=True, exist_ok=True)
    builder = SequenceDatasetBuilder(window_size=30, step_size=5, num_keypoints=33, channels=3)
    X, y = builder.generate_synthetic_demo_sequences(num_samples_per_class=20)
    builder.save_dataset(X, y, output_dir=seq_dir)

    # Step 2: Instantiate Tier 2 PyTorch Model
    logger.info("\n--- STEP 1: INSTANTIATE TIER-2 SPATIO-TEMPORAL MODEL ---")
    model = DistressSpatioTemporalNet(num_keypoints=33, channels=3, num_classes=4)
    logger.info(f"Model Total Trainable Parameters: {model.count_parameters():,}")

    # Step 3: Run Tier 2 PyTorch Training Engine
    logger.info("\n--- STEP 2: RUN TIER-2 AMP TRAINING LOOP (Focal Loss + Cosine Annealing) ---")
    dataset = KeypointSequenceDataset(X, y)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = DistressModelTrainer(model=model, device=device, lr=1e-3, use_focal_loss=True)
    
    ckpt_path = base_dir / "models" / "distress_model.pt"
    trainer.train(train_loader, val_loader, epochs=5, checkpoint_path=str(ckpt_path))

    # Step 4: Edge Optimization & Model Export Engine
    logger.info("\n--- STEP 3: EXPORT MODELS TO ONNX AND TORCHSCRIPT ---")
    exporter = ModelExporter(models_dir=base_dir / "models")
    dummy_seq = torch.randn(1, 30, 33, 3)
    
    traced_path = exporter.export_distress_to_torchscript(model, dummy_seq, out_filename="distress_traced.pt")
    onnx_path = exporter.export_distress_to_onnx(model, dummy_seq, out_filename="distress_model.onnx", opset=13)

    # Step 5: Operational Benchmark & Profiling
    logger.info("\n--- STEP 4: OPERATIONAL LATENCY & THROUGHOUT BENCHMARKING ---")
    bench = OperationalBenchmark()
    res = bench.benchmark_model_latency(model, dummy_seq, device_str=device, num_iters=50)
    res["model_name"] = "Tier-2 Spatio-Temporal Model"
    
    bench.print_operational_profile([res])

    logger.info("Stage 2 Pipeline Execution Finished Successfully!")


if __name__ == "__main__":
    main()
