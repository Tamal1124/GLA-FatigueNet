"""Model benchmark: parameter count, FLOPs, inference speed."""
import os, sys, time, torch, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import load_config, count_parameters, format_params
from models.gla_fatiguenet import GLAFatigueNet


def benchmark(config_path="config/config.yaml"):
    config = load_config(config_path)
    model = GLAFatigueNet(config)
    model.eval()
    total, trainable = count_parameters(model)
    # Per-stream params
    ghost_p = sum(p.numel() for p in model.ghost_cnn.parameters())
    cat_p = sum(p.numel() for p in model.cat_transformer.parameters())
    gla_p = sum(p.numel() for p in model.geometric_stream.parameters())
    fusion_p = sum(p.numel() for p in model.fusion.parameters())
    heads_p = sum(p.numel() for p in model.fatigue_head.parameters()) + sum(p.numel() for p in model.emotion_head.parameters())

    print("="*50)
    print("GLA-FatigueNet Model Benchmark")
    print("="*50)
    print(f"Total Parameters:     {format_params(total)} ({total:,})")
    print(f"Trainable Parameters: {format_params(trainable)} ({trainable:,})")
    print(f"\nPer-Component Breakdown:")
    print(f"  GhostCNN:           {format_params(ghost_p)}")
    print(f"  CAT Transformer:    {format_params(cat_p)}")
    print(f"  GLA Stream:         {format_params(gla_p)}")
    print(f"  Fusion Module:      {format_params(fusion_p)}")
    print(f"  Task Heads:         {format_params(heads_p)}")

    # Inference speed
    dummy_img = torch.randn(1, 3, 224, 224)
    dummy_geo = torch.randn(1, config['model']['gla']['geometric_features'])
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            model(dummy_img, dummy_geo)
    # Measure
    times = []
    with torch.no_grad():
        for _ in range(50):
            start = time.time()
            model(dummy_img, dummy_geo)
            times.append(time.time() - start)
    avg_ms = sum(times) / len(times) * 1000
    fps = 1000 / avg_ms
    print(f"\nInference Speed (CPU, batch=1):")
    print(f"  Average: {avg_ms:.1f} ms/image")
    print(f"  FPS:     {fps:.1f}")

    # Batch inference
    batch_img = torch.randn(32, 3, 224, 224)
    batch_geo = torch.randn(32, config['model']['gla']['geometric_features'])
    with torch.no_grad():
        start = time.time()
        model(batch_img, batch_geo)
        batch_time = (time.time() - start) * 1000
    print(f"  Batch=32: {batch_time:.1f} ms ({batch_time/32:.1f} ms/image)")

    results = {'total_params': total, 'trainable_params': trainable, 'inference_ms': avg_ms,
               'fps': fps, 'ghost_params': ghost_p, 'cat_params': cat_p, 'gla_params': gla_p,
               'fusion_params': fusion_p, 'heads_params': heads_p}
    os.makedirs('./results/plots', exist_ok=True)
    with open('./results/plots/benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    return results

if __name__ == '__main__':
    benchmark()
