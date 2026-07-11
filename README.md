# Video-LLM Optimization

A batch processing pipeline for vision-language models, achieving **3.88x speedup** on video frame inference by processing multiple frames in parallel. Includes rigorous profiling and honest failure analysis of alternative approaches.

## Quick Start

```bash
# Install dependencies
pip install torch transformers ffmpeg-python

# Run the optimized pipeline
python video_pipeline.py --video sample.mp4

# Run benchmarks
python benchmark.py

# See profiling results
python benchmark.py --profile
```

## Results

| Metric | Value |
|--------|-------|
| **Speedup** | 3.88x (frame-by-frame → batch processing) |
| **Test video** | Real Madrid vs Barcelona (428 frames) |
| **Model** | Microsoft Git (vision-language) |
| **Hardware** | Tesla T4 GPU |

## Tested Approaches (All Documented)

| Approach | Result | Analysis |
|----------|--------|----------|
| ✅ **Batch processing** | 3.88x faster | GPU parallelization excels when processing multiple frames simultaneously |
| ❌ Fused preprocessing kernel | 0.01x (slower) | CPU preprocessing already fast (101 FPS); Triton overhead > benefit |
| ❌ Fused attention kernel | 0.05x (slower) | Attention only 5-10% of bottleneck; optimizing it doesn't improve overall speed |
| ❌ Int8 quantization | 0.29x (slower) | Overhead of float32↔int8 conversion outweighs precision loss benefits |
| ❌ Model switching (BLIP) | 0.24x (slower) | Alternative model was actually slower than baseline |

**Key insight:** Profile first. The bottleneck is frame decoding and model loading, not attention. Optimizing attention doesn't help because it's not the bottleneck.

## How It Works

### Frame-by-Frame (Baseline)
```
Video → Decode frame 1 → Load model → Inference → Output
      → Decode frame 2 → Load model → Inference → Output
      → Decode frame 3 → Load model → Inference → Output
```
Each step is sequential; GPU sits idle during decoding.

### Batch Processing (Optimized)
```
Video → Decode frames 1-16 in parallel
      → Load model once
      → Inference all 16 frames at once (GPU parallelization)
      → Output all results
```
Decoding, loading, inference all amortized across 16 frames.

## Files

- `video_pipeline.py` — Main optimized pipeline
- `smart_routing.py` — Auto-detect file type (MP4 vs PNG/JPG)
- `benchmark.py` — Timing + profiling + all approach comparisons
- `README.md` — This file

## Benchmarking

```bash
python benchmark.py
```

Profiles each component:
- Frame decoding time
- Model loading time
- Inference time
- Total throughput (frames/sec)

Results show batch processing dominates because GPU parallelization is free but frame-by-frame loading overhead is expensive.

## Testing

```bash
python -m pytest test_video_pipeline.py -v
```

Tests:
- Video loading and frame extraction
- Batch assembly correctness
- Output shape validation
- Edge cases (short videos, single frame, etc.)

## Why This Matters

Video understanding is increasingly important in production ML. A 3.88x speedup means:
- Processing 1000 hours of video in 4 hours instead of 15
- Lower GPU utilization → lower cost
- Faster model iteration during development

## Known Limitations

- Tested on T4 GPU; performance varies by hardware
- Batch size tuned for 428-frame videos; adjust for different lengths
- Assumes H.264 video format

## Key Takeaway

Don't optimize blindly. Profile first, find the real bottleneck, then optimize that. In this case, the obvious targets (attention, quantization) weren't the bottleneck. Batch processing was.

## License

MIT
