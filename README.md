# AVTR-1 Talking Avatar on Kaggle (TensorRT build + custom avatar)

This notebook runs [avaturn-live/avtr-1](https://github.com/avaturn-live/avtr-1) end-to-end on a free Kaggle GPU. It builds the model's TensorRT engines, makes a test video with the bundled `maria` avatar, adds a custom **doctor** avatar from your own image, drives it with **Hindi speech**, and packages the finished engines as a zip you can download.

Pipeline: **audio → HuBERT speech features → speech-to-motion (keypoints) → neural renderer (warp + decode + matting + stitch) → MP4 at 25 fps**

---

## Requirements

| Item | Value used in this run |
|---|---|
| Platform | Kaggle Notebook, **GPU accelerator on**, **Internet on** |
| GPU | NVIDIA Tesla T4 (15 GB VRAM), driver 580, CUDA 13.0 |
| Disk | `/kaggle/working` (20 GB). Peak use is about 16 GB, so you have to clean up (see step 10) |
| Python | 3.12 (managed by Pixi) |
| Env manager | [Pixi](https://pixi.sh), with the repo's `pixi.toml` pinning CUDA 12.8 |
| Accounts | Hugging Face access token (read) |
| Kaggle dataset | An avatar image, e.g. `henil2132/faceimage` |

---

## Steps

| Cell | What it does |
|---|---|
| 0 | Checks the GPU (`nvidia-smi`) and free disk space |
| 1 | Clones `avtr-1` into `/kaggle/working/avtr-1` |
| 2 | Installs Pixi and adds `/root/.pixi/bin` to `PATH` |
| 3 | `pixi install` creates the environment (about 12 GB in `.pixi/`) |
| 4 | Logs in to Hugging Face |
| 5 | `scripts/download_artifacts.py` downloads the ONNX / TorchScript weights (HuBERT-LBS 1.26 GB, avtr1 613 MB, MODNet, etc.) |
| 6 | Installs `onnx` and `onnxscript` into the Pixi env |
| 7 | `scripts/build_avtr1_engines.py` builds the speech-to-motion encode/decode engines and the normalizer stats |
| 8 | Installs `onnx-graphsurgeon` |
| 9 | `scripts/build_renderer_engines.py` builds the 4 renderer engines (batch 5, FP16) |
| 10 | `scripts/build_hubert_engine.py` builds the HuBERT audio-feature engine |
| 11 | Checks that all 7 engines exist |
| 12–13 | Frees about 2.4 GB by deleting `artifacts/main/build_artifacts` (no longer needed once the engines are built) |
| 14–15 | Test render: `example/speaker_1.ogg` + avatar `maria`, then plays it inline |
| 16–17 | Custom avatar: copies your image to `reference_frames/doctor.png` and letterboxes it onto a transparent 1920×1080 canvas |
| 18 | Downloads a Hindi doctor–patient clip (`convo_001.mp3`) |
| 19–20 | Renders the doctor avatar speaking the Hindi audio, then plays it inline |
| 21–23 | Lists the engines, packages them into `avtr1_production_engines.zip`, and shows a download link |

---

## Quick start

1. Create a Kaggle notebook, set **Accelerator → GPU T4**, and turn **Internet** on.
2. Add your avatar image as a Kaggle dataset, then change the `source` path in cell 16.
3. Set your Hugging Face token in cell 4.
4. **Run All.** The engine builds take roughly 15–20 minutes on a T4.

### Generating a video

```bash
cd /kaggle/working/avtr-1
pixi run generate_offline \
    --speech <path/to/audio.(ogg|mp3|wav)> \
    --avatar <avatar_name> \
    --bg plain_white \
    --out /kaggle/working/output.mp4
```

- `--avatar` is the file name, without extension, of an image in `artifacts/main/avatars_artifacts/reference_frames/` (e.g. `maria`, `doctor`).
- `--bg` sets the background preset (`plain_white` in this notebook).

### Adding your own avatar

```python
from PIL import Image
import shutil

src_img = "/kaggle/input/<your-dataset>/<image>.png"
dst = "/kaggle/working/avtr-1/artifacts/main/avatars_artifacts/reference_frames/<name>.png"
shutil.copy2(src_img, dst)

# Fit onto a transparent 1920x1080 canvas, centred, aspect ratio kept
img = Image.open(dst).convert("RGBA")
W, H = 1920, 1080
s = min(W / img.width, H / img.height)
img = img.resize((int(img.width * s), int(img.height * s)), Image.Resampling.LANCZOS)
canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
canvas.alpha_composite(img, ((W - img.width) // 2, (H - img.height) // 2))
canvas.save(dst)
```

Use a clear, front-facing portrait with the head and shoulders visible.

---

## Built engines

All engines are FP16 TensorRT, built for the GPU they were compiled on.

| Stage | Engine | Size | I/O |
|---|---|---|---|
| speech2motion | `hubert_lbs_fp16.engine` | 607 MB | audio → speech features |
| speech2motion | `avtr1_encode_fp16.engine` | 21 MB | features → latent |
| speech2motion | `avtr1_decode_fp16.engine` | 279 MB | latent → motion / keypoints |
| renderer | `warp_network_b5_fp16.engine` | 99 MB | `feature_3d (B,32,16,64,64)`, `kp_source`, `kp_driving (B,21,3)` → `(B,256,64,64)` |
| renderer | `decoder_b5_fp16.engine` | 108 MB | `(B,256,64,64)` → RGB `(B,3,512,512)` |
| renderer | `modnet_b5_fp16.engine` | 16 MB | `(B,3,288,512)` → alpha matte `(B,1,288,512)` |
| renderer | `stitch_network_b5_fp16.engine` | 0.5 MB | `kp_source`, `kp_driving` → stitched keypoints `(B,21,3)` |

The engines are written to `artifacts/main/speech2motion_runtime_artifacts_cc/` and `artifacts/main/renderer_runtime_artifacts_cc/`, along with `avtr1_normalizer.safetensors`.

The exported package (`avtr1_production_engines.zip`, about 1 GB):

```
avtr1_production_engines/
├── speech2motion/
│   ├── avtr1_encode_fp16.engine
│   ├── avtr1_decode_fp16.engine
│   └── hubert_lbs_fp16.engine
└── renderer/
    ├── decoder_b5_fp16.engine
    ├── warp_network_b5_fp16.engine
    ├── modnet_b5_fp16.engine
    └── stitch_network_b5_fp16.engine
```

> ⚠️ **TensorRT engines are not portable.** They only work on the same GPU architecture (T4 / sm_75) with the same TensorRT version. On a different GPU (A100, L4, RTX 40xx…), rebuild them by running cells 5–10.

---

## Results (Tesla T4)

| Run | Audio | Avatar | Output | Length | Avg time / chunk |
|---|---|---|---|---|---|
| Test | `example/speaker_1.ogg` | `maria` | `avtr_test.mp4` | 60.2 s (1505 frames) | ~453 ms |
| Custom | Hindi `convo_001.mp3` | `doctor` | `doctor_avatar.mp4` | 64.2 s (1605 frames) | ~457 ms |

Each chunk is 5 frames (0.2 s of video at 25 fps), so a T4 renders at about **0.44× real-time**. A 1-minute clip takes about 2.3 minutes. Getting real-time output needs a faster GPU (e.g. L4/A10 or better).

---

## Troubleshooting

- **`[system-requirements]` deprecation warning from Pixi.** This is harmless and comes from the upstream `pixi.toml`.
- **ONNX Runtime `Error merging shape info ... 'boxes1'` warnings.** These are harmless; it falls back to lenient merge.
- **`mp3 ... Estimating duration from bitrate`.** This is harmless for MP3 input. Convert to WAV if you need exact timing.
- **Disk full.** `.pixi` (12 GB) plus the build artifacts (2.4 GB) come close to the 20 GB limit. Delete `artifacts/main/build_artifacts` after the engines are built (cell 13).
- **"Missing engines" in cell 11.** A build script failed. Re-run cells 7, 9 and 10 and check each one's log.
- **Avatar looks cropped or distorted.** Use a higher-resolution, centred portrait, and re-run cell 17.
