# Hướng dẫn setup & chạy pipeline (Ubuntu + RTX 5090)

Máy lab giả định: Ubuntu · 1× RTX 5090 32GB · CUDA 13.2.

Pipeline gốc (CVPR 2025) giữ nguyên logic & chất lượng (LISA fp16, SD2 inpaint 512).

**Triết lý đường dẫn**

1. Toàn bộ code + sibling clones + mask/output nằm **trong repo** (ví dụ `MVP1/`).
2. Cache Hugging Face (model nặng) cấu hình bằng file **`.env`** — mặc định `HF_HOME=/backup/data/art-gen`.

---

## Quy tắc conda env (đọc trước khi chạy)

Có **đúng 2 env**. Không bao giờ activate nhầm khi chạy lệnh.

| Env | Dùng khi nào | Không dùng để |
|-----|----------------|---------------|
| **`lisa`** | Chỉ khi **bật LISA server** (`scripts/start_lisa.sh`) | Không chạy `main.py`, Gradio UI, `run_lisa_batch.py` |
| **`amodal`** | Mọi thứ còn lại: setup check GPU, tải model phụ, **gọi LISA batch**, **full pipeline**, Gradio UI | Không start server LISA |

**Script nào tự activate env?**

| Script | Tự `conda activate` | Bạn cần làm gì |
|--------|---------------------|----------------|
| `scripts/start_lisa.sh` | → **`lisa`** | Chỉ cần `source scripts/paths.env` rồi chạy script |
| `scripts/start_gradio.sh` | → **`amodal`** | Như trên |
| `scripts/run_full_batch.sh` | → **`amodal`** (sau khi LISA đã chạy ở terminal khác) | Như trên |
| `scripts/stop_lisa.sh` | Không cần GPU env | Chạy ở terminal nào cũng được |
| `scripts/env.sh` / `model.sh` | Dùng `conda run -n ...` | Không cần activate trước |

Nếu gọi **trực tiếp** `python ...` (không qua script trên) thì **phải** `conda activate amodal` trước.

**Port**

| Dịch vụ | Port | Env |
|---------|------|-----|
| LISA server | **7860** | `lisa` |
| Gradio full pipeline | **7861** | `amodal` |

**Thứ tự VRAM bắt buộc (1 GPU)**

```text
START LISA (env lisa, :7860)
    → cache mask (env amodal gọi server)
    → STOP LISA  ← giải phóng VRAM
    → full pipeline (env amodal, CLI hoặc Gradio :7861)
```

Không chạy LISA và full pipeline **cùng lúc**.

---

## 0. Cấu trúc thư mục

```text
MVP1/                                 # AMODAL_ROOT
├── .env                              # tạo thủ công từ .env.example
├── .env.example
├── LISA/app.py                       # patch (copy sang third_party/LISA)
├── third_party/
│   ├── LISA/                         # upstream LISA + weights
│   ├── InstaOrder/
│   ├── Grounded-Segment-Anything/
│   └── recognize-anything/
├── InstaOrder -> third_party/...     # symlink (model.sh tạo)
├── cache/lisa_masks/                 # mask .pkl sau bước LISA
├── output/
├── scripts/
├── main.py
└── gradio_app.py
```

HF cache (ổ lớn, ngoài repo):

```text
/backup/data/art-gen/                 # HF_HOME
```

---

## 1. Setup một lần (chỉ làm khi mới cài)

### 1.1 Điều kiện

- Driver CUDA 13.2 (`nvidia-smi`)
- Miniconda/Anaconda, `git`, `wget`/`curl`
- Token Hugging Face
- Thư mục `/backup/data/art-gen` ghi được (hoặc đổi trong `.env`)

### 1.2 Clone + `.env`

```bash
git clone <URL_REPO> MVP1
cd MVP1

cp .env.example .env
nano .env
```

Tối thiểu trong `.env`:

```bash
HF_HOME=/backup/data/art-gen
HF_TOKEN=hf_xxxxxxxx
```

```bash
sudo mkdir -p /backup/data/art-gen
sudo chown "$USER":"$USER" /backup/data/art-gen
```

### 1.3 Tạo 2 conda env + cài thư viện

```bash
cd /path/to/MVP1
source scripts/paths.env
bash scripts/env.sh          # tạo env amodal + lisa (không dùng -y; xác nhận tay)
```

Nếu `env.sh` dừng giữa chừng (ví dụ lỗi CLIP) mà **torch đã cài xong** cho cả hai env: **không cần xóa env**. Chỉ cài tiếp phần còn thiếu:

```bash
source scripts/paths.env
conda activate amodal
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation \
  "git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1"
conda activate lisa
python -m pip install -r requirements-lisa.txt
conda activate amodal
python scripts/check_gpu.py
```

### 1.4 Tải model / clone sibling

```bash
source scripts/paths.env
bash scripts/model.sh        # clone third_party/, tải ckpt, prefetch SD vào HF_HOME
```

Không tải LaMa.

### 1.5 Kiểm tra GPU (env **amodal**)

```bash
source scripts/paths.env
conda activate amodal
python scripts/check_gpu.py
```

Kỳ vọng: có `sm_120`, `matmul_ok=True`.

(Tuỳ chọn) đăng nhập HF:

```bash
conda activate amodal
source scripts/paths.env
huggingface-cli login
```

### 1.6 Chuẩn bị input trước khi chạy

- Ảnh trong repo (ví dụ `images_example/`)
- `img_filenames_example.txt` — mỗi dòng một path relative tới `--input_dir`
- `example_annotation.json` — mỗi ảnh một `labels[0]` (text query)

---

## 2. Cách chạy A — Batch CLI (2 terminal)

Dùng khi xử lý nhiều ảnh theo list + JSON.

### Tổng quan thời gian

| Bước | Việc | Terminal | Env | LISA server |
|------|------|----------|-----|-------------|
| A1 | **START** LISA | T1 | `lisa` (script tự activate) | **ON** :7860 |
| A2 | Cache mask toàn bộ ảnh | T2 | `amodal` | vẫn ON |
| A3 | **STOP** LISA | T2 | bất kỳ | **OFF** |
| A4 | Full pipeline (`main.py`) | T2 | `amodal` | phải OFF |

### Terminal 1 — chỉ giữ LISA server

Mở terminal 1, để nguyên đến khi xong bước cache mask:

```bash
cd /path/to/MVP1
source scripts/paths.env
bash scripts/start_lisa.sh
# Script tự: conda activate lisa
# Đợi log kiểu: Running on local URL: http://0.0.0.0:7860
# Để terminal này chạy — ĐỪNG Ctrl+C cho tới khi Terminal 2 báo đã stop
```

### Terminal 2 — cache mask → stop → main

**Cách nhanh (khuyến nghị):** một script làm A2→A4:

```bash
cd /path/to/MVP1
source scripts/paths.env
bash scripts/run_full_batch.sh
# Script hỏi Enter khi LISA đã ready → nhấn Enter
# Tự: activate amodal → run_lisa_batch → stop_lisa → main.py theo batch
```

**Cách tách tay** (cùng Terminal 2, env **amodal**):

```bash
cd /path/to/MVP1
source scripts/paths.env
conda activate amodal

# A2 — LISA vẫn đang ON ở Terminal 1
python scripts/run_lisa_batch.py \
  --input_dir "$AMODAL_ROOT" \
  --img_filenames_txt ./img_filenames_example.txt \
  --json_label_path ./example_annotation.json \
  --lisa_mask_dir "$LISA_OUTPUT_PATH"

# A3 — STOP LISA (giải phóng VRAM). Có thể chạy ở T1 hoặc T2.
bash scripts/stop_lisa.sh
# Kiểm tra: nvidia-smi  → không còn process LISA

# A4 — full pipeline (LISA phải đã OFF)
python main.py \
  --input_dir "$AMODAL_ROOT" \
  --img_filenames_txt ./img_filenames_example.txt \
  --json_label_path ./example_annotation.json \
  --output_dir "$OUTPUT_DIR" \
  --lisa_mask_dir "$LISA_OUTPUT_PATH" \
  --skip_lisa_server \
  --require_lisa_cache \
  --line_num 0 \
  --round_number 5
```

Hoặc sau khi đã có đủ `.pkl` và đã `stop_lisa`:

```bash
source scripts/paths.env
conda activate amodal
bash main_batch_example.sh
```

**Sau A3:** Terminal 1 sẽ bị kill / hết process — bình thường. Không cần `start_lisa` lại trừ khi muốn cache mask thêm ảnh mới.

---

## 3. Cách chạy B — Gradio UI (2 terminal)

Dùng khi thử từng ảnh trên web UI (`:7861`).

### Tổng quan thời gian

| Bước | Việc | Terminal | Env | LISA |
|------|------|----------|-----|------|
| B1 | **START** LISA | T1 | `lisa` | **ON** |
| B2 | **START** Gradio UI | T2 | `amodal` | vẫn ON |
| B3 | Trong UI: Cache LISA mask | trình duyệt | — | ON |
| B4 | Trong UI: **Stop LISA** (hoặc `stop_lisa.sh`) | UI / T2 | — | **OFF** |
| B5 | Trong UI: Run main pipeline | trình duyệt | — | phải OFF |

### Terminal 1 — LISA

```bash
cd /path/to/MVP1
source scripts/paths.env
bash scripts/start_lisa.sh
# Đợi ready trên :7860 — giữ terminal này
```

### Terminal 2 — Gradio full pipeline

```bash
cd /path/to/MVP1
source scripts/paths.env
bash scripts/start_gradio.sh
# Script tự: conda activate amodal
# Mở http://<lab-ip>:7861  (hoặc localhost:7861)
```

### Trong trình duyệt (đúng thứ tự nút)

1. Upload ảnh + nhập text query  
2. Bấm **Cache LISA mask** (LISA đang ON ở T1) → stem được điền  
3. Bấm **Stop LISA server** (VRAM trống)  
4. Bấm **Run main pipeline**  

Không bấm Run main trong khi LISA còn sống.

Khi xong hẳn: Ctrl+C Terminal 2 (Gradio). LISA đã stop ở bước 3.

---

## 4. START / STOP nhanh (cheat sheet)

| Muốn… | Lệnh | Env |
|-------|------|-----|
| Bật LISA | `source scripts/paths.env && bash scripts/start_lisa.sh` | tự `lisa` |
| Tắt LISA | `bash scripts/stop_lisa.sh` | không cần |
| Batch mask + main | T1: start_lisa · T2: `bash scripts/run_full_batch.sh` | T2 tự `amodal` |
| Chỉ main (đã có `.pkl`, LISA đã tắt) | `conda activate amodal` rồi `python main.py ...` hoặc `main_batch_example.sh` | `amodal` |
| Gradio UI | T1: start_lisa · T2: `bash scripts/start_gradio.sh` | T2 tự `amodal` |

**Nhớ:** START LISA trước khi cache mask · STOP LISA trước khi full pipeline.

---

## 5. SD2 inpainting & OOM

Model: `sd2-community/stable-diffusion-2-inpainting` (đổi trong `.env` bằng `SD_MODEL_ID` nếu cần).

OOM khi chạy main (LISA đã tắt rồi mà vẫn thiếu VRAM): thêm `--sd_cpu_offload`.

---

## 6. Xử lý sự cố

### Nhầm env
- Server LISA phải là env **`lisa`** (`start_lisa.sh`).
- `run_lisa_batch.py` / `main.py` / Gradio phải là **`amodal`**.
- Triệu chứng hay gặp: import lỗi / transformers version conflict → đang activate sai env.

### `ModuleNotFoundError: No module named 'pkg_resources'`
### `AttributeError: 'FieldInfo' object has no attribute 'in_'`
### `ModuleNotFoundError: No module named 'gradio_client.serializing'`

Ba lỗi này đều do **stack Gradio 3.39 trong env `lisa` bị lệch version** (setuptools mới / pydantic 2 / `gradio_client` 1.x–2.x).

**Fix nhanh (env hiện tại — không xóa env):**

```bash
cd /path/to/MVP1
source scripts/paths.env
bash scripts/fix_lisa_deps.sh
bash scripts/start_lisa.sh
```

Hoặc tay:

```bash
conda activate lisa
python -m pip install \
  "setuptools>=68,<81" \
  "pydantic==1.10.13" \
  "fastapi==0.100.1" \
  "starlette==0.27.0" \
  "gradio_client==0.5.0" \
  "gradio==3.39.0" \
  "huggingface_hub>=0.16,<0.23" \
  "websockets>=10,<12" \
  "uvicorn==0.23.2"
```

Kiểm tra version đúng:

```bash
conda activate lisa
python -c "import pydantic,gradio,gradio_client; from gradio_client.serializing import JSONSerializable; print(pydantic.__version__, gradio.__version__, gradio_client.__version__)"
# Kỳ vọng: 1.10.x  3.39.x  0.5.x
```

Log `[LISA] DeepSpeed init_inference failed ... using plain fp16` là **bình thường** (fallback sẵn).

### Gradio UI crashes on open (`bool is not iterable` / `unhashable type: 'dict'` / localhost)
Ba lỗi thường đi cùng lệch stack Gradio 4.44:

| Lỗi | Nguyên nhân |
|-----|-------------|
| `argument of type 'bool' is not iterable` | `pydantic>=2.11` |
| `unhashable type: 'dict'` (Jinja2 TemplateResponse) | `starlette>=1.0` |
| `localhost is not accessible` | lab fail health-check → cần `gradio_app.py` mới |

```bash
conda activate amodal
git pull
bash scripts/fix_amodal_deps.sh   # pins gradio + pydantic<2.11 + starlette<1
source scripts/paths.env
bash scripts/start_gradio.sh
# Mở http://<IP-lab>:7861
```

Hotfix tạm (chưa pull được):

```bash
conda activate amodal
python -m pip install \
  "gradio==4.44.1" "gradio_client==1.3.0" \
  "pydantic>=2.0,<2.11" "starlette>=0.37,<1.0" "fastapi>=0.111,<0.116"
# kiểm tra: python -c "import starlette,pydantic,gradio; print(starlette.__version__, pydantic.__version__, gradio.__version__)"
# kỳ vọng: starlette 0.3x–0.4x, pydantic 2.x (<2.11), gradio 4.44.x
```

### `cannot import name 'SamPredictor' from 'segment_anything' (unknown location)`
Hai nguyên nhân thường gặp:

1. Package pip hỏng/stub trong env `amodal`
2. Process **Gradio cũ** vẫn chạy (import `main` lần đầu bị lỗi / path sai) — **không cần** `conda deactivate`

```bash
conda activate amodal
source scripts/paths.env

# cài lại SAM (nếu chưa)
python -m pip uninstall -y segment-anything segment_anything
python -m pip install -e "$GROUNDED_SAM_REPO/segment_anything"

# bắt buộc: kill Gradio cũ rồi start lại (Ctrl+C hoặc:)
pkill -f "python gradio_app.py" || true
# kiểm tra import giống Gradio (qua main.py):
python -c "import main; from segment_anything import SamPredictor; print('OK')"

bash scripts/start_gradio.sh
```

Mask LISA đã cache vẫn dùng lại — không cần Start LISA / cache lại.

### `ModuleNotFoundError: No module named 'fairscale'`
RAM (`recognize-anything`) cần `fairscale`:

```bash
conda activate amodal
python -m pip install "fairscale>=0.4.13"
# hoặc: bash scripts/fix_amodal_deps.sh
python -c "import main; print('OK')"
```

### `ModuleNotFoundError: No module named 'clip'`
OpenAI CLIP phải cài riêng (`--no-build-isolation`): `bash scripts/fix_amodal_deps.sh`.

### `cannot import name 'Dinov2WithRegistersConfig'` (amodal / start_gradio / main.py)
`diffusers 0.39` cần `transformers` rất mới. Repo **cố ý pin** `diffusers==0.32.2` + `transformers==4.46.3` (đủ cho SD2 inpaint).

**Fix nhanh (env hiện tại):**

```bash
cd /path/to/MVP1
source scripts/paths.env
bash scripts/fix_amodal_deps.sh
python scripts/verify_amodal.py
bash scripts/start_gradio.sh
```

### OOM
Thường do LISA chưa tắt: `bash scripts/stop_lisa.sh` rồi `nvidia-smi`.

### Port bị chiếm
```bash
lsof -iTCP:7860 -sTCP:LISTEN
lsof -iTCP:7861 -sTCP:LISTEN
```

### HF 401 / rate limit
Kiểm tra `.env` (`HF_TOKEN`, `HF_HOME=/backup/data/art-gen`).

### Checkpoint 0 byte / `InstaOrder` treo khi load / `verify_checkpoints` FAIL
Download hỏng (URL GitHub release InstaOrder 404) để lại file **0 byte**. `start_gradio.sh` giờ chạy `scripts/verify_checkpoints.py` trước khi launch.

```bash
conda activate amodal
source scripts/paths.env
# xem file hỏng:
du -h "$INSTAORDER_CKPT" "$SAM_CKPT" "$GDINO_CKPT" "$RAM_CKPT"
# xóa stub + tải lại (InstaOrder qua Google Drive / gdown)
rm -f -- "$INSTAORDER_CKPT"
bash scripts/model.sh
python scripts/verify_checkpoints.py
bash scripts/start_gradio.sh
```

Tải tay InstaOrderNet_od (nếu `model.sh` fail):

```bash
conda activate amodal
source scripts/paths.env          # bắt buộc — set $INSTAORDER_CKPT
python -m pip install -q "gdown>=5.0"
# gdown>=6 bỏ --fuzzy; truyền file id trực tiếp:
python -m gdown 1QLikFxNOEW1Ld2oAZff8mL26FO4Mwwpv -O "$INSTAORDER_CKPT"
du -h "$INSTAORDER_CKPT"   # phải >> 0 (thường ~100MB+)
echo "$INSTAORDER_CKPT"    # kiểm tra path không rỗng
```

### Thiếu mask `.pkl`
Chưa chạy bước cache LISA, hoặc đã stop sớm. Xem `cache/lisa_masks/`.

### DeepSpeed LISA lỗi
`LISA/app.py` có fallback fp16 `.cuda()`.

---

## 7. Tóm tắt thay đổi so với repo gốc

| Hạng mục | Bản này |
|----------|---------|
| Workspace | Trong repo MVP1 |
| Sibling | `third_party/` |
| HF cache | `.env` → `/backup/data/art-gen` |
| Torch | 2.13.0+cu132, Python 3.11 |
| SD | `sd2-community/...` |
| LaMa | Bỏ |
| Chạy | LISA batch → **STOP** → main/Gradio |
| Env | `lisa` = server · `amodal` = mọi thứ khác |
