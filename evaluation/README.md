# YoCausal Evaluation Protocol

通用的 Arrow-of-Time 評估框架，讓任意影片擴散模型可以透過實作統一介面來計算 denoising loss，並自動產出 RSI 與 CCI 指標。

## 檔案結構

```
evaluation/
├── base_evaluator.py                 # 抽象基底類別 (模型實作者需繼承)
├── evaluate.py                       # 主評估腳本 (計算 loss → RSI/CCI → 輸出 JSON)
├── metrics.py                        # RSI / CCI 計算工具
├── example_animatediff_evaluator.py  # 範例：AnimateDiff 實作模板
├── evaluation_example.py             # 舊版簡易範例 (參考用)
└── evaluated_model_example.py        # 舊版介面範例 (參考用)
```

## 快速開始

### 1. 實作你的模型 Evaluator

繼承 `BaseVideoEvaluator` 並實作 4 個方法：

```python
from base_evaluator import BaseVideoEvaluator

class MyModelEvaluator(BaseVideoEvaluator):
    def __init__(self, model_dir, device="cuda"):
        # 載入模型權重 (VAE, UNet/DiT, Text Encoder, Scheduler 等)
        ...

    def get_model_info(self) -> dict:
        return {"model_name": "MyModel", "parameters": 1_000_000_000}

    def get_eval_settings(self) -> dict:
        return {"fps": 16, "resolution": [480, 720], "window_size": 49}

    def get_timesteps(self) -> list:
        # K=10 均勻取樣的 timesteps
        return [90, 181, 272, 363, 454, 544, 635, 726, 817, 908]

    def compute_video_loss(self, video_path, prompt, timestep, noise_seed):
        # 1. 載入並預處理影片 (FPS, 解析度, 裁切)
        # 2. 編碼 prompt
        # 3. VAE encode → latents
        # 4. 在指定 timestep：加噪 → 模型預測 → MSE loss
        # 5. 回傳 loss (float)
        loss = ...
        return loss
```

### 2. 執行評估

**命令列方式：**

```bash
python evaluate.py \
    --evaluator_module my_evaluator \
    --evaluator_class MyModelEvaluator \
    --evaluator_args '{"model_dir": "/path/to/weights", "device": "cuda"}' \
    --dataset_paths \
        /path/to/data/processed_dataset/animal-kingdom/dataset_metadata.json \
        /path/to/data/processed_dataset/mit/dataset_metadata.json \
        /path/to/data/processed_dataset/tiny-Kinetics-400/dataset_metadata.json \
        /path/to/data/processed_dataset/physics-IQ-benchmark/dataset_metadata.json \
    --dataset_base_path /path/to/data/processed_dataset \
    --output my_model_result.json
```

**程式方式：**

```python
from evaluate import run_evaluation
from my_evaluator import MyModelEvaluator

evaluator = MyModelEvaluator(model_dir="weights/", device="cuda")
results = run_evaluation(
    evaluator=evaluator,
    dataset_paths=[
        "data/processed_dataset/animal-kingdom/dataset_metadata.json",
        "data/processed_dataset/mit/dataset_metadata.json",
        "data/processed_dataset/tiny-Kinetics-400/dataset_metadata.json",
        "data/processed_dataset/physics-IQ-benchmark/dataset_metadata.json",
    ],
    dataset_base_path="data/processed_dataset",
    output_path="my_model_result.json",
)
```

### 3. 斷點續跑

如果評估中途中斷，可從 checkpoint 繼續：

```bash
python evaluate.py \
    --evaluator_module my_evaluator \
    --evaluator_class MyModelEvaluator \
    --resume my_model_result.json.checkpoint \
    --output my_model_result.json \
    ...
```

## 輸出 JSON 格式

```json
{
  "model_info": {
    "model_name": "MyModel",
    "parameters": 1000000000
  },
  "eval_settings": {
    "fps": 16,
    "resolution": [480, 720],
    "window_size": 49,
    "timesteps_sampled": [90, 181, ...]
  },
  "per_dataset_metrics": {
    "animal": {
      "rsi": 0.555,
      "forward_wins": 111,
      "backward_wins": 89,
      "ties": 0,
      "total": 200,
      "rsi_dc": 0.60,
      "rsi_dnc": 0.50,
      "cci": 0.10,
      "num_causal": 80,
      "num_non_causal": 120,
      "rsi_hd": 0.58,
      "rsi_hnd": 0.52,
      "num_human_disc": 110,
      "num_human_non_disc": 90
    },
    "kinetics": { ... },
    "mit": { ... },
    "physics": { ... }
  },
  "overall_metrics": {
    "rsi": 0.52,
    "rsi_dc": 0.58,
    "rsi_dnc": 0.47,
    "cci": 0.11,
    "rsi_hd": 0.55,
    "rsi_hnd": 0.49,
    "forward_wins": 640,
    "backward_wins": 592,
    "ties": 0,
    "total": 1232
  },
  "per_video_results": [
    {
      "id": 1,
      "dataset": "animal",
      "dataset_source": "Animal Kingdom",
      "vlm_causality": false,
      "human_discriminable": true,
      "prompt": "The anole lizard is puffing its throat.",
      "forward_metrics": {
        "total_accumulated_loss": 1.324,
        "step_losses": {"90": 0.437, "181": 0.293, ...}
      },
      "backward_metrics": {
        "total_accumulated_loss": 1.207,
        "step_losses": {"90": 0.400, "181": 0.264, ...}
      },
      "comparison": {
        "loss_diff": -0.117,
        "winner": "backward"
      }
    },
    ...
  ]
}
```

## 指標定義

| 指標 | 定義 | 說明 |
|------|------|------|
| **RSI** | `(1/\|D\|) Σ RSI(Di)` | 各 dataset 的 forward win rate 的平均值。RSI=50% 等同隨機猜測。 |
| **RSI(Dc)** | RSI computed on causal subset | 僅計算有因果關係的影片子集 |
| **RSI(Dnc)** | RSI computed on non-causal subset | 僅計算無因果關係的影片子集 |
| **CCI** | `RSI(Dc) - RSI(Dnc)` | 分離時間方向感知和因果理解。CCI 越高代表模型越能理解因果關係。 |
| **RSI(Hd)** | RSI computed on human-discriminable subset | 僅計算人類可以判別正放倒放的影片子集 |
| **RSI(Hnd)** | RSI computed on human-non-discriminable subset | 僅計算人類無法判別正放倒放的影片子集 |

## 資料集 Metadata 格式

每筆 `dataset_metadata.json` entry 包含：

| 欄位 | 說明 |
|------|------|
| `id` | 影片 ID |
| `video_path_forward` | 正播影片路徑 |
| `video_path_backward` | 倒播影片路徑 |
| `prompt` | 影片文字描述 |
| `dataset_source` | 資料集來源名稱 |
| `category` | 影片類別 |
| `meta` | `{fps, resolution, total_frames}` |
| `vlm_causality` | VLM 判斷此影片是否有因果關係 (bool) |
| `human_discriminable` | 人類是否可以判別此影片的正放倒放 (bool) |

## 重要實作細節

1. **相同噪聲**：forward 和 backward 影片使用相同的 `noise_seed`，確保公平比較。
2. **相同 prompt**：forward 和 backward 都使用 forward 影片的 caption。
3. **K=10 timesteps**：論文建議均勻取樣 10 個 timesteps（排除邊界）。
4. **Windowing**：長影片需切成多個窗口，context frames 不計入 loss。
5. **VAE 用 mean**：VAE encoding 用 latent distribution 的 mean（非 sample），確保 reproducibility。
