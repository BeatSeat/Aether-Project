# DART 动作生成服务部署

DART (DartControl, ICLR 2025 Spotlight) 将文本描述转换为 SMPL-X 人体骨架动画数据。

- 原始仓库: https://github.com/zkf1997/DART
- 本目录包含经过修改的 DART 源码，已内置 PyTorch3D 绕过补丁，支持 NVIDIA (CUDA) 和 AMD (ROCm) GPU。

## 前置条件

- Python 3.10+
- NVIDIA GPU (CUDA) 或 AMD GPU (ROCm, 仅 Linux)
- 约 8GB GPU 显存

## 安装

### 1. 进入 DART 目录

```bash
cd dart
```

### 2. 创建虚拟环境并安装 PyTorch

```bash
python -m venv venv
source venv/bin/activate        # Linux / WSL2
# venv\Scripts\activate         # Windows
```

**NVIDIA (CUDA):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**AMD (ROCm, 仅 Linux/WSL2):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2.4
```

### 3. 安装 DART 依赖

```bash
pip install -r requirements_dart.txt
```

> **注意**: PyTorch3D 不需要安装。代码已内置替代方案 (`utils/rotation_conversions.py`)，
> 所有原本依赖 `pytorch3d.transforms` 的地方都已替换为内置实现。
> 同样，`pyrender` 和 `open3d` 也不需要（仅可视化/渲染用，推理不需要）。

### 4. 下载模型权重

从 DART 原始仓库获取预训练权重:
- 仓库: https://github.com/zkf1997/DART
- 下载 checkpoint 文件放到: `mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt`

或通过环境变量指定其他路径:
```bash
export DART_CHECKPOINT=/path/to/checkpoint_300000.pt
```

### 5. 下载 SMPL-X 模型文件

**必需** — SMPL-X 模型文件用于前向运动学 (FK)，将旋转参数转换为关节世界坐标位置。

1. 注册 https://smpl-x.is.tue.mpg.de/ (需同意使用条款)
2. 下载 SMPL-X 模型文件 (`.npz` 格式，包含 MALE/FEMALE/NEUTRAL)
3. 放置到以下目录结构:

```
dart/data/smplx_lockedhead_20230207/models_lockedhead/smplx/
├── SMPLX_MALE.npz
├── SMPLX_FEMALE.npz
└── SMPLX_NEUTRAL.npz
```

> 没有 SMPL-X 模型文件，服务可以输出 poses (旋转参数) 和 betas (体型参数)，
> 但 **joints (关节位置) 将为空**。建议始终安装以获得完整输出。

### 6. 准备 standing pose 数据

`data/stand.pkl` 和 `data/stand_20fps.pkl` 是推理所需的站立姿态种子文件。
这些文件已包含在仓库中（体积小）。如果缺失，从 DART 原始仓库的 `data/` 目录获取。

## 运行

```bash
cd dart
uvicorn server:app --host 0.0.0.0 --port 8900
```

首次启动会加载模型到 GPU（约 5 秒），之后推理延迟约 1-3 秒。

## API

### `POST /generate`

请求:
```json
{"text": "walk forward*5"}
```

可选参数: `batch_size` (默认 1), `guidance_param` (默认 5.0)

响应:
```json
{
  "poses": [[...], ...],      // (T, 165) SMPL-X 轴角参数，前66维为身体关节，后99维零填充
  "trans": [[...], ...],      // (T, 3) 每帧全局位移
  "betas": [0.0, ...],       // (10,) 体型参数
  "joints": [[[...], ...]],  // (T, 22, 3) 22个关节的世界坐标位置 (由 SMPL-X FK 计算)
  "framerate": 30,
  "num_frames": 150,
  "text_prompt": "walk forward*5"
}
```

> **joints 输出说明**: joints 由 SMPL-X body model 前向运动学计算，
> 是所有旋转参数在全局坐标系下的关节位置，可直接用于驱动 VRChat 骨骼。
> 如果没有安装 SMPL-X 模型文件，joints 将为空。

### `GET /health`

返回 GPU 状态和模型加载状态。

### `GET /status`

返回模型信息和当前生成状态。
