# Aether — VR Agent System

基于 Google Gemini Robotics ER 2 + DartControl + Gemini TTS 的 VR 虚拟角色驱动系统。

## 架构

```
ER2 Streaming (规划层, 云端) → 函数调用 → DART (动作层, 本地 7900XT) → VRChat OSC
                                       → TTS (语音层, 云端) → 音频播放
```

- **ER2 Streaming**: 视觉感知 + 任务规划 + 自然对话，通过 Live API WebSocket 工作
- **DART (DartControl)**: 文本→SMPL-X 人体骨架数据，ICLR 2025 Spotlight，本地 ROCm GPU 推理
- **Gemini TTS**: 文本→语音合成，支持 200+ 情感标签和 30 种预置语音
- **Sub-agent Runtime**: 上下文管理（重要性评分、RAG 召回、分级驱逐、心跳生成）

## 快速开始

### 环境要求
- Python 3.10+
- Windows（主程序）+ WSL2 Ubuntu（DART 服务）
- AMD RX 7900 XT（或其他 ROCm 兼容 GPU）
- Google AI Studio API Key

### 安装

```bash
# Windows 侧
pip install -r requirements.txt

# WSL2 侧（DART 服务）
# 参见 DART 项目部署文档
```

### 配置

设置 API Key：
```bash
export GEMINI_API_KEY="your-api-key"
```

编辑 `config.yaml` 调整端口、语音、OSC 等参数。

### 运行

```bash
# 1. 启动 DART 服务（WSL2）
wsl -- bash -c "source /root/pytorch_rocm/bin/activate && cd /root/DART && uvicorn server:app --host 0.0.0.0 --port 8900"

# 2. 启动 Agent
python -m aether
```

### CLI 命令
- `/status` — 系统状态
- `/tokens` — Token 用量
- `/memory` — 上下文记忆统计
- `/idle` — 发送空闲姿态
- `/motion <text>` — 测试动作生成（如 `/motion walk forward*5`）
- `/tts <text>` — 测试语音合成（如 `/tts 你好`）
- `/quit` — 退出

## 项目结构

```
aether/
├── __init__.py          # 包定义
├── __main__.py          # python -m aether 入口
├── main.py              # AetherAgent 主控制器
├── config.py            # Pydantic 配置管理
├── logging_config.py    # CLI 彩色日志 + 文件日志
├── er2_client.py        # ER2 Live API WebSocket 客户端
├── tool_dispatcher.py   # 函数调用分发器
├── dart_client.py       # DART HTTP 客户端
├── tts_engine.py        # TTS 兼容层
├── audio_pipeline.py    # 音频输入/输出管道
├── osc_sender.py        # VRChat OSC 骨骼映射
├── audio/               # TTS 引擎子包
│   ├── base.py          # TTS 抽象基类
│   ├── tts.py           # Gemini TTS 分段引擎
│   └── chatterbox.py    # 语音克隆扩展
└── context/             # Sub-agent Runtime
    ├── block_store.py   # 上下文块存储 + FAISS 索引
    ├── importance_scorer.py  # 重要性评分器
    ├── evictor.py       # 分级上下文驱逐器
    ├── rag_retriever.py # RAG 召回器
    ├── heartbeat.py     # 心跳生成器
    └── task_tracker.py  # 任务状态追踪
```

## 技术栈
- **AI**: Google Gemini ER2 Streaming, Gemini TTS, Gemini Embedding
- **动作生成**: DartControl (ICLR 2025), SMPL-X
- **通信**: Live API (WebSocket), HTTP, OSC (UDP)
- **音频**: sounddevice, Silero VAD
- **向量检索**: FAISS

## License
Apache 2.0
