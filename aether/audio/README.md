# Audio 子包

语音合成模块，提供统一的 TTS 接口。

## 组件

| 文件 | 说明 |
|------|------|
| `base.py` | TTS 引擎抽象基类 |
| `tts.py` | Gemini TTS 分段引擎（默认） |
| `chatterbox.py` | Chatterbox 语音克隆（扩展） |

## 使用

```python
from aether.audio import TTSEngine, ChatterboxEngine

# 默认 TTS（Gemini API）
tts = TTSEngine(config.tts)
await tts.synthesize("Hello world", emotion="happy", speech_rate="normal")

# 语音克隆（需安装 chatterbox）
clone = ChatterboxEngine(device="cuda", reference_audio="voice_sample.wav")
await clone.synthesize("Hello world")
```

## 分割模式

`_split_text()` 支持两种标点模式：

- `"english"`（默认）：按 `. , ! ? ;` 分割，适用于 TTS 语音输出
- `"chinese"`：按 `。 ， ！ ？ ；` 分割，适用于中文文本

```python
segments = tts._split_text("Hello, world. How are you?", split_on="english")
```
