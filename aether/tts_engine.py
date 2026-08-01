"""TTS 语音合成引擎（向后兼容模块）

实际实现已迁移到 aether.audio 子包。
此模块保留向后兼容导入。
"""

# 从新位置重新导出，保持向后兼容
from .audio.tts import TTSEngine, EMOTION_TO_TAGS, RATE_TO_TAG
from .audio.base import BaseTTSEngine

__all__ = ["TTSEngine", "BaseTTSEngine", "EMOTION_TO_TAGS", "RATE_TO_TAG"]
