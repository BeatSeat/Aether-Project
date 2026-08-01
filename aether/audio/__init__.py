"""音频合成子包

提供统一的语音合成接口，支持：
- TTS 分段引擎（Gemini TTS API）
- Chatterbox 语音克隆（扩展）
"""

from .tts import TTSEngine
from .chatterbox import ChatterboxEngine
from .base import BaseTTSEngine

__all__ = ["TTSEngine", "ChatterboxEngine", "BaseTTSEngine"]
