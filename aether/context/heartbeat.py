"""心跳生成器与任务状态追踪"""
import hashlib
import logging
import time
from dataclasses import dataclass, field
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TaskState:
    """当前任务状态"""
    current_task: str = "idle"
    recent_actions: deque = field(default_factory=lambda: deque(maxlen=5))
    known_objects: dict = field(default_factory=dict)
    environment_summary: str = "Unknown environment"
    emotion_state: str = "neutral"


class HeartbeatGenerator:
    """心跳生成器

    定期生成世界状态摘要，通过 send_realtime_input 注入 ER2。
    状态哈希去重：无变化时不发送。
    """

    def __init__(self, interval_seconds: int = 30):
        self.interval = interval_seconds
        self.task_state = TaskState()
        self._last_heartbeat_time = 0
        self._last_state_hash = ""

    def update_from_tool_call(self, function_name: str, args: dict):
        """从工具调用更新任务状态"""
        if function_name == "execute_motion":
            action = args.get("action", "unknown")
            self.task_state.recent_actions.append(action)
            self.task_state.current_task = f"executing: {action}"
        elif function_name == "execute_tts":
            self.task_state.emotion_state = args.get("emotion", "neutral")
        elif function_name == "report_status":
            status_type = args.get("status_type", "")
            if status_type == "motion_complete":
                self.task_state.current_task = "idle"
            elif status_type == "idle":
                self.task_state.current_task = "idle"

    def update_from_text(self, text: str):
        """从 ER2 文本输出推断状态更新（简单关键词匹配）"""
        # 简单的状态推断，不需要复杂的 NLP
        pass

    def should_send(self) -> bool:
        """检查是否应该发送心跳"""
        return time.time() - self._last_heartbeat_time >= self.interval

    def generate(self) -> Optional[str]:
        """生成心跳消息，如果状态无变化返回 None"""
        state = self.task_state
        recent = list(state.recent_actions)

        heartbeat = (
            f"[WORLD STATE UPDATE]\n"
            f"Task: {state.current_task}\n"
            f"Emotion: {state.emotion_state}\n"
            f"Recent Actions: {', '.join(recent[-3:]) if recent else 'none'}\n"
            f"Known Objects: {', '.join(state.known_objects.keys()) if state.known_objects else 'none'}\n"
            f"[END UPDATE]"
        )

        # 状态哈希去重
        state_hash = hashlib.md5(heartbeat.encode()).hexdigest()[:8]
        if state_hash == self._last_state_hash:
            return None   # 状态未变化，跳过

        self._last_state_hash = state_hash
        self._last_heartbeat_time = time.time()
        return heartbeat
