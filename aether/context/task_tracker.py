"""任务状态追踪器 — 从 ER2 输出中解析和维护任务状态"""
import logging

from .heartbeat import TaskState

logger = logging.getLogger(__name__)


class TaskTracker:
    """从 ER2 输出块中追踪任务状态变化"""

    def __init__(self):
        self.state = TaskState()
        self._task_history: list[dict] = []   # 历史任务记录

    def update_from_block(self, block_type: str, content: str, metadata: dict = None):
        """从上下文块更新状态"""
        metadata = metadata or {}

        # 从工具调用中提取信息
        if block_type == "tool_call":
            if "execute_motion" in content:
                # 提取动作信息
                pass
            elif "execute_tts" in content:
                # 提取情绪信息
                pass

    def get_state(self) -> TaskState:
        """获取当前任务状态"""
        return self.state

    def get_history_summary(self, max_items: int = 5) -> str:
        """获取历史任务摘要"""
        recent = self._task_history[-max_items:]
        if not recent:
            return "No task history"
        return "; ".join(
            f"{t.get('action', 'unknown')} ({t.get('status', '?')})"
            for t in recent
        )
