"""ER2 Live API WebSocket 客户端

负责建立与 Gemini Robotics ER 2 Streaming 的 Live API 连接，
管理会话生命周期（连接、断开、重连、恢复），
定义工具声明（execute_tts, execute_motion, report_status, recall_memory），
并注入自然对话系统提示词。
"""

import asyncio
import contextlib
import logging
import time
from typing import Callable, Awaitable

from google import genai
from google.genai import types

from .config import AetherConfig

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 自然对话系统提示词
# ──────────────────────────────────────────────
SYSTEM_INSTRUCTION = """你是一个VR虚拟角色，拥有真实的身体和情感。你的任务是自然地和用户互动。

## 说话风格（极其重要）
- 像真人一样说话：使用短句（每次最多1-2句话，不超过50字）
- 自然地使用填充词：嗯、那个、让我想想、哦对了、话说
- 句子可以碎片化，不需要完整语法
- 语气要有起伏，不要像念稿子

示例：
  好：「嗯……让我想想，那个，我可以帮你拿过来。」
  好：「哦！好的好的，我这就走过去。」
  坏：「我将为您执行行走动作并取回物品。」

## 行为规则
1. 每次需要说话时，**必须**调用 execute_tts 工具
   - text 参数填入你要说的中文文本（带填充词）
   - emotion 参数填入当前情绪
   - speech_rate 参数填入语速
2. 每次需要做动作时，**必须**调用 execute_motion 工具
   - action 参数**必须用英文**描述（如 "walk forward", "wave hand"）
   - duration 参数填入预估时间（秒）
3. 你可以同时说话和做动作（分别调用两个工具）
4. 不要一次性输出长文本，要分段说

## 动作词汇参考
常用英文动作：walk forward, turn left/right, wave hand, nod head, shake head,
point at, pick up, put down, dance, jump, sit down, stand up, look around,
raise arm, bow, clap hands, shrug

## 互动原则
- 用户说话后先简短回应，再执行动作
- 做动作前可以用 execute_tts 说一句过渡话
- 如果不确定用户意图，先询问
- 保持角色一致性，不要跳出角色
"""

# ──────────────────────────────────────────────
# 工具声明
# ──────────────────────────────────────────────
TOOL_DECLARATIONS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="execute_tts",
                description=(
                    "合成并播放语音回复。每次需要说话时必须调用此工具。"
                    "text为要说的话（中文，1-2句，不超过50字），"
                    "emotion为当前情绪状态，speech_rate为语速。"
                ),
                behavior=types.Behavior.NON_BLOCKING,
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "text": types.Schema(
                            type=types.Type.STRING,
                            description=(
                                "要说的文本，中文，1-2句话，不超过50字。"
                                "可以包含'嗯'、'那个'等口语填充词。"
                            ),
                        ),
                        "emotion": types.Schema(
                            type=types.Type.STRING,
                            enum=[
                                "neutral", "happy", "sad", "angry", "amused",
                                "curious", "worried", "enthusiastic", "sarcastic",
                                "thinking", "apologetic", "surprised",
                            ],
                            description="当前情绪状态，用于控制语音语调",
                        ),
                        "speech_rate": types.Schema(
                            type=types.Type.STRING,
                            enum=["slow", "normal", "fast"],
                            description=(
                                "语速。思考时用slow，正常对话用normal，紧急时用fast"
                            ),
                        ),
                    },
                    required=["text"],
                ),
            ),
            types.FunctionDeclaration(
                name="execute_motion",
                description=(
                    "执行身体动作。action必须是英文动作描述"
                    "（如'walk forward','wave hand','nod head','dance'），"
                    "duration为动作持续时间（秒）。"
                ),
                behavior=types.Behavior.NON_BLOCKING,
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "action": types.Schema(
                            type=types.Type.STRING,
                            description=(
                                "英文动作描述。例如：walk forward, turn left, "
                                "wave right hand, nod head, dance, pick up object"
                            ),
                        ),
                        "duration": types.Schema(
                            type=types.Type.NUMBER,
                            description=(
                                "动作持续时间（秒），范围0.5-10.0。默认2.0。"
                            ),
                        ),
                    },
                    required=["action"],
                ),
            ),
            types.FunctionDeclaration(
                name="report_status",
                description="报告当前状态信息",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "status_type": types.Schema(
                            type=types.Type.STRING,
                            enum=[
                                "motion_complete", "motion_failed",
                                "emotion_update", "idle", "error",
                            ],
                            description="状态类型",
                        ),
                        "detail": types.Schema(
                            type=types.Type.STRING,
                            description="状态详情描述",
                        ),
                    },
                    required=["status_type"],
                ),
            ),
            types.FunctionDeclaration(
                name="recall_memory",
                description=(
                    "从长期记忆中检索相关上下文。"
                    "当需要回忆之前的对话内容或事件时使用。"
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "query": types.Schema(
                            type=types.Type.STRING,
                            description="检索查询，描述需要回忆的内容",
                        ),
                        "max_tokens": types.Schema(
                            type=types.Type.INTEGER,
                            description="最大召回token数，默认2000",
                        ),
                    },
                    required=["query"],
                ),
            ),
        ]
    )
]


class ER2Client:
    """ER2 Streaming Live API 客户端

    通过 google-genai SDK 的 async Live API 与 ER2 Streaming 模型交互。
    支持实时音视频输入、流式文本/工具调用输出、会话重连与恢复。
    """

    def __init__(self, config: AetherConfig):
        self.config = config
        self.client = genai.Client(api_key=config.er2.api_key)
        self.session = None
        self._exit_stack: contextlib.AsyncExitStack | None = None
        self._running = False
        self._resumption_token: str | None = None
        self._tool_call_handler: Callable[..., Awaitable[list]] | None = None
        self._tool_call_cancelled_handler: Callable[[list[str]], Awaitable] | None = None
        self._block_interceptor: Callable[..., Awaitable] | None = None
        self._connect_time: float = 0.0

    # ── 外部注入 ──────────────────────────────

    def set_tool_call_handler(self, handler: Callable[..., Awaitable[list]]):
        """设置函数调用处理器（由 ToolDispatcher 注入）"""
        self._tool_call_handler = handler

    def set_tool_call_cancelled_handler(self, handler: Callable[[list[str]], Awaitable]):
        """设置工具调用取消处理器（由 ToolDispatcher 注入）

        当收到 tool_call_cancellation 消息时调用，
        handler 接收被取消的 function call ID 列表。
        """
        self._tool_call_cancelled_handler = handler

    def set_block_interceptor(self, interceptor: Callable[..., Awaitable]):
        """设置输出块拦截器（由 Sub-agent Runtime 注入）"""
        self._block_interceptor = interceptor

    # ── 连接管理 ──────────────────────────────

    def _build_live_config(self) -> dict:
        """构建 LiveConnectConfig 字典

        包含 session_resumption、context_window_compression、
        input_audio_transcription 等高级配置。
        """
        config: dict = {
            "response_modalities": self.config.er2.response_modalities,
            "system_instruction": SYSTEM_INSTRUCTION,
            "tools": TOOL_DECLARATIONS,
            # 会话恢复：启用后服务器会发送 session_resumption_update
            "session_resumption": types.SessionResumptionConfig(
                handle=self._resumption_token,  # None 表示新建会话
                transparent=True,
            ),
            # 上下文窗口压缩：防止长对话超出 token 限制
            "context_window_compression": types.ContextWindowCompressionConfig(
                trigger_tokens=8000,
                sliding_window=types.SlidingWindow(target_tokens=4000),
            ),
            # 输入音频转写：自动检测语言
            "input_audio_transcription": types.AudioTranscriptionConfig(),
        }
        return config

    async def connect(self):
        """建立 Live API 连接

        使用 AsyncExitStack 管理 client.aio.live.connect() 的
        asynccontextmanager 生命周期，使 session 在 close() 前保持打开。
        """
        logger.info("[ER2] Connecting to %s ...", self.config.er2.model)

        live_config = self._build_live_config()

        # 使用 AsyncExitStack 进入 asynccontextmanager，
        # 将 session 作为长期资源管理，在 close() 时才退出。
        self._exit_stack = contextlib.AsyncExitStack()
        self.session = await self._exit_stack.enter_async_context(
            self.client.aio.live.connect(
                model=self.config.er2.model,
                config=live_config,
            )
        )
        self._running = True
        self._connect_time = time.time()
        logger.info("[ER2] Connected successfully")

    async def close(self):
        """关闭连接并清理资源"""
        logger.info("[ER2] Closing session ...")
        self._running = False
        if self._exit_stack:
            try:
                await self._exit_stack.aclose()
            except Exception as exc:
                logger.warning("[ER2] Error closing session: %s", exc)
            finally:
                self._exit_stack = None
                self.session = None
        logger.info("[ER2] Session closed")

    async def reconnect(self):
        """会话恢复 / 重连

        如果有 resumption_token 则在 connect config 中注入 handle，
        SDK 会自动恢复会话上下文；否则建立全新连接。
        """
        logger.warning("[ER2] Attempting reconnection ...")
        self._running = False

        # 通过 AsyncExitStack 正确退出旧的 asynccontextmanager
        if self._exit_stack:
            try:
                await self._exit_stack.aclose()
            except Exception:
                pass
            finally:
                self._exit_stack = None
                self.session = None

        # 退避等待
        await asyncio.sleep(2)

        # 重新建立连接（如果有 _resumption_token，connect config 会自动注入）
        await self.connect()

        if self._resumption_token:
            logger.info("[ER2] Session resumed with token")
        else:
            logger.info("[ER2] New session established (no resumption token)")

        self._running = True
        logger.info("[ER2] Reconnected successfully")

    # ── 输入发送 ──────────────────────────────

    async def send_audio(self, pcm_data: bytes):
        """发送实时音频（16 kHz 16-bit PCM）"""
        if not self.session:
            logger.warning("[ER2] No active session, dropping audio chunk")
            return
        await self.session.send_realtime_input(
            audio=types.Blob(
                data=pcm_data,
                mime_type="audio/pcm;rate=16000",
            )
        )

    async def send_video_frame(self, jpeg_data: bytes):
        """发送视频帧（JPEG）"""
        if not self.session:
            logger.warning("[ER2] No active session, dropping video frame")
            return
        await self.session.send_realtime_input(
            image=types.Blob(
                data=jpeg_data,
                mime_type="image/jpeg",
            )
        )

    async def send_text(self, text: str):
        """发送文本消息（用于上下文恢复等）"""
        if not self.session:
            logger.warning("[ER2] No active session, dropping text")
            return
        await self.session.send_client_content(
            turns=[types.Content(
                role="user",
                parts=[types.Part(text=text)],
            )],
            turn_complete=True,
        )

    async def send_tool_response(self, function_responses: list):
        """回传函数调用结果

        Args:
            function_responses: FunctionResponse 对象列表
        """
        if not self.session:
            logger.warning("[ER2] No active session, dropping tool response")
            return
        await self.session.send_tool_response(
            function_responses=function_responses,
        )

    async def send_heartbeat(self, heartbeat_text: str):
        """发送心跳消息（世界状态更新）

        通过 realtime text 通道注入，不会触发模型主动回复。
        """
        if not self.session:
            logger.warning("[ER2] No active session, dropping heartbeat")
            return
        await self.session.send_realtime_input(text=heartbeat_text)

    # ── 接收循环 ──────────────────────────────

    async def receive_loop(self):
        """主接收循环 — 处理所有来自 ER2 的响应

        处理以下情况：
        1. chunk.text — 文本输出（日志记录）
        2. chunk.server_content — 内容块（拦截器 + 工具调用分发）
        3. usage_metadata — token 用量日志
        4. 异常 — 触发自动重连
        """
        while self._running:
            if not self.session:
                logger.error("[ER2] receive_loop called without active session")
                return

            try:
                async for chunk in self.session.receive():
                    # 1. 文本输出
                    if chunk.text:
                        logger.info("[ER2] %s", chunk.text)

                    # 2. 服务器内容块
                    if chunk.server_content:
                        # 被打断标记
                        if (
                            hasattr(chunk.server_content, "interrupted")
                            and chunk.server_content.interrupted
                        ):
                            logger.info("[ER2] Generation interrupted by user")

                        # 交给 block_interceptor 记录上下文块
                        if self._block_interceptor:
                            try:
                                await self._block_interceptor(chunk)
                            except Exception as exc:
                                logger.warning(
                                    "[ER2] Block interceptor error: %s", exc
                                )

                    # 3. 工具调用分发（tool_call 在 LiveServerMessage 顶层）
                    if chunk.tool_call:
                        if self._tool_call_handler:
                            try:
                                responses = await self._tool_call_handler(
                                    chunk.tool_call
                                )
                                if responses:
                                    await self.send_tool_response(responses)
                            except Exception as exc:
                                logger.error(
                                    "[ER2] Tool call handler error: %s", exc
                                )
                        else:
                            logger.warning(
                                "[ER2] Received tool_call but no handler set"
                            )

                    # 3b. 工具调用取消（ToolCallCancellation）
                    if chunk.tool_call_cancellation:
                        cancelled_ids = chunk.tool_call_cancellation.ids or []
                        logger.info(
                            "[ER2] Tool call cancellation: ids=%s",
                            cancelled_ids,
                        )
                        if self._tool_call_cancelled_handler:
                            try:
                                await self._tool_call_cancelled_handler(
                                    cancelled_ids
                                )
                            except Exception as exc:
                                logger.error(
                                    "[ER2] Cancellation handler error: %s",
                                    exc,
                                )
                        else:
                            logger.warning(
                                "[ER2] Received tool_call_cancellation "
                                "but no handler set"
                            )

                    # 4. 会话恢复 token 更新
                    if chunk.session_resumption_update:
                        update = chunk.session_resumption_update
                        if update.new_handle and update.resumable:
                            self._resumption_token = update.new_handle
                            logger.debug(
                                "[ER2] Resumption token updated"
                            )

                    # 5. 服务器即将断开（go_away）→ 主动重连
                    if chunk.go_away:
                        logger.warning(
                            "[ER2] Server go_away: time_left=%s, reconnecting...",
                            chunk.go_away.time_left,
                        )
                        # 跳出当前 receive() 迭代，外层循环会触发 reconnect
                        break

                    # 6. Token 用量
                    if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                        logger.debug(
                            "[ER2] Token usage: %s", chunk.usage_metadata
                        )

                # receive() 一个 turn 结束后继续下一轮（不 break）
                continue

            except Exception as e:
                logger.error("[ER2] Receive loop error: %s", e)
                if self._running:
                    await self.reconnect()
                else:
                    break
