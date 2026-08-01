"""上下文块存储与 FAISS 向量索引管理"""
import uuid
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ContextBlock:
    """单个上下文块"""
    block_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    block_type: str = ""           # "text" | "tool_call" | "tool_response"
    content: str = ""              # 原始内容
    timestamp: float = field(default_factory=time.time)
    token_estimate: int = 0        # token 估算
    importance_score: float = 0.5  # 0.0 ~ 1.0
    embedding: Optional[np.ndarray] = None  # 嵌入向量
    summary: Optional[str] = None  # 压缩后的摘要
    metadata: dict = field(default_factory=dict)


class ContextBlockStore:
    """上下文块存储"""

    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.blocks: dict[str, ContextBlock] = {}  # block_id -> ContextBlock
        self.total_active_tokens: int = 0
        self._faiss_index = None  # 延迟初始化（需要 faiss-cpu）
        self._id_to_index: dict[str, int] = {}  # block_id -> FAISS 索引位置
        self._next_index: int = 0
        self._pending_embeddings: list[ContextBlock] = []  # 待嵌入的块

    def _init_faiss(self):
        """延迟初始化 FAISS 索引"""
        try:
            import faiss
            self._faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # 内积相似度
        except ImportError:
            logger.warning("[BlockStore] faiss-cpu not installed, RAG retrieval will be disabled")

    def add_block(self, block: ContextBlock):
        """添加新块（embedding 可稍后异步填充）"""
        self.blocks[block.block_id] = block
        self.total_active_tokens += block.token_estimate
        if block.embedding is not None:
            self._add_to_faiss(block)
        else:
            self._pending_embeddings.append(block)

    def _add_to_faiss(self, block: ContextBlock):
        """将块的 embedding 添加到 FAISS"""
        if self._faiss_index is None:
            self._init_faiss()
        if self._faiss_index is None:
            return
        vec = block.embedding.reshape(1, -1).astype(np.float32)
        # 归一化用于内积相似度
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        self._faiss_index.add(vec)
        self._id_to_index[block.block_id] = self._next_index
        self._next_index += 1

    def remove_block(self, block_id: str):
        """移除块"""
        if block_id in self.blocks:
            block = self.blocks[block_id]
            self.total_active_tokens -= block.token_estimate
            del self.blocks[block_id]
            # FAISS 不支持单条删除（IndexFlatIP），标记为已删除即可
            if block_id in self._id_to_index:
                del self._id_to_index[block_id]

    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[ContextBlock, float]]:
        """搜索最相似的块"""
        if self._faiss_index is None or self._faiss_index.ntotal == 0:
            return []

        vec = query_embedding.reshape(1, -1).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        scores, indices = self._faiss_index.search(vec, min(top_k, self._faiss_index.ntotal))

        results = []
        # 反向映射 index -> block_id
        index_to_id = {v: k for k, v in self._id_to_index.items()}
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx in index_to_id:
                block_id = index_to_id[idx]
                if block_id in self.blocks:
                    results.append((self.blocks[block_id], float(score)))

        return results

    def get_blocks_by_importance(self, threshold: float = 0.3) -> list[ContextBlock]:
        """获取重要性低于阈值的块"""
        return [b for b in self.blocks.values() if b.importance_score < threshold]

    def get_old_blocks(self, max_age_seconds: float = 300) -> list[ContextBlock]:
        """获取超过指定年龄的块"""
        cutoff = time.time() - max_age_seconds
        return [b for b in self.blocks.values() if b.timestamp < cutoff]

    def estimate_token_count(self, text: str) -> int:
        """简单的 token 估算（1 token ≈ 4 字符英文 / 2 字符中文）"""
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        return max(1, chinese_chars // 2 + other_chars // 4)

    def get_summary(self) -> dict:
        """获取存储状态摘要"""
        return {
            "total_blocks": len(self.blocks),
            "total_active_tokens": self.total_active_tokens,
            "pending_embeddings": len(self._pending_embeddings),
            "faiss_size": self._faiss_index.ntotal if self._faiss_index else 0,
        }
