"""
Finance RAG Main Pipeline (对齐 financebench-rag-eval 的 CRAG 流程)
==================================================================
流程（宽进严出 + 2-step relax + 分层评测）:

    1. query_preprocessor.preprocess()
       → search_query + filter_token(公司+年) + company_filter(仅公司)
    2. HybridVectorStore.search(query_filter=...)
       → 带.metadata 过滤的混合检索（宽进 k=12）
       → 结果为空时 2-step relax: 公司+年 → 仅公司 → NONE（对齐 nodes/router.py）
    3. get_reranker().rerank()
       → CrossEncoder 重排，严出 top=5（对齐 nodes/rerank.py）
    4. LLM generate
       → 基于 top-k 上下文生成答案并引用来源
    5. RAGEvaluator
       → Tier1 检索指标(Recall@k/Precision@k/MRR) + Tier2 LLM-as-Judge 四轴

运行方式:
    cd D:\finance_rag\finance_rag_evaluation
    python main_pipeline.py
"""
import sys
from unittest.mock import MagicMock

# 模拟整个 vertexai 子模块，防止 ragas 导入时报错
sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()
import os
import time
import logging

# 修复 Windows 终端 GBK 编码问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from qdrant_client.models import Filter, FieldCondition, MatchText

from src.core.query_preprocessor import QueryPreprocessor
from src.core.reranker import get_reranker
from src.core.vectorstore import HybridVectorStore
from langchain_openai import ChatOpenAI


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ──────────────────────────────────────────────
#  配置
# ──────────────────────────────────────────────
QDRANT_PATH = r"D:\finance_rag\finance_rag_evaluation\qdrant_data"
DATA_DIR = r"D:\finance_rag\finance_rag_evaluation\data"

RETRIEVE_K = 12      # 宽进：过滤检索数量（financebench-rag-eval: k=8~12）
RERANK_TOP_K = 5     # 严出：重排后保留数量（financebench-rag-eval: top=5）
MAX_RETRIES = 2      # relax 重试上限（公司+年 → 仅公司 → NONE）


# 新增：带 Citation 的 Prompt
GENERATE_PROMPT_WITH_CITATION = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a financial analyst assistant. Answer the user's question using ONLY the retrieved documents. "
        "CRITICAL: You MUST cite your sources using [1], [2], etc. at the end of sentences or paragraphs. "
        "The number corresponds to the document index provided in the context (starting from 1). "
        "If the documents don't contain the answer, say 'I don't know' and do not invent facts."
    ),
    (
        "human",
        "Retrieved documents (indexed):\n{context}\n\nQuestion: {question}"
    ),
])

def build_llm():
    """
    构建生成/评测用的 LLM。
    默认本地 Ollama（免 API Key）；如需 OpenAI，替换为:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    """
    llm=ChatOpenAI(
        model="qwen3.7-plus",
        api_key= os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0
    )
    return llm


# ──────────────────────────────────────────────
#  主 Pipeline
# ──────────────────────────────────────────────
class FinanceRAGPipeline:
    """CRAG 风格主流水线：预处理 → 过滤检索(relax) → 重排 → 生成。"""

    def __init__(self, llm, store: Optional[HybridVectorStore] = None, use_rewrite: bool = False,use_adaptive: bool = True):
        self.llm = llm
        self.store = store or self._init_store()
        self.preprocessor = QueryPreprocessor(self.llm, use_rewrite=use_rewrite)
        self.reranker = get_reranker()  # 单例 CrossEncoder，全局只加载一次
        self.use_adaptive = use_adaptive
        self._init_generate_chain()

    @staticmethod
    def _init_store() -> HybridVectorStore:
        store = HybridVectorStore(qdrant_path=QDRANT_PATH)
        if not store.load_index():
            # 索引不存在时从 PDF 构建（首次运行）
            from src.core.loader import PDFLoader

            logging.info("索引不存在，从 PDF 构建索引 ...")
            loader = PDFLoader(data_dir=DATA_DIR, chunk_size=512, chunk_overlap=50)
            chunks = loader.load_all_pdfs()
            if not chunks:
                raise RuntimeError(f"数据目录无可用 PDF: {DATA_DIR}")
            store.build_index(chunks)
        return store

    def _init_generate_chain(self):
        # 统一 Prompt：CoT 推理 + 结构化输出 + 强 Citation 编号（与 _parse_citations 配套）
        self.generate_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a financial analyst assistant. Answer the user's question using "
                    "ONLY the retrieved documents provided in the context.\n\n"
                    "REASONING (Chain-of-Thought): First identify and quote the relevant figures, "
                    "dates, or statements from the documents, then reason step by step toward the "
                    "conclusion before giving the final answer.\n\n"
                    "OUTPUT STRUCTURE: Structure your answer in three sections:\n"
                    "  (1) Conclusion — the direct answer in 1-2 sentences.\n"
                    "  (2) Reasoning — step-by-step analysis citing the specific data you used.\n"
                    "  (3) Citations — the list of [n] references you relied on.\n\n"
                    "CITATION (mandatory): You MUST cite sources inline using [1], [2], ... where the "
                    "number matches the document index in the context (starting from 1). Every "
                    "quantitative claim or specific fact must carry a citation. Do not fabricate "
                    "numbers or citations.\n\n"
                    "If the documents don't contain the answer, say 'I don't know' and do not invent facts.",
                ),
                (
                    "human",
                    "Retrieved documents (indexed):\n\n{context}\n\nQuestion: {question}",
                ),
            ]
        )
        self.generate_chain = self.generate_prompt | self.llm

    # ── 检索（带 2-step relax）──

    @staticmethod
    def _build_filter(filter_token: str) -> Optional[Filter]:
        """把 filter_token（如 AMD_2022）转成 Qdrant metadata 过滤器（子串匹配文件名）。"""
        if not filter_token or filter_token == "NONE":
            return None
        return Filter(
            must=[
                FieldCondition(
                    key="metadata.original_filename",
                    match=MatchText(text=filter_token),
                )
            ]
        )

    def retrieve_with_relax(
        self,
        search_query: str,
        filter_token: str,
        company_filter: str,
        mode: str = "hybrid",
    ) -> tuple:
        """
        带过滤检索；结果为空时按 financebench-rag-eval 的 2-step relax 放宽：
          retry 0: 公司+年 → 仅公司
          retry 1: 仅公司 → NONE

        Args:
            mode: 检索模式 'dense' / 'sparse' / 'hybrid'（量化对比时可切换）。

        Returns:
            (docs, final_filter_token)
        """
        current_token = filter_token
        docs: List[Dict[str, Any]] = []
        for retry in range(MAX_RETRIES + 1):
            query_filter = self._build_filter(current_token)
            docs = self.store.search(
                search_query, top_k=RETRIEVE_K, mode=mode, query_filter=query_filter
            )
            if docs:
                if retry > 0:
                    logging.info(f"Relax 成功（retry={retry}）: filter='{current_token}'")
                return docs, current_token
            current_token = QueryPreprocessor.relax_filter_token(
                filter_token, company_filter, retry
            )
            logging.warning(f"过滤检索为空 (filter='{current_token}')，relax 后重试 ...")
        return docs, current_token

    # ── 完整问答 ──

    def answer(self, query: str, retrieval_mode: str = "hybrid") -> Dict[str, Any]:
        """
        端到端问答：预处理 → 检索(宽进 k=12 + relax) → 重排(严出 top=5) → 生成。

        Args:
            retrieval_mode: 检索模式 'dense' / 'sparse' / 'hybrid'（量化对比时可切换）。

        Returns:
            {
                "question":     原始问题,
                "answer":       生成的答案,
                "contexts":     重排后的上下文文本列表（供 Tier2 评测用）,
                "sources":      重排后上下文来源文件名列表（供 Tier1 评测用）,
                "sources_pre_rerank": 重排前（宽进 k=12）来源文件名列表,
                "citations":    引用列表,
                "filter_token": 最终使用的过滤 token,
                "strategy_used"/"query_type": 预处理路由信息,
                "timings":      各阶段耗时(ms),
            }
        """
        timings: Dict[str, float] = {}

        # 1. 预处理：结构化过滤抽取（+ 可选改写）
        t0 = time.perf_counter()
        if self.use_adaptive:
            pre = self.preprocessor.preprocess_adaptive(query)
        else:
            pre_old = self.preprocessor.preprocess(query)
            pre = {**pre_old, "strategy": "original", "use_filter": True, "query_type": "manual"}

        search_query, filter_token, company_filter, use_filter = (
            pre["search_query"], pre["filter_token"], pre["company_filter"], pre["use_filter"]
        )
        timings["preprocess_ms"] = (time.perf_counter() - t0) * 1000
        logging.info(f"[1/4] 预处理: search_query='{search_query}', filter='{filter_token}'")

        # 2. 检索（宽进 k=12，空结果 relax）
        # use_filter=False（复杂查询）时 relax 无意义，直接全库检索
        t0 = time.perf_counter()
        if use_filter:
            docs, final_token = self.retrieve_with_relax(
                search_query, filter_token, company_filter, mode=retrieval_mode
            )
        else:
            docs = self.store.search(
                search_query, top_k=RETRIEVE_K, mode=retrieval_mode, query_filter=None
            )
            final_token = "NONE"
        timings["retrieve_ms"] = (time.perf_counter() - t0) * 1000
        sources_pre_rerank = [d["metadata"].get("original_filename", "Unknown") for d in docs]
        logging.info(f"[2/4] 检索({retrieval_mode}): {len(docs)} docs")

        # 3. 重排（严出 top=5）
        t0 = time.perf_counter()
        reranked = self.reranker.rerank(search_query, docs, top_k=RERANK_TOP_K) if docs else []
        timings["rerank_ms"] = (time.perf_counter() - t0) * 1000
        logging.info(f"[3/4] 重排: {len(reranked)} docs")

        # 4. 生成 (带 Citation 格式化)
        t0 = time.perf_counter()
        if reranked:
            # 格式化 Context，加上索引 [1], [2]...
            context_str = "\n\n".join(
                f"[{i + 1}] Source: {d['metadata'].get('original_filename', 'Unknown')} (page {d.get('page', 0)})\n{d['text']}"
                for i, d in enumerate(reranked)
            )

            raw_answer = self.generate_chain.invoke(
                {"context": context_str, "question": query}
            ).content.strip()

            # 5. 后处理：解析 Citation
            final_answer, citations = self._parse_citations(raw_answer, reranked)
        else:
            final_answer = "I don't know — no relevant documents were found."
            citations = []
        timings["generate_ms"] = (time.perf_counter() - t0) * 1000

        return {
            "question": query,
            "answer": final_answer,
            "contexts": [d["text"] for d in reranked],
            "citations": citations,
            "sources": [d["metadata"].get("original_filename", "Unknown") for d in reranked],
            "sources_pre_rerank": sources_pre_rerank,
            "filter_token": final_token,
            "strategy_used": pre["strategy"],
            "query_type": pre["query_type"],
            "retrieval_mode": retrieval_mode,
            "timings": timings,
        }

    def _parse_citations(self, raw_answer: str, docs: List[Dict]) -> tuple:
        """
        解析 LLM 输出的 [1], [2] 引用，映射回具体文档信息
        """
        import re
        citations = []
        # 匹配 [1], [2] 等标记
        pattern = r'\[(\d+)\]'
        found_indices = set(int(m) for m in re.findall(pattern, raw_answer))

        for idx in found_indices:
            if 1 <= idx <= len(docs):
                doc = docs[idx - 1]
                citations.append({
                    "index": idx,
                    "source": doc["metadata"].get("original_filename", "Unknown"),
                    "page": doc.get("page", 0),
                    "snippet": doc["text"][:100] + "..."  # 截取前100字作为预览
                })

        return raw_answer, citations
    # ── 批量评测（Tier1 + Tier2）──

    def evaluate(
        self,
        questions: List[str],
        ground_truths: List[str],
        ground_truth_docs: List[str],
        use_judge: bool = True,
    ) -> Dict[str, Any]:
        """
        对问题集跑完整 pipeline，然后:
          Tier1: recall@k / precision@k / mrr（无需 LLM）
          Tier2: LLM-as-Judge 四轴（可选）

        Args:
            questions: 问题列表。
            ground_truths: 标准答案列表（Tier2 用）。
            ground_truth_docs: 正确文档名列表（Tier1 用，如 "AMD_2022_10K.pdf"）。
        """
        from evaluation.rag_evaluator import RAGEvaluator, compute_retrieval_metrics

        results = [self.answer(q) for q in questions]
        # sources 为重排后 top-5（进入生成上下文的文档），Tier1 用它衡量最终召回
        retrieved_sources = [r["sources"] for r in results]
        contexts = [r["contexts"] for r in results]
        answers = [r["answer"] for r in results]

        tier1 = compute_retrieval_metrics(
            retrieved_sources=retrieved_sources,
            ground_truth_docs=ground_truth_docs,
            k=RERANK_TOP_K,
        )

        output: Dict[str, Any] = {"tier1": tier1, "results": results}

        if use_judge:
            evaluator = RAGEvaluator(eval_llm=self.llm)
            tier2 = evaluator.evaluate_tier2(
                questions=questions,
                ground_truths=ground_truths,
                contexts=contexts,
                answers=answers,
            )
            output["tier2"] = tier2

        return output


# ──────────────────────────────────────────────
#  Demo
# ──────────────────────────────────────────────
def main():
    print("=" * 80)
    print("Finance RAG Pipeline — CRAG 流程演示")
    print("=" * 80)

    llm = build_llm()
    pipeline = FinanceRAGPipeline(llm, use_rewrite=False)

    question = "What was AMD's net income in 2022?"
    print(f"\nQuestion: {question}\n")

    result = pipeline.answer(question)

    print(f"\nFilter token: {result['filter_token']}")
    print(f"Sources: {result['sources']}")
    print(f"\nAnswer:\n{result['answer']}")

    # 评测示例（Tier1 无需 LLM judge 之外的调用；Tier2 使用同一 LLM）
    print("\n" + "=" * 80)
    print("评测示例（单样本）")
    print("=" * 80)
    eval_output = pipeline.evaluate(
        questions=[question],
        ground_truths=["AMD's net income was $1.3 billion in 2022."],
        ground_truth_docs=["AMD_2022_10K.pdf"],
        use_judge=True,
    )
    print(f"\nTier1: {eval_output['tier1']}")
    if "tier2" in eval_output:
        print(f"Tier2: {eval_output['tier2']['summary']}")


if __name__ == "__main__":
    main()
