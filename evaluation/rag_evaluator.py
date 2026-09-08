"""
RAG Evaluation Module (Multi-tier Evaluation, 对齐 financebench-rag-eval 的 eval 体系)
====================================================================================
参照 financebench-rag-eval（eval/Phase02 judge.py + metrics.py、Phase03 judge_v2.py）
的三层评测思路：

Tier 1 — 无 LLM 检索指标（参照 metrics.py compute_tier1）:
    Recall@k / Precision@k / MRR，基于 ground-truth 文档名，零成本、可高频回归。

Tier 2 — LLM-as-Judge 四轴（参照 judge.py / judge_v2.py，1-5 分 + JSON 输出）:
    C|Q  上下文相关性: 检索上下文是否包含回答问题所需信息
    A|C  忠实度:       答案中每个事实性断言是否可溯源到上下文（幻觉检测）
    A|Q  答案相关性:   答案是否正面回答了所问的指标/期间/单位
    A|GT 答案正确性:   对照 ground truth，含 judge_v2 的数值提取 + 5%/15% 容差规则

Tier 3 — Ragas（保留原有能力）:
    context_precision / faithfulness。
"""

import json
import logging
from typing import List, Dict, Any

import pandas as pd
from datasets import Dataset

# Ragas imports（需要 `pip install ragas datasets`）
from ragas.metrics import context_precision, faithfulness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ══════════════════════════════════════════════
#  Tier 1 — 检索指标（无 LLM，参照 Phase02/metrics.py）
# ══════════════════════════════════════════════
def compute_retrieval_metrics(
    retrieved_sources: List[List[str]],
    ground_truth_docs: List[str],
    k: int = 6,
) -> Dict[str, Any]:
    """
    基于 ground-truth 文档名计算 Recall@k / Precision@k / MRR。

    Args:
        retrieved_sources: 每个问题检索到的文档名列表（如 ["AMD_2022_10K.pdf", ...]）。
        ground_truth_docs: 每个问题对应的正确文档名（如 "AMD_2022_10K.pdf"）。
        k: 截断位置（仅影响指标命名，序列本身应已按检索排序）。

    Returns:
        {"n", "recall@k", "precision@k", "mrr", "misses_by_doc"}
    """
    assert len(retrieved_sources) == len(ground_truth_docs), "样本数不一致"

    recalls, precisions, mrrs = [], [], []
    misses: List[str] = []

    for sources, gt in zip(retrieved_sources, ground_truth_docs):
        sources = sources or []
        # Recall@k: 正确文档是否被检回
        hit = 1.0 if gt in sources else 0.0
        recalls.append(hit)
        if hit == 0.0:
            misses.append(gt)

        # Precision@k: 检回结果中来自正确文档的比例
        precisions.append(sum(s == gt for s in sources) / len(sources) if sources else 0.0)

        # MRR: 第一个正确文档的倒数排名
        rr = 0.0
        for rank, s in enumerate(sources, 1):
            if s == gt:
                rr = 1.0 / rank
                break
        mrrs.append(rr)

    n = len(ground_truth_docs)
    metrics = {
        "n": n,
        f"recall@{k}": round(sum(recalls) / n, 4) if n else 0.0,
        f"precision@{k}": round(sum(precisions) / n, 4) if n else 0.0,
        "mrr": round(sum(mrrs) / n, 4) if n else 0.0,
    }

    # 按文档统计检索失败次数（定位哪些 PDF 常态检不回）
    misses_by_doc: Dict[str, int] = {}
    for doc in misses:
        misses_by_doc[doc] = misses_by_doc.get(doc, 0) + 1
    metrics["misses_by_doc"] = dict(sorted(misses_by_doc.items(), key=lambda x: -x[1]))

    logging.info(
        f"Tier1 Retrieval Metrics (n={n}, k={k}): recall@{k}={metrics[f'recall@{k}']}, "
        f"precision@{k}={metrics[f'precision@{k}']}, mrr={metrics['mrr']}"
    )
    return metrics


# ══════════════════════════════════════════════
#  Tier 2 — LLM-as-Judge（参照 Phase02/judge.py + Phase03/judge_v2.py）
# ══════════════════════════════════════════════
_CQ_PROMPT = """\
You are evaluating a retrieval-augmented generation system for financial documents (SEC filings).

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

Task: Does the retrieved context contain the information needed to answer the question?

For financial questions, consider:
- Is the context from the correct company?
- Does it cover the correct fiscal year / period?
- Does it contain the specific financial metric being asked about?

Score on a scale of 1 to 5:
1 = Completely irrelevant (wrong company, wrong period, wrong metric)
2 = Mentions the company but lacks the specific data needed
3 = Partially relevant (related data, but not the exact answer)
4 = Contains the answer but requires significant inference or calculation
5 = Directly and explicitly contains data to answer the question

Respond with valid JSON only:
{{"score": <1-5>, "reasoning": "<one sentence>"}}"""

_AC_PROMPT = """\
You are evaluating a retrieval-augmented generation system for financial documents (SEC filings).

RETRIEVED CONTEXT:
{context}

MODEL ANSWER:
{answer}

Task: Is every factual claim in the model answer supported by the retrieved context?

Pay special attention to:
- Specific numbers, percentages, dollar amounts — are they in the context?
- Company names, fiscal years — do they match?
- Claims that sound plausible but are not in the context (hallucination)

Score on a scale of 1 to 5:
1 = Major claims are not in the context (clear hallucination)
2 = Most claims are not supported by the context
3 = Some claims supported, some not — mixed faithfulness
4 = Almost all claims supported; minor unsupported details
5 = Every factual claim is directly traceable to the provided context

Respond with valid JSON only:
{{"score": <1-5>, "reasoning": "<one sentence>"}}"""

_AQ_PROMPT = """\
You are evaluating a retrieval-augmented generation system for financial documents (SEC filings).

QUESTION:
{question}

MODEL ANSWER:
{answer}

Task: Does the model answer address the question that was asked?

For financial questions, consider:
- Does it provide the specific metric requested (e.g., revenue, capex, net income)?
- Does it cover the correct time period (fiscal year, quarter)?
- Is the unit correct (USD millions, percentage, ratio)?
- A response of "I don't know" or "not found in context" scores 1 even if technically honest.

Score on a scale of 1 to 5:
1 = Does not answer the question (refuses, goes off-topic, or says "I don't know")
2 = Addresses the topic but not the specific question asked
3 = Partially answers (right direction but incomplete, imprecise, or wrong period)
4 = Answers the question with minor gaps or imprecision
5 = Directly, completely, and precisely answers the question as asked

Respond with valid JSON only:
{{"score": <1-5>, "reasoning": "<one sentence>"}}"""

# A|GT v2（参照 Phase03/judge_v2.py：先数值比对，再做定性判断）
_GT_PROMPT_V2 = """\
You are evaluating a financial RAG system. Your job is to score whether the model answer matches the ground truth.

QUESTION:
{question}

EXPECTED ANSWER (ground truth):
{expected}

MODEL ANSWER:
{answer}

## MANDATORY EVALUATION PROCESS — follow in order:

**Step 1 — Extract numbers**
- Extract the final numerical value(s) from the ground truth.
- Extract the final numerical value(s) from the model answer.
- Normalize to the same format before comparing: 0.8 = 80%; $2.018B = $2,018M; ratio 0.02 = 2%.

**Step 2 — Compute relative difference**
- relative_diff = |model_value - gt_value| / |gt_value|
- If gt_value is 0, use absolute difference.

**Step 3 — Apply scoring rules**

For NUMERICAL questions (any answer with a specific number):
- relative_diff <= 5%  AND reasoning correct -> score 5
- relative_diff <= 5%  but minor explanation issue -> score 4
- relative_diff 5-15% with correct methodology -> score 3
- relative_diff > 15% regardless of reasoning quality -> score 1 or 2
- A well-explained wrong number is STILL WRONG.
- Matching direction only ("yes it increased") without matching the specific numbers -> score 2 max.

For YES/NO or qualitative questions (no specific number in ground truth):
- Direction must match exactly (yes vs no = score 1)
- Key entities/facts must match (wrong segment, wrong company = score 1-2)
- Partial match on key facts -> score 3

**Step 4 — Final check**
- "I don't know" or refusal -> score 1 always
- Ignore formatting differences, focus on numerical accuracy and factual correctness

Score scale:
5 = Numbers match within 5% AND reasoning correct
4 = Numbers match within 5% but minor explanation issue
3 = Numbers off by 5-15% with correct method, OR qualitative partial match
2 = Numbers off >15% but direction/approach correct
1 = Wrong numbers (>15% off), wrong direction, hallucinated values, or refusal

Respond with valid JSON only:
{{"score": <1-5>, "reasoning": "<one sentence explaining the numerical comparison and why you chose this score>"}}"""


def _parse_judge_json(text: str) -> Dict[str, Any]:
    """解析 judge 的 JSON 输出，容忍 ```json 代码块包裹（参照 judge.py _call）。"""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


class RAGEvaluator:
    """
    多层 RAG 评测器:
      - Tier1: compute_retrieval_metrics（模块级函数，无需实例化）
      - Tier2: LLM-as-Judge 四轴（C|Q / A|C / A|Q / A|GT）
      - Tier3: Ragas（context_precision / faithfulness）
    """

    def __init__(self, eval_llm, eval_embeddings=None):
        """
        Args:
            eval_llm: LangChain Chat LLM（temperature=0 推荐）。
            eval_embeddings: LangChain Embeddings，仅 Tier3 (Ragas) 需要。
        """
        self.llm = eval_llm
        self.embeddings_wrapper = (
            LangchainEmbeddingsWrapper(eval_embeddings) if eval_embeddings is not None else None
        )
        if eval_embeddings is not None:
            self.llm_wrapper = LangchainLLMWrapper(eval_llm)
        logging.info("RAG Evaluator initialized (Tier1 + Tier2 + Tier3).")

    # ── Tier 2: LLM-as-Judge ──

    def _judge(self, prompt: str) -> Dict[str, Any]:
        """调用 judge LLM 并解析 JSON；失败时安全降级。"""
        try:
            response = self.llm.invoke(prompt)
            result = _parse_judge_json(response.content)
            result["score"] = int(result.get("score", 1))
            return result
        except Exception as e:
            logging.warning(f"Judge call failed: {e}")
            return {"score": 1, "reasoning": f"judge_error: {e}"}

    def judge_context_relevance(self, question: str, context: str) -> Dict[str, Any]:
        """C|Q — 检索上下文与问题相关吗？"""
        #你是一个用于评估金融文档（SEC 上市公司年报）检索增强生成（RAG）系统的裁判。问题{问题}  检索到的上下文： {检索内容} 任务：检索到的上下文中，是否包含回答该问题所需的信息？
        return self._judge(_CQ_PROMPT.format(question=question, context=context))

    def judge_answer_faithfulness(self, context: str, answer: str) -> Dict[str, Any]:
        """A|C — 答案是否忠实于上下文（无幻觉）？"""
        return self._judge(_AC_PROMPT.format(context=context, answer=answer))

    def judge_answer_relevance(self, question: str, answer: str) -> Dict[str, Any]:
        """A|Q — 答案是否正面回答了问题？"""
        return self._judge(_AQ_PROMPT.format(question=question, answer=answer))

    def judge_answer_correctness(self, question: str, expected: str, answer: str) -> Dict[str, Any]:
        """A|GT v2 — 答案与 ground truth 一致吗（数值提取 + 容差规则）？"""
        return self._judge(
            _GT_PROMPT_V2.format(question=question, expected=expected, answer=answer)
        )

    def evaluate_tier2(
        self,
        questions: List[str],
        ground_truths: List[str],
        contexts: List[List[str]],
        answers: List[str],
        context_join_sep: str = "\n\n---\n\n",
    ) -> Dict[str, Any]:
        """
        批量运行四轴 judge，返回逐样本明细（DataFrame）与均值汇总。

        Args:
            questions: 问题列表。
            ground_truths: 标准答案列表。
            contexts: 每个问题检回的上下文片段列表 List[List[str]]。
            answers: 模型生成的答案列表。
            context_join_sep: 多片段拼接分隔符。

        Returns:
            {"detail": DataFrame, "summary": {metric: mean_score}}
        """
        rows = []
        for q, gt, ctx_list, ans in zip(questions, ground_truths, contexts, answers):
            context = context_join_sep.join(ctx_list)
            cq = self.judge_context_relevance(q, context)
            ac = self.judge_answer_faithfulness(context, ans)
            aq = self.judge_answer_relevance(q, ans)
            agt = self.judge_answer_correctness(q, gt, ans)
            rows.append(
                {
                    "question": q,
                    "context_relevance_cq": cq["score"],
                    "faithfulness_ac": ac["score"],
                    "answer_relevance_aq": aq["score"],
                    "correctness_agt": agt["score"],
                    "cq_reasoning": cq.get("reasoning", ""),
                    "ac_reasoning": ac.get("reasoning", ""),
                    "aq_reasoning": aq.get("reasoning", ""),
                    "agt_reasoning": agt.get("reasoning", ""),
                }
            )

        df = pd.DataFrame(rows)
        summary = {
            "context_relevance_cq": round(df["context_relevance_cq"].mean(), 4),
            "faithfulness_ac": round(df["faithfulness_ac"].mean(), 4),
            "answer_relevance_aq": round(df["answer_relevance_aq"].mean(), 4),
            "correctness_agt": round(df["correctness_agt"].mean(), 4),
            "n": len(df),
        }
        logging.info(f"Tier2 Judge Summary: {summary}")
        return {"detail": df, "summary": summary}

    # ── Tier 3: Ragas（保留原有能力）──

    def build_dataset(
        self,
        questions: List[str],
        ground_truths: List[str],
        retrieved_contexts: List[List[str]],
        generated_answers: List[str],
    ) -> Dataset:
        """构建 Ragas 所需的 HuggingFace Dataset（contexts 需为 List[List[str]]）。"""
        data_dict = {
            "question": questions,
            "ground_truth": ground_truths,
            "contexts": retrieved_contexts,
            "answer": generated_answers,
        }
        return Dataset.from_dict(data_dict)

    def evaluate_pipeline(
        self,
        questions: List[str],
        ground_truths: List[str],
        retrieved_contexts: List[List[str]],
        generated_answers: List[str],
    ) -> Dict[str, Any]:
        """运行 Ragas 评测（context_precision + faithfulness）。"""
        if self.embeddings_wrapper is None:
            raise ValueError("Tier3 (Ragas) 需要 eval_embeddings，请在初始化时提供。")

        logging.info("Building evaluation dataset...")
        dataset = self.build_dataset(questions, ground_truths, retrieved_contexts, generated_answers)

        logging.info("Running Ragas evaluation (this may take a moment)...")

        # 调用实例的 evaluate 方法
        results = evaluate(
            dataset=dataset,
            metrics=[context_precision, faithfulness],
            llm=self.llm_wrapper,
            embeddings=self.embeddings_wrapper,
        )

        df = results.to_pandas()
        logging.info("\n" + df.to_string(index=False))
        return {
            "context_precision": df["context_precision"].mean(),
            "faithfulness": df["faithfulness"].mean(),
            "detailed_results": df,
        }


# ──────────────────────────────────────────────
#  Demo
# ──────────────────────────────────────────────
def run_evaluation_demo():
    """演示三层评测的调用方式（需要真实 LLM 才能运行 Tier2/Tier3）。"""
    print("⚠️  请先配置 eval_llm / eval_embeddings（如 ChatOllama、ChatOpenAI）。")
    print("""
示例:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from evaluation.rag_evaluator import RAGEvaluator, compute_retrieval_metrics

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    evaluator = RAGEvaluator(eval_llm=llm, eval_embeddings=embeddings)

    # Tier 1: 检索指标（无需 LLM）
    metrics = compute_retrieval_metrics(
        retrieved_sources=[["AMD_2022_10K.pdf", "AMD_2015_10K.pdf"]],
        ground_truth_docs=["AMD_2022_10K.pdf"],
        k=6,
    )

    # Tier 2: LLM-as-Judge 四轴
    tier2 = evaluator.evaluate_tier2(
        questions=["What was AMD's net income in 2022?"],
        ground_truths=["AMD's net income was $1.3 billion in 2022."],
        contexts=[["AMD reported a net income of $1.3 billion for fiscal 2022."]],
        answers=["AMD's net income in 2022 was $1.3 billion."],
    )
    print(tier2["summary"])

    # Tier 3: Ragas
    # results = evaluator.evaluate_pipeline(questions, ground_truths, contexts, answers)
""")


if __name__ == "__main__":
    run_evaluation_demo()
