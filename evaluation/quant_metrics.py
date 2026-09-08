"""
Quantitative Metrics Benchmark — 金融 RAG 全链路量化指标
=========================================================
两段式量化评测，口径对齐简历/面试表述：

【Stage 1 — 召回阶段（Retrieval）】召回率衡量"检索阶段"
    对同一批问题，分别用以下 4 种配置检索同一套索引：
        - Dense            : all-MiniLM-L6-v2 稠密向量（基线）
        - Sparse           : Qdrant/bm25 稀疏向量（BM25）
        - Hybrid           : Dense + Sparse，Qdrant 原生 RRF 融合
        - Hybrid + Rerank  : Hybrid 宽进 k=12 → CrossEncoder 重排严出 top=5
    指标（文档级，基于 ground-truth PDF 文件名）：
        Recall@k    : GT 文档被检回的比例（多 GT 文档题为命中率均值）
        Hit Rate@k  : 至少检回 1 个 GT 文档的题目比例
        Precision@k : 检回 chunk 中来自 GT 文档的比例
        MRR@k       : 首个 GT 文档 chunk 的倒数排名
    关键产出：Dense → Hybrid 的召回提升（绝对 pp + 相对 %）、
              重排对 top-5 召回的影响。

【Stage 2 — 生成阶段（Generation）】准确率衡量"大模型生成质量"
    在 **重排后 top-5 上下文 → LLM 生成答案** 之后，用 LLM-as-Judge 打分：
        - Answer Accuracy  : A|GT >=4（宽松）/ ==5（严格）  ← 准确率主指标
        - Faithfulness     : A|C  >=4 比例（忠实度/无幻觉）
        - Answer Relevance : A|Q  >=4 比例（答其所问）
        - Context Support  : C|Q  >=4 比例（上下文是否支撑）
    另含：1-5 分均值、按题目难度(L1-L5)分组准确率、各阶段平均延迟。
    注意：准确率只在 rerank + generate 之后计算（重排序是生成的前置环节）。

运行方式:
    cd D:\\finance_rag\\finance_rag_evaluation
    python -m evaluation.quant_metrics --retrieval-only              # 免 LLM，纯检索对比（推荐先跑）
    python -m evaluation.quant_metrics --retrieval-only --production-preprocess
    python -m evaluation.quant_metrics --limit 20                    # 全流程 20 题（需 LLM API Key）
    python -m evaluation.quant_metrics                               # 全量 100 题
结果输出到 evaluation/results/ 下的 JSON + CSV。
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# 修复 Windows 终端 GBK 编码问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 项目根目录（finance_rag_evaluation/）加入 sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd

from src.core.vectorstore import HybridVectorStore
from src.core.reranker import get_reranker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("quant_metrics")

QDRANT_PATH = os.path.join(_PROJECT_ROOT, "qdrant_data")
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
TEST_SET_PATH = os.path.join(DATA_DIR, "test_set.json")
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "evaluation", "results")

K_WIDE = 12    # 宽进：混合检索候选数
K_STRICT = 5   # 严出：重排后进入生成的上下文数


# ══════════════════════════════════════════════
#  Ground-truth 文档映射：test_set 的 "source" 字段 → PDF 文件名
# ══════════════════════════════════════════════
_COMPANY_NORM = {
    "3m": "3M",
    "aes": "AES",
    "activision blizzard": "ACTIVISIONBLIZZARD",
    "amazon": "AMAZON",
    "amcor": "AMCOR",
    "amd": "AMD",
    "american express": "AMERICANEXPRESS",
    "american water works": "AMERICANWATERWORKS",
}
_TYPE_NORM = {"10-k": "10K", "10-q": "10Q", "8-k": "8K", "earnings": "EARNINGS"}

# 形如 "3M 2022 10-K" / "3M 2023 Q2 10-Q" / "Activision Blizzard 2019 10-K"
_DOC_MENTION_RE = re.compile(
    r"(3M|AES|Activision Blizzard|Amazon|Amcor|AMD|American Express|American Water Works)"
    r"\s+(20\d{2})(?:\s+(Q[1-4]))?\s+(10-K|10-Q|8-K|EARNINGS)",
    re.IGNORECASE,
)


def _norm(s: str) -> str:
    """归一化：去非字母数字并大写，使 '3M 2022 10-K' 与 '3M_2022_10K.pdf' 可比对。"""
    return re.sub(r"[^A-Za-z0-9]", "", s).upper()


def list_pdf_files() -> List[str]:
    """列出 data 目录下全部 PDF 文件名。"""
    return sorted(f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf"))


def extract_gt_docs(source_field: str, pdf_files: List[str]) -> List[str]:
    """
    把 test_set 的 source 描述（如 '3M 2022 10-K (p.48) + 3M 2018 10-K (p.14)'）
    解析为 GT PDF 文件名列表（支持跨文档多 GT）。
    'N/A（合规拒答类 / 知识库中无相关信息）' → [] （不参与召回指标）。
    """
    if not source_field or source_field.strip().upper().startswith("N/A"):
        return []

    tokens = set()
    for m in _DOC_MENTION_RE.finditer(source_field):
        company = _COMPANY_NORM[m.group(1).lower()]
        year = m.group(2)
        quarter = (m.group(3) or "").upper()
        dtype = _TYPE_NORM[m.group(4).lower()]
        tokens.add(_norm(f"{company}{year}{quarter}{dtype}"))

    gt = []
    for pdf in pdf_files:
        pdf_norm = _norm(pdf)
        if pdf_norm in tokens or any(pdf_norm.startswith(t) for t in tokens):
            gt.append(pdf)
    return sorted(set(gt))


# ══════════════════════════════════════════════
#  召回阶段指标（文档级，多 GT 感知）
# ══════════════════════════════════════════════
def _doc_of(chunk: Dict[str, Any]) -> str:
    return chunk.get("metadata", {}).get("original_filename", "Unknown")


def retrieval_metrics_at_cutoffs(
    retrieved_files: List[str], gt_docs: List[str], cutoffs: Tuple[int, ...]
) -> Dict[str, float]:
    """单题指标：在各 k 截断下计算 recall / hit / precision / MRR。"""
    gt = set(gt_docs)
    out: Dict[str, float] = {}
    for k in cutoffs:
        topk = retrieved_files[:k]
        hits = sum(1 for g in gt if g in topk)
        out[f"recall@{k}"] = hits / len(gt) if gt else 0.0
        out[f"hit@{k}"] = 1.0 if hits > 0 else 0.0
        out[f"precision@{k}"] = (
            sum(1 for f in topk if f in gt) / len(topk) if topk else 0.0
        )
        rr = 0.0
        for rank, f in enumerate(topk, 1):
            if f in gt:
                rr = 1.0 / rank
                break
        out[f"mrr@{k}"] = rr
    return out


def aggregate_metrics(per_question: List[Dict[str, float]], cutoffs: Tuple[int, ...]) -> Dict[str, float]:
    """对单题指标取均值。"""
    n = len(per_question)
    agg: Dict[str, float] = {"n": n}
    if n == 0:
        return agg
    keys = []
    for k in cutoffs:
        keys += [f"recall@{k}", f"hit@{k}", f"precision@{k}", f"mrr@{k}"]
    for key in keys:
        agg[key] = round(sum(d[key] for d in per_question) / n, 4)
    return agg


# ══════════════════════════════════════════════
#  Stage 1 — 召回阶段 Benchmark
# ══════════════════════════════════════════════
def _retrieve_one(
    question: str,
    mode: str,
    store: HybridVectorStore,
    reranker,
    pipeline: Optional[Any],
    use_preprocess: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    返回 (宽进 docs[k=12], 重排后 docs[k=5])。
    use_preprocess=False: 原始问题 + 无 metadata 过滤（纯检索器对比，零 LLM）。
    use_preprocess=True : 走 pipeline 的 adaptive 预处理 + 2-step relax（生产链路）。
    """
    if use_preprocess:
        assert pipeline is not None
        pre = pipeline.preprocessor.preprocess_adaptive(question)
        search_query = pre["search_query"]
        if pre["use_filter"]:
            docs, _ = pipeline.retrieve_with_relax(
                search_query, pre["filter_token"], pre["company_filter"], mode=mode
            )
        else:
            docs = store.search(search_query, top_k=K_WIDE, mode=mode, query_filter=None)
    else:
        search_query = question
        docs = store.search(question, top_k=K_WIDE, mode=mode, query_filter=None)

    reranked = reranker.rerank(search_query, docs, top_k=K_STRICT) if docs else []
    return docs, reranked


def run_retrieval_stage(
    samples: List[Dict[str, Any]],
    pdf_files: List[str],
    use_preprocess: bool = False,
    pipeline: Optional[Any] = None,
    store: Optional[HybridVectorStore] = None,
) -> Dict[str, Any]:
    """对有 GT 文档的题目跑 dense/sparse/hybrid/rerank 四配置，返回汇总与逐题明细。

    store 可传入已有实例（Qdrant 本地模式同目录不允许两个客户端同时打开，
    故 Stage 2 的 pipeline 必须复用 Stage 1 的 store）。
    """
    logger.info("=" * 70)
    logger.info("Stage 1 — 召回阶段 Benchmark（%s）", "生产预处理+过滤" if use_preprocess else "原始问题+无过滤")
    logger.info("=" * 70)

    if store is None:
        store = HybridVectorStore(qdrant_path=QDRANT_PATH)
        if not store.load_index():
            raise RuntimeError("Qdrant 索引不存在，请先运行 main_pipeline.py 构建索引。")
    reranker = get_reranker()

    eval_samples = [s for s in samples if s["gt_docs"]]
    logger.info("参与召回评测的题目数: %d（%d 道 N/A 题不计入召回）",
                len(eval_samples), len(samples) - len(eval_samples))

    # 每题 × 每配置的文件名列表
    per_q_rows: List[Dict[str, Any]] = []
    cfg_metrics: Dict[str, List[Dict[str, float]]] = {
        "dense": [], "sparse": [], "hybrid": [], "hybrid_rerank": []
    }

    for i, s in enumerate(eval_samples, 1):
        q, gt = s["question"], s["gt_docs"]
        row: Dict[str, Any] = {"id": s["id"], "question": q, "gt_docs": ";".join(gt)}

        dense_docs, _ = _retrieve_one(q, "dense", store, reranker, pipeline, use_preprocess)
        sparse_docs, _ = _retrieve_one(q, "sparse", store, reranker, pipeline, use_preprocess)
        hybrid_docs, reranked_docs = _retrieve_one(q, "hybrid", store, reranker, pipeline, use_preprocess)

        files = {
            "dense": [_doc_of(d) for d in dense_docs],
            "sparse": [_doc_of(d) for d in sparse_docs],
            "hybrid": [_doc_of(d) for d in hybrid_docs],
            "hybrid_rerank": [_doc_of(d) for d in reranked_docs],
        }
        for cfg, fl in files.items():
            m = retrieval_metrics_at_cutoffs(fl, gt, cutoffs=(K_STRICT, K_WIDE))
            cfg_metrics[cfg].append(m)
            row[f"{cfg}_recall@{K_STRICT}"] = m[f"recall@{K_STRICT}"]
            row[f"{cfg}_recall@{K_WIDE}"] = m[f"recall@{K_WIDE}"]
            row[f"{cfg}_mrr@{K_STRICT}"] = m[f"mrr@{K_STRICT}"]
        per_q_rows.append(row)

        if i % 10 == 0 or i == len(eval_samples):
            logger.info("  召回评测进度: %d/%d", i, len(eval_samples))

    # 汇总
    summary: Dict[str, Any] = {
        "n_questions": len(eval_samples),
        "mode": "production_preprocess" if use_preprocess else "raw_query_no_filter",
        "k_wide": K_WIDE,
        "k_strict": K_STRICT,
        "configs": {},
    }
    for cfg, per_q in cfg_metrics.items():
        summary["configs"][cfg] = aggregate_metrics(per_q, cutoffs=(K_STRICT, K_WIDE))

    # 关键差值：Dense → Hybrid 召回提升
    def _cfg(name, key):
        return summary["configs"][name][key]

    def _delta(a: float, b: float) -> Dict[str, float]:
        return {
            "abs_pp": round((a - b) * 100, 2),           # 绝对提升（百分点）
            "rel_pct": round((a - b) / b * 100, 1) if b > 0 else None,  # 相对提升（%）
        }

    summary["deltas"] = {
        "hybrid_vs_dense@12_recall": _delta(_cfg("hybrid", f"recall@{K_WIDE}"),
                                            _cfg("dense", f"recall@{K_WIDE}")),
        "hybrid_vs_dense@5_recall": _delta(_cfg("hybrid", f"recall@{K_STRICT}"),
                                           _cfg("dense", f"recall@{K_STRICT}")),
        "hybrid_vs_dense@12_mrr": _delta(_cfg("hybrid", f"mrr@{K_WIDE}"),
                                         _cfg("dense", f"mrr@{K_WIDE}")),
        "rerank_vs_hybrid@5_recall": _delta(_cfg("hybrid_rerank", f"recall@{K_STRICT}"),
                                            _cfg("hybrid", f"recall@{K_STRICT}")),
        "rerank_vs_hybrid@5_mrr": _delta(_cfg("hybrid_rerank", f"mrr@{K_STRICT}"),
                                         _cfg("hybrid", f"mrr@{K_STRICT}")),
        "rerank_vs_dense@5_recall": _delta(_cfg("hybrid_rerank", f"recall@{K_STRICT}"),
                                           _cfg("dense", f"recall@{K_STRICT}")),
    }
    return {"summary": summary, "detail": pd.DataFrame(per_q_rows), "store": store}


def print_retrieval_report(summary: Dict[str, Any]) -> None:
    """控制台打印召回阶段对比表。"""
    k5, k12 = K_STRICT, K_WIDE
    print("\n" + "=" * 78)
    print("Stage 1 — 召回阶段量化指标（Recall / Precision / MRR，文档级）")
    print(f"题目数 n={summary['n_questions']}，宽进 k={k12}，严出 top={k5}，"
          f"模式：{summary['mode']}")
    print("=" * 78)
    header = f"{'检索配置':<22}{'Recall@5':>10}{'Recall@12':>11}{'Prec@5':>9}{'MRR@5':>9}{'MRR@12':>9}"
    print(header)
    print("-" * 78)
    labels = [
        ("dense", "Dense (稠密基线)"),
        ("sparse", "Sparse (BM25)"),
        ("hybrid", "Hybrid (RRF融合)"),
        ("hybrid_rerank", "Hybrid+Rerank(重排)"),
    ]
    for cfg, label in labels:
        c = summary["configs"][cfg]
        print(f"{label:<22}{c[f'recall@{k5}']:>10.4f}{c[f'recall@{k12}']:>11.4f}"
              f"{c[f'precision@{k5}']:>9.4f}{c[f'mrr@{k5}']:>9.4f}{c[f'mrr@{k12}']:>9.4f}")

    d = summary["deltas"]
    print("-" * 78)
    print("关键提升（简历口径）：")
    hv = d["hybrid_vs_dense@12_recall"]
    hv5 = d["hybrid_vs_dense@5_recall"]
    print(f"  ① Dense → Hybrid 召回提升  Recall@12: {hv['abs_pp']:+.2f} pp"
          f"（相对 {hv['rel_pct']:+.1f}%） | Recall@5: {hv5['abs_pp']:+.2f} pp")
    rr = d["rerank_vs_hybrid@5_recall"]
    rrm = d["rerank_vs_hybrid@5_mrr"]
    print(f" ② 重排对 top-5 的影响      Recall@5: {rr['abs_pp']:+.2f} pp"
          f" | MRR@5: {rrm['abs_pp']:+.2f} pp（重排提升的是排名质量而非候选池）")
    rd = d["rerank_vs_dense@5_recall"]
    print(f"  ③ 全链路 vs 稠密基线       Recall@5(混合+重排) 较 Dense@5: {rd['abs_pp']:+.2f} pp"
          f"（相对 {rd['rel_pct']:+.1f}%）")
    print("=" * 78)


# ══════════════════════════════════════════════
#  Stage 2 — 生成阶段 Benchmark（重排 + 生成之后算准确率）
# ══════════════════════════════════════════════
def run_generation_stage(
    samples: List[Dict[str, Any]],
    pipeline: Any,
    evaluator: Any,
) -> Dict[str, Any]:
    """
    全链路：预处理 → Hybrid 检索(宽进 k=12, relax) → CrossEncoder 重排(严出 top=5)
            → LLM 生成 → LLM-as-Judge 四轴打分。
    准确率（Answer Accuracy）在重排+生成之后计算。
    """
    logger.info("=" * 70)
    logger.info("Stage 2 — 生成阶段 Benchmark（重排 → 生成 → Judge 准确率）")
    logger.info("=" * 70)

    records: List[Dict[str, Any]] = []
    n_errors = 0
    for i, s in enumerate(samples, 1):
        q = s["question"]
        t0 = time.perf_counter()
        try:
            r = pipeline.answer(q, retrieval_mode="hybrid")
            answer, contexts, sources = r["answer"], r["contexts"], r["sources"]
            timings, filter_token = r["timings"], r["filter_token"]
        except Exception as e:
            # 单题失败（如 LLM 配额/网络）不中断整体评测：记为错误答案，
            # Judge 会自然打 1 分，准确率如实反映系统可用性。
            n_errors += 1
            logger.error("Q%d 生成失败: %s", s["id"], e)
            answer, contexts, sources = f"[PIPELINE ERROR] {e}", [], []
            timings, filter_token = {}, "ERROR"
        wall_ms = (time.perf_counter() - t0) * 1000
        records.append({
            "id": s["id"],
            "level": s.get("level", ""),
            "category": s.get("category", ""),
            "question": q,
            "ground_truth": s["ground_truth"],
            "answer": answer,
            "contexts": contexts,
            "sources": sources,
            "gt_docs": ";".join(s["gt_docs"]),
            "filter_token": filter_token,
            "preprocess_ms": round(timings.get("preprocess_ms", 0), 1),
            "retrieve_ms": round(timings.get("retrieve_ms", 0), 1),
            "rerank_ms": round(timings.get("rerank_ms", 0), 1),
            "generate_ms": round(timings.get("generate_ms", 0), 1),
            "wall_ms": round(wall_ms, 1),
        })
        if i % 5 == 0 or i == len(samples):
            logger.info("  全链路进度: %d/%d", i, len(samples))
    if n_errors:
        logger.warning("Stage 2 共 %d/%d 题生成失败（已记为错误答案参与统计）", n_errors, len(samples))

    # ── LLM-as-Judge 四轴（在重排后的上下文 + 生成答案上评测）──
    logger.info("运行 LLM-as-Judge（C|Q / A|C / A|Q / A|GT）...")
    tier2 = evaluator.evaluate_tier2(
        questions=[r["question"] for r in records],
        ground_truths=[r["ground_truth"] for r in records],
        contexts=[r["contexts"] for r in records],
        answers=[r["answer"] for r in records],
    )
    detail = tier2["detail"].reset_index(drop=True)
    for col in ["id", "level", "category", "gt_docs", "sources",
                "preprocess_ms", "retrieve_ms", "rerank_ms", "generate_ms", "wall_ms"]:
        detail[col] = [r[col] for r in records]

    # ── 准确率汇总 ──
    n = len(detail)
    summary: Dict[str, Any] = {
        "n_questions": n,
        "accuracy": {
            # 准确率主指标：A|GT 对照标准答案（数值容差 5%/15% 规则在 judge prompt 内）
            "answer_accuracy_strict (A|GT==5)": round((detail["correctness_agt"] == 5).mean(), 4),
            "answer_accuracy_lenient (A|GT>=4)": round((detail["correctness_agt"] >= 4).mean(), 4),
            "faithfulness_pass (A|C>=4)": round((detail["faithfulness_ac"] >= 4).mean(), 4),
            "answer_relevance_pass (A|Q>=4)": round((detail["answer_relevance_aq"] >= 4).mean(), 4),
            "context_support_pass (C|Q>=4)": round((detail["context_relevance_cq"] >= 4).mean(), 4),
        },
        "mean_judge_scores_1to5": {
            "context_relevance_cq": round(detail["context_relevance_cq"].mean(), 3),
            "faithfulness_ac": round(detail["faithfulness_ac"].mean(), 3),
            "answer_relevance_aq": round(detail["answer_relevance_aq"].mean(), 3),
            "correctness_agt": round(detail["correctness_agt"].mean(), 3),
        },
        "latency_ms": {
            "preprocess": round(detail["preprocess_ms"].mean(), 1),
            "retrieve": round(detail["retrieve_ms"].mean(), 1),
            "rerank": round(detail["rerank_ms"].mean(), 1),
            "generate": round(detail["generate_ms"].mean(), 1),
            "end_to_end": round(detail["wall_ms"].mean(), 1),
        },
    }

    # 按难度 L1-L5 分组准确率
    by_level = {}
    for level, grp in detail.groupby("level"):
        by_level[level] = {
            "n": len(grp),
            "accuracy_lenient": round((grp["correctness_agt"] >= 4).mean(), 4),
            "accuracy_strict": round((grp["correctness_agt"] == 5).mean(), 4),
            "faithfulness_pass": round((grp["faithfulness_ac"] >= 4).mean(), 4),
            "mean_correctness": round(grp["correctness_agt"].mean(), 3),
        }
    summary["by_level"] = by_level

    # 召回阶段（重排后实际进入生成的 top-5）文档级召回
    gt_present = [r for r in records if r["gt_docs"]]
    if gt_present:
        hit = 0
        for r in gt_present:
            gt_set = set(r["gt_docs"].split(";"))
            if gt_set & set(r["sources"]):
                hit += 1
        summary["final_recall_after_rerank"] = round(hit / len(gt_present), 4)

    return {"summary": summary, "detail": detail}


def print_generation_report(summary: Dict[str, Any]) -> None:
    """控制台打印生成阶段准确率表。"""
    print("\n" + "=" * 78)
    print("Stage 2 — 生成阶段量化指标（重排 top-5 上下文 → LLM 生成 → Judge）")
    print(f"题目数 n={summary['n_questions']}")
    print("=" * 78)
    print("【准确率 / 通过率】（评分 1-5，>=4 视为通过）")
    for k, v in summary["accuracy"].items():
        print(f"  {k:<42}{v*100:>6.2f}%")
    print("\n【Judge 平均分（1-5）】")
    for k, v in summary["mean_judge_scores_1to5"].items():
        print(f"  {k:<26}{v:>6.3f}")
    print("\n【按难度分组 — Answer Accuracy (A|GT>=4)】")
    for level in sorted(summary["by_level"]):
        s = summary["by_level"][level]
        print(f"  {level}: n={s['n']:<3} 准确率={s['accuracy_lenient']*100:>6.2f}%"
              f"  严格(=5)={s['accuracy_strict']*100:>6.2f}%"
              f"  忠实度={s['faithfulness_pass']*100:>6.2f}%")
    if "final_recall_after_rerank" in summary:
        print(f"\n【重排后进入上下文的 GT 文档召回率】{summary['final_recall_after_rerank']*100:.2f}%")
    print("\n【平均延迟 ms】")
    for k, v in summary["latency_ms"].items():
        print(f"  {k:<12}{v:>8.1f} ms")
    print("=" * 78)


# ══════════════════════════════════════════════
#  主入口
# ══════════════════════════════════════════════
def load_test_set(pdf_files: List[str], limit: Optional[int] = None) -> List[Dict[str, Any]]:
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    samples = []
    for item in raw:
        gt_docs = extract_gt_docs(item.get("source", ""), pdf_files)
        samples.append({
            "id": item["id"],
            "level": item.get("level", ""),
            "category": item.get("category", ""),
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "source": item.get("source", ""),
            "gt_docs": gt_docs,
        })
    if limit:
        samples = samples[:limit]
    return samples


def save_results(name: str, summary: Dict[str, Any], detail: pd.DataFrame) -> Tuple[str, str]:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, f"{name}_summary.json")
    csv_path = os.path.join(RESULTS_DIR, f"{name}_detail.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    detail.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info("结果已保存: %s | %s", json_path, csv_path)
    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser(description="Finance RAG 量化指标 Benchmark")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="只跑 Stage 1 召回阶段（默认原始问题无过滤，零 LLM 成本）")
    parser.add_argument("--production-preprocess", action="store_true",
                        help="Stage 1 使用生产链路预处理（adaptive 路由 + metadata 过滤 + relax），需 LLM")
    parser.add_argument("--limit", type=int, default=None, help="只取前 N 道题（调试用）")
    args = parser.parse_args()

    pdf_files = list_pdf_files()
    logger.info("data 目录 PDF 数: %d", len(pdf_files))
    samples = load_test_set(pdf_files, limit=args.limit)
    logger.info("测试集题目数: %d", len(samples))
    n_gt = sum(1 for s in samples if s["gt_docs"])
    logger.info("其中含 GT 文档（参与召回评测）: %d；N/A 题（合规/无答案）: %d",
                n_gt, len(samples) - n_gt)

    # ── Stage 1：召回阶段 ──
    pipeline = None
    if args.production_preprocess:
        from main_pipeline import FinanceRAGPipeline, build_llm
        pipeline = FinanceRAGPipeline(build_llm(), use_rewrite=False)

    retrieval_out = run_retrieval_stage(
        samples, pdf_files,
        use_preprocess=args.production_preprocess,
        pipeline=pipeline,
    )
    store = retrieval_out["store"]
    print_retrieval_report(retrieval_out["summary"])
    save_results("stage1_retrieval", retrieval_out["summary"], retrieval_out["detail"])

    if args.retrieval_only:
        print("\n✅ 召回阶段完成（--retrieval-only，未跑生成阶段）。")
        return

    # ── Stage 2：生成阶段（重排 + 生成之后算准确率）──
    from main_pipeline import FinanceRAGPipeline, build_llm
    from evaluation.rag_evaluator import RAGEvaluator

    llm = pipeline.llm if pipeline is not None else build_llm()
    if pipeline is None:
        # 复用 Stage 1 已加载的 store，避免 Qdrant 本地模式同目录双客户端锁冲突
        pipeline = FinanceRAGPipeline(llm, store=store, use_rewrite=False)
    evaluator = RAGEvaluator(eval_llm=llm)

    gen_out = run_generation_stage(samples, pipeline, evaluator)
    print_generation_report(gen_out["summary"])
    save_results("stage2_generation", gen_out["summary"], gen_out["detail"])

    print("\n✅ 全链路量化评测完成。JSON/CSV 见 evaluation/results/ 目录。")


if __name__ == "__main__":
    main()
