"""
RAG Pipeline Observer - 可视化观察脚本
======================================
一步步展示：
  1. PDF 加载结果（原始文本块长什么样）
  2. 向量化结果（dense 向量 + sparse 向量长什么样）
  3. Qdrant 存储结果（向量数据库里存了什么）
  4. 搜索结果（三种模式的对比）

运行方式:
  cd D:\finance_rag\finance_rag_evaluation
  python observe_pipeline.py
"""
import sys
import os
import json
import numpy as np

# 修复 Windows 终端 GBK 编码问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# Part 1: 观察 PDF 加载结果
# ============================================================
def observe_loader():
    """观察 PDF 加载后得到的文本块"""
    print("\n" + "=" * 80)
    print("Part 1: PDF 加载结果观察")
    print("=" * 80)

    from src.core.loader import PDFLoader

    data_dir = r"D:\finance_rag\finance_rag_evaluation\data"
    loader = PDFLoader(data_dir=data_dir, chunk_size=512, chunk_overlap=50)
    chunks = loader.load_all_pdfs()

    print(f"\n总共有 {len(chunks)} 个文本块\n")

    # 看看每个 PDF 生成了多少块
    file_counts = {}
    for chunk in chunks:
        fn = chunk["metadata"].get("original_filename", "unknown")
        file_counts[fn] = file_counts.get(fn, 0) + 1

    print("--- 各 PDF 的分块统计 ---")
    for fn, count in sorted(file_counts.items()):
        print(f"  {fn}: {count} 块")

    # 展示 3 个完整的文本块示例
    print("\n--- 文本块示例（展示前 3 个完整块）---")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n[块 {i+1}]")
        print(f"  metadata: {json.dumps(chunk['metadata'], ensure_ascii=False)}")
        print(f"  page: {chunk['page']}")
        print(f"  text 长度: {len(chunk['text'])} 字符")
        print(f"  text 完整内容:")
        print(f"  {'-' * 60}")
        # 打印完整文本，每行加缩进
        for line in chunk['text'][:500].split('\n'):
            print(f"  {line}")
        if len(chunk['text']) > 500:
            print(f"  ... (还有 {len(chunk['text']) - 500} 字符)")
        print(f"  {'-' * 60}")

    return chunks


# ============================================================
# Part 2: 观察向量化结果
# ============================================================
def observe_embeddings(chunks):
    """观察 dense 和 sparse 向量长什么样"""
    print("\n\n" + "=" * 80)
    print("Part 2: 向量化结果观察")
    print("=" * 80)

    from src.core.vectorstore import HybridVectorStore

    store = HybridVectorStore(
        qdrant_path=r"D:\finance_rag\finance_rag_evaluation\qdrant_data"
    )

    # 取第一个文本块做演示
    sample_text = chunks[0]["text"]
    print(f"\n--- 取第一个文本块做向量化演示 ---")
    print(f"  原始文本 (前 100 字符): {sample_text[:100]}...")

    # Dense 向量 (sentence-transformers)
    print(f"\n--- Dense 向量 (sentence-transformers all-MiniLM-L6-v2) ---")
    dense_vec = store.embeddings.embed_query(sample_text)
    dense_arr = np.array(dense_vec)
    print(f"  维度: {len(dense_vec)}")
    print(f"  类型: {type(dense_vec)}")
    print(f"  前 10 个值: {[f'{v:.6f}' for v in dense_vec[:10]]}")
    print(f"  最小值: {dense_arr.min():.6f}")
    print(f"  最大值: {dense_arr.max():.6f}")
    print(f"  平均值: {dense_arr.mean():.6f}")
    print(f"  L2 范数: {np.linalg.norm(dense_arr):.6f}")

    # Sparse 向量 (BM25)
    print(f"\n--- Sparse 向量 (FastEmbed BM25) ---")
    sparse_vec = store.sparse_embeddings.embed_query(sample_text)
    print(f"  类型: {type(sparse_vec)}")
    print(f"  非零元素个数: {len(sparse_vec.indices)}")
    print(f"  indices (词ID): {sparse_vec.indices}")
    print(f"  values  (权重): {sparse_vec.values}")
    print(f"  解释: BM25 将文本分词后，每个词映射到一个 hash ID")
    print(f"        只保留有意义的词，所以向量非常稀疏（大部分维度是 0）")

    # 对比两个不同文本块的 dense 向量相似度
    print(f"\n--- 两个文本块的 Dense 向量相似度对比 ---")
    text1 = chunks[0]["text"]
    text2 = chunks[100]["text"] if len(chunks) > 100 else chunks[1]["text"]

    vec1 = np.array(store.embeddings.embed_query(text1))
    vec2 = np.array(store.embeddings.embed_query(text2))
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    fn1 = chunks[0]["metadata"].get("original_filename", "?")
    fn2 = (chunks[100] if len(chunks) > 100 else chunks[1])["metadata"].get("original_filename", "?")
    print(f"  文本块1: {fn1} (前 60 字: {text1[:60]}...)")
    print(f"  文本块2: {fn2} (前 60 字: {text2[:60]}...)")
    print(f"  余弦相似度: {similarity:.4f}")
    print(f"  (越接近 1 表示语义越相似)")


# ============================================================
# Part 3: 观察 Qdrant 存储结果
# ============================================================
def observe_qdrant():
    """观察 Qdrant 里存了什么"""
    print("\n\n" + "=" * 80)
    print("Part 3: Qdrant 向量数据库存储观察")
    print("=" * 80)

    from qdrant_client import QdrantClient

    client = QdrantClient(path=r"D:\finance_rag\finance_rag_evaluation\qdrant_data")
    collection_name = "finance_rag_hybrid"

    # 集合信息
    info = client.get_collection(collection_name)
    print(f"\n--- 集合信息 ---")
    print(f"  集合名: {collection_name}")
    print(f"  状态: {info.status}")
    print(f"  总点数: {info.points_count}")
    print(f"  Dense 向量配置: {info.config.params.vectors}")
    print(f"  Sparse 向量配置: {info.config.params.sparse_vectors}")

    # 取 2 个点看具体存储
    points, _ = client.scroll(
        collection_name=collection_name,
        limit=2,
        with_payload=True,
        with_vectors=True,
    )

    print(f"\n--- 存储的点示例（前 2 个）---")
    for i, p in enumerate(points):
        print(f"\n[点 {i+1}]")
        print(f"  Point ID: {p.id}")
        print(f"  Payload keys: {list(p.payload.keys())}")
        print(f"    page_content (前 80 字): {p.payload.get('page_content', '')[:80]}...")
        print(f"    metadata: {json.dumps(p.payload.get('metadata', {}), ensure_ascii=False)}")

        # 向量
        vec = p.vector
        if isinstance(vec, list):
            # 只有 dense 向量
            print(f"  Dense 向量: 维度={len(vec)}, 前 5 个值={[f'{v:.4f}' for v in vec[:5]]}")
        elif isinstance(vec, dict):
            # 既有 dense 也有 sparse
            for k, v in vec.items():
                if hasattr(v, 'indices'):
                    print(f"  Sparse 向量 ({k}): 非零元素={len(v.indices)}, indices[:5]={v.indices[:5]}")
                elif isinstance(v, list):
                    print(f"  Dense 向量 ({k}): 维度={len(v)}, 前 5 个值={[f'{x:.4f}' for x in v[:5]]}")

    client.close()


# ============================================================
# Part 4: 观察搜索结果
# ============================================================
def observe_search():
    """观察三种搜索模式的结果对比"""
    print("\n\n" + "=" * 80)
    print("Part 4: 搜索结果观察（3 种模式对比）")
    print("=" * 80)

    from src.core.vectorstore import HybridVectorStore

    store = HybridVectorStore(
        qdrant_path=r"D:\finance_rag\finance_rag_evaluation\qdrant_data"
    )
    loaded = store.load_index()
    if not loaded:
        print("索引不存在，请先运行 build_index")
        return

    queries = [
        ("Amazon revenue 2019", "应该匹配 AMAZON_2019_10K.pdf"),
        ("AMD net income 2022", "应该匹配 AMD_2022_10K.pdf"),
        ("quarterly earnings AMCOR", "应该匹配 AMCOR_2023Q4_EARNINGS.pdf"),
    ]

    for query, expected in queries:
        print(f"\n{'=' * 70}")
        print(f"查询: \"{query}\"")
        print(f"期望: {expected}")
        print(f"{'=' * 70}")

        for mode in ["dense", "sparse", "hybrid"]:
            results = store.search(query, top_k=3, mode=mode)
            mode_label = {"dense": "Dense (语义)", "sparse": "Sparse (关键词)", "hybrid": "Hybrid (混合)"}[mode]
            print(f"\n  --- {mode_label} ---")

            if not results:
                print("    (无结果)")
                continue

            for j, r in enumerate(results, 1):
                fn = r["metadata"].get("original_filename", "N/A")
                page = r["page"]
                score = r["score"]
                text_preview = r["text"][:120].replace("\n", " ")
                print(f"    [{j}] Score={score:.4f} | {fn} | Page {page}")
                print(f"        {text_preview}...")


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    print("=" * 80)
    print("RAG Pipeline Observer - 可视化观察脚本")
    print("一步步展示: PDF加载 → 向量化 → Qdrant存储 → 搜索结果")
    print("=" * 80)

    # Part 1: PDF 加载
    chunks = observe_loader()

    # Part 2: 向量化
    observe_embeddings(chunks)

    # Part 3: Qdrant 存储
    observe_qdrant()

    # Part 4: 搜索结果
    observe_search()

    print("\n\n" + "=" * 80)
    print("观察完毕！以上就是 RAG 系统的完整数据流。")
    print("=" * 80)
