"""
Query Preprocessor Module (对齐 financebench-rag-eval 的 query_analysis 链)
==========================================================================
两个核心能力：

1. Query Filter 结构化提取（参照 chains/query_analysis.py）:
   用 Pydantic 结构化输出抽取 filter_token（公司+年）与 company_filter（仅公司），
   供检索阶段做 metadata 过滤，并支持 2-step relax（公司+年 → 仅公司 → NONE）。

2. Query 变换（保留原有能力）:
   Query Rewriting / HyDE，用于提升检索召回。
"""

import logging
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseLanguageModel
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ──────────────────────────────────────────────
#  Query Filter 结构化输出模型
#  （对齐 financebench-rag-eval QueryFilter）
# ──────────────────────────────────────────────
class QueryFilter(BaseModel):
    """从金融问题中抽取的结构化过滤条件。"""

    filter_token: str = Field(
        description="Company + year token, e.g. AMD_2022 or NONE"
    )
    company_filter: str = Field(
        description="Company name only, e.g. AMD or NONE"
    )


QUERY_FILTER_SYSTEM = (
    "Extract the company name and fiscal year from the financial question. "
    "Return a JSON object with two fields:\n"
    "- 'filter_token': company name + year combined (e.g. AMD_2022, AMAZON_2019). "
    "  If year is unknown, return just the company name (e.g. AMD). "
    "  If no company is mentioned, return NONE.\n"
    "- 'company_filter': company name only, no year (e.g. AMD, 3M, AMAZON). "
    "  If no company is mentioned, return NONE.\n"
    "Use SEC filing filename conventions as they appear in document names: "
    "3M, AES, AMAZON, AMCOR, AMD, AMERICANEXPRESS, AMERICANWATERWORKS, "
    "ACTIVISIONBLIZZARD, JOHNSON_JOHNSON, KRAFTHEINZ, MGMRESORTS, etc.\n"
    "Return ONLY the JSON object, nothing else."
)


class QueryType(BaseModel):
    """用于 Adaptive Router 的查询分类模型"""
    query_type: str = Field(description="Query type: 'simple' (specific fact/entity) or 'complex' (analysis/comparison/vague)")
    reasoning: str = Field(description="Brief reasoning for the classification")

QUERY_CLASSIFY_SYSTEM = (
    "You are a query router for a financial RAG system. "
    "Classify the user's query into 'simple' or 'complex'.\n"
    "- 'simple': Specific factual questions with clear entities (company, year, metric). e.g., 'AMD net income 2022'.\n"
    "- 'complex': Comparative, analytical, or vague questions requiring multi-hop reasoning. e.g., 'Compare AMD and Intel revenue trends'.\n"
    "Return ONLY JSON."
)

class QueryPreprocessor:
    """查询预处理：过滤条件抽取（结构化输出）+ 查询改写 / HyDE。"""

    def __init__(self, llm: BaseLanguageModel, use_rewrite: bool = False):
        """
        Args:
            llm: LangChain 兼容的 Chat LLM（如 ChatOpenAI、ChatOllama、init_chat_model 结果）。
            use_rewrite: 是否在 preprocess 中默认启用查询改写（默认关闭，
                参照 financebench-rag-eval：直接用原查询 + metadata 过滤即可）。
        """
        self.llm = llm
        self.use_rewrite = use_rewrite
        self._init_prompts()
        self._init_filter_chain()


    def _init_prompts(self):
        # 1. Query Rewriting：把短/模糊查询改写成关键词丰富的完整查询
        self.rewrite_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert research assistant. Rewrite the user's short or vague "
                    "query into a detailed, self-contained, keyword-rich search query. "
                    "Output ONLY the rewritten query, no explanations.",
                ), #你是一个专家级研究助理。请将用户简短或模糊的查询，改写为一个详细、独立且包含丰富关键词的搜索查询。仅输出改写后的查询语句，不要包含任何解释。
                ("human", "Original Query: {query}"),
            ]
        )

        # 2. HyDE：生成假设性文档片段引导语义检索
        self.hyde_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Write a concise, factual, professional paragraph answering the question "
                    "below, based on typical financial/SEC annual-report language. "
                    "Do not worry about exact accuracy; the goal is a hypothetical document "
                    "snippet with the right terminology and structure. Output ONLY the paragraph.",
                ),
                ("human", "Question: {query}"),
            ]
        )

    def _init_filter_chain(self):
        """构建结构化输出链（对齐 chains/query_analysis.py: prompt | llm.with_structured_output）。"""
        self._filter_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", QUERY_FILTER_SYSTEM),
                ("human", "Query: '{query}'"),
            ]
        )
        self.filter_chain = self._filter_prompt | self.llm.with_structured_output(QueryFilter)

        self._classify_prompt = ChatPromptTemplate.from_messages([
            ("system", QUERY_CLASSIFY_SYSTEM),
            ("human", "Query: '{query}'")
        ])
        self.classify_chain = self._classify_prompt | self.llm.with_structured_output(QueryType)

    def classify_query(self, query: str) -> QueryType:
        """判断查询复杂度，用于 Adaptive Routing"""
        try:
            result = self.classify_chain.invoke({"query": query})
            logging.info(f"Query classified as: {result.query_type} ({result.reasoning})")
            return result
        except Exception as e:
            logging.warning(f"Query classification failed, defaulting to complex: {e}")
            return QueryType(query_type="complex", reasoning="fallback")

        # 修改 preprocess 方法，支持路由策略

    def preprocess_adaptive(self, query: str) -> dict:
        """
        Adaptive Preprocessing: 根据查询复杂度自动选择策略
        Returns:
            {
                "search_query": str,
                "strategy": str ("original", "rewrite", "hyde"),
                "filter_token": str,
                "company_filter": str,
                "use_filter": bool (simple查询用强过滤，complex查询弱过滤)
            }
        """
        # 1. 分类
        q_type = self.classify_query(query)

        # 2. 抽取 Filter (始终抽取，但后续决定是否使用)
        filter_result = self.extract_filter(query)

        # 3. 路由决策
        if q_type.query_type == "simple":
            # 简单查询：原词检索 + 强过滤 (精准匹配)
            strategy = "original"
            use_filter = True
            search_query = query
        else:
            # 复杂查询：HyDE/改写 + 弱过滤/无过滤 (扩大召回)
            strategy = "hyde"  # 也可以根据情况选 "rewrite"
            use_filter = False  # 复杂查询往往跨越多个文档，强过滤容易漏
            search_query = self.generate_hyde(query)

        return {
            "search_query": search_query,
            "strategy": strategy,
            "filter_token": filter_result.filter_token,
            "company_filter": filter_result.company_filter,
            "use_filter": use_filter,
            "query_type": q_type.query_type
        }


    # ── Query Filter（参照 financebench-rag-eval nodes/query_analysis.py）──

    def extract_filter(self, query: str) -> QueryFilter:
        """抽取 filter_token 与 company_filter（结构化输出，失败时安全降级为 NONE）。"""
        try:
            result = self.filter_chain.invoke({"query": query})
            result.filter_token = result.filter_token.strip().upper() or "NONE"
            result.company_filter = result.company_filter.strip().upper() or "NONE"
            logging.info(
                f"Query filter extracted: filter_token='{result.filter_token}', "
                f"company_filter='{result.company_filter}'"
            )
            return result
        except Exception as e:
            logging.warning(f"Query filter extraction failed, fallback to NONE: {e}")
            return QueryFilter(filter_token="NONE", company_filter="NONE")

    @staticmethod
    def relax_filter_token(filter_token: str, company_filter: str, retry_count: int) -> str:
        """
        2-step relax（对齐 nodes/router.py 的 relax_filter）：
          retry 0: 公司+年 → 仅公司
          retry 1+: 全部放开（NONE）
        """
        if retry_count == 0:
            return company_filter or "NONE"
        return "NONE"

    # ── Query 变换 ──

    def rewrite_query(self, query: str) -> str:
        """扩展查询：补充隐含上下文与同义关键词。"""
        logging.info(f"Rewriting query: '{query}'")
        rewritten = (self.rewrite_prompt | self.llm).invoke({"query": query}).content.strip()
        logging.info(f"Rewritten to: '{rewritten}'")
        return rewritten

    def generate_hyde(self, query: str) -> str:
        """生成假设性文档片段，用于 HyDE 检索。"""
        logging.info(f"Generating HyDE for query: '{query}'")
        hyde_text = (self.hyde_prompt | self.llm).invoke({"query": query}).content.strip()
        logging.info(f"HyDE generated: '{hyde_text[:100]}...'")
        return hyde_text

    def process(
        self,
        query: str,
        strategy: str = "original",  # 可选: "rewrite" / "hyde" / "original"
    ) -> str:
        """按所选策略处理查询，返回用于嵌入检索的查询字符串。"""
        if strategy == "rewrite":
            return self.rewrite_query(query)
        if strategy == "hyde":
            return self.generate_hyde(query)
        return query

    def preprocess(
        self,
        query: str,
        strategy: Optional[str] = None,
    ) -> dict:
        """
        主入口：一次调用同时产出检索所需的一切。

        Returns:
            {
                "search_query":  处理后的查询（默认原查询）,
                "filter_token":  公司+年 token（如 AMD_2022 / NONE）,
                "company_filter": 仅公司 token（如 AMD / NONE）,
            }
        """
        strategy = strategy or ("rewrite" if self.use_rewrite else "original")
        filter_result = self.extract_filter(query)
        return {
            "search_query": self.process(query, strategy=strategy),
            "filter_token": filter_result.filter_token,
            "company_filter": filter_result.company_filter,
        }


# ──────────────────────────────────────────────
#  Demo
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("⚠️  请先配置 LLM（如 ChatOllama 或 ChatOpenAI），再运行本 demo。")
    print("""
示例:
    from langchain_ollama import ChatOllama
    from src.core.query_preprocessor import QueryPreprocessor

    llm = ChatOllama(model="llama3:8b", base_url="http://localhost:11434")
    qp = QueryPreprocessor(llm)

    result = qp.preprocess("What was AMD's net income in 2022?")
    # {'search_query': "...", 'filter_token': 'AMD_2022', 'company_filter': 'AMD'}
""")
