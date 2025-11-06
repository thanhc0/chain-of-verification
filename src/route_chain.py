# route_chain.py
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.prompts import ROUTER_CHAIN_PROMPT
from src.cove_chains import (
    WikiDataCategoryListCOVEChain,
    MultiSpanCOVEChain,
    LongFormCOVEChain,
    SingleFactCOVEChain,
)


class RouteCOVEChain:
    """
    Router chain: quyết định loại CoVE chain cần dùng cho từng câu hỏi
    (WIKI_CHAIN, MULTI_CHAIN, LONG_CHAIN, SINGLEFACT_CHAIN).
    """

    def __init__(self, question: str, route_llm, chain_llm, show_intermediate_steps=False, use_search=True):
        self.question = question
        self.route_llm = route_llm
        self.chain_llm = chain_llm
        self.show_steps = show_intermediate_steps
        self.use_search = use_search

    def __call__(self):
        """Thực hiện routing và trả về runnable chain tương ứng."""
        router_prompt = ChatPromptTemplate.from_template(ROUTER_CHAIN_PROMPT)
        router_chain = router_prompt | self.route_llm | StrOutputParser()

        # Lấy output JSON {"category": "XYZ"}
        raw_route = router_chain.invoke({"question": self.question})
        route_str = raw_route.strip().lower()

        if self.show_steps:
            print(f"[Router raw output]: {raw_route}")

        # Xác định loại chain
        if "wiki_chain" in route_str:
            category = "WIKI_CHAIN"
        elif "multi_chain" in route_str:
            category = "MULTI_CHAIN"
        elif "long_chain" in route_str:
            category = "LONG_CHAIN"
        elif "singlefact_chain" in route_str or "single_chain" in route_str:
            category = "SINGLEFACT_CHAIN"
        else:
            category = "SINGLEFACT_CHAIN"  # fallback

        if self.show_steps:
            print(f"[Router decision]: {category}")
        else:
            print(f"🧭 Category selected: {category}")

        # Trả về runnable tương ứng (chuẩn cú pháp langchain)
        if category == "WIKI_CHAIN":
            return WikiDataCategoryListCOVEChain(self.chain_llm, self.use_search)()
        elif category == "MULTI_CHAIN":
            return MultiSpanCOVEChain(self.chain_llm, self.use_search)()
        elif category == "LONG_CHAIN":
            return LongFormCOVEChain(self.chain_llm, self.use_search)()
        else:  # SINGLEFACT hoặc fallback
            return SingleFactCOVEChain(self.chain_llm, self.use_search)()
