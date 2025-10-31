# execute_verification_chain_refactored.py

import itertools
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from . import prompts


class ExecuteVerificationChain:
    """
    Chuỗi xử lý xác minh thông tin (Verification Chain).
    Dựa vào danh sách câu hỏi xác minh, có thể dùng search tool hoặc chính LLM.
    """

    def __init__(self, llm: ChatOllama, use_search_tool=True):
        self.llm = llm
        self.use_search_tool = use_search_tool
        self.search_tool = DuckDuckGoSearchRun()

        # Prompt template# execute_verification_chain.py
        # import pandas as pd
        # from datasets import load_dataset
        # from tqdm import tqdm
        # from dotenv import load_dotenv
        # import os
        #
        # from langchain_community.chat_models import ChatOllama
        # from langchain_ollama import Ollama
        # from src.route_chain import RouteCOVEChain
        #
        # load_dotenv()
        #
        # OLLAMA_MODEL_NAME = "llama3.1:8b"  # đổi tùy model bạn đã pull
        # LLM_TEMP = 0.1
        #
        # try:
        #     chain_llm = Ollama(model=OLLAMA_MODEL_NAME, temperature=LLM_TEMP)
        #     route_llm = ChatOllama(model=OLLAMA_MODEL_NAME, temperature=0.1)
        #     print(f"✅ Khởi tạo LLMs bằng Ollama ({OLLAMA_MODEL_NAME}) thành công.")
        # except Exception as e:
        #     print(f"❌ Lỗi khởi tạo LLMs bằng Ollama: {e}")
        #     exit()
        #
        #
        # def load_truthfulqa_dataset(sample_size=20):
        #     print(f"⏳ Đang tải TruthfulQA ({sample_size} mẫu)...")
        #     try:
        #         dataset = load_dataset("truthful_qa", "generation")
        #         return dataset["validation"].select(range(sample_size))
        #     except Exception as e:
        #         print(f"❌ Lỗi tải dataset: {e}")
        #         return None
        #
        #
        # def evaluate_cove(dataset, route_llm, chain_llm):
        #     if dataset is None:
        #         return pd.DataFrame()
        #
        #     results = []
        #     for example in tqdm(dataset, desc="Đánh giá CoVe"):
        #         q = example["question"]
        #         try:
        #             router = RouteCOVEChain(q, route_llm, chain_llm, show_intermediate_steps=True)
        #             selected_chain = router()  # runnable
        #             output = selected_chain.invoke({"original_question": q})
        #             ans = output.get("final_answer", "")
        #         except Exception as e:
        #             ans = f"LỖI: {e}"
        #
        #         results.append({
        #             "question": q,
        #             "cove_answer": ans,
        #             "true_answer": example.get("best_answer", "")
        #         })
        #
        #     return pd.DataFrame(results)
        #
        #
        # if __name__ == "__main__":
        #     data = load_truthfulqa_dataset(10)
        #     df = evaluate_cove(data, route_llm, chain_llm)
        #     output_path = "cove_truthfulqa_results_ollama.csv"
        #     df.to_csv(output_path, index=False)
        #     print(f"✨ Đã lưu kết quả tại: {output_path}")
        #     print(df[["question", "cove_answer", "true_answer"]].head().to_markdown(index=False))