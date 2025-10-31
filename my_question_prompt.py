import os
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from dotenv import load_dotenv

# --- LangChain & Ollama ---
#from langchain_ollama import ChatOllama  # <-- dùng bản chính thức, không dùng langchain_community
#from langchain.llms import Ollama
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from src.route_chain import RouteCOVEChain  # dùng file refactor
# Nếu file vẫn là route_chain.py cũ thì đổi import lại cho phù hợp

# --- Tải biến môi trường (nếu cần thiết) ---
load_dotenv()

# --- Cấu hình mô hình Ollama ---
OLLAMA_MODEL_NAME = "gpt-oss:20b"  # hoặc "llama3", "mistral", v.v.
LLM_TEMP = 0.1

# --- Khởi tạo các LLM ---
try:
    chain_llm = ChatOllama(model=OLLAMA_MODEL_NAME, temperature=LLM_TEMP)
    route_llm = ChatOllama(model=OLLAMA_MODEL_NAME, temperature=0.1)
    print(f"✅ Khởi tạo Ollama LLMs thành công ({OLLAMA_MODEL_NAME}).")
except Exception as e:
    print(f"❌ Lỗi khởi tạo Ollama: {e}")
    exit()


# --- Hàm tải dataset TruthfulQA ---
def load_truthfulqa_dataset(sample_size=3):
    print(f"⏳ Đang tải bộ dữ liệu TruthfulQA (sample={sample_size})...")
    try:
        dataset = load_dataset("truthful_qa", "generation")
        subset = dataset["validation"].select(range(sample_size))
        print("✅ Đã tải dữ liệu thành công.")
        return subset
    except Exception as e:
        print(f"❌ Lỗi khi tải dataset TruthfulQA: {e}")
        return None


# --- Hàm đánh giá CoVE ---
def evaluate_cove(dataset, route_llm, chain_llm):
    """Chạy CoVE routing và ghi lại kết quả."""
    if dataset is None:
        return pd.DataFrame()

    results = []

    for example in tqdm(dataset, desc="Đánh giá CoVE"):
        question = example["question"]

        try:
            # 1️⃣ Khởi tạo Router
            router_chain = RouteCOVEChain(
                question=question,
                route_llm=route_llm,
                chain_llm=chain_llm,
                show_intermediate_steps=False
            )

            # 2️⃣ Route ra chain phù hợp
            selected_chain = router_chain()

            # 3️⃣ Gọi chain đó để lấy câu trả lời
            response = selected_chain.invoke({"original_question": question})

            # Với ChatOllama, output có thể là object có .content
            if hasattr(response, "content"):
                final_answer = response.content
            elif isinstance(response, dict) and "final_answer" in response:
                final_answer = response["final_answer"]
            else:
                final_answer = str(response)

            # 4️⃣ Lưu kết quả
            results.append({
                "question": question,
                "cove_answer": final_answer.strip(),
                "true_answer": example["best_answer"]
            })

        except Exception as e:
            results.append({
                "question": question,
                "cove_answer": f"LỖI: {e}",
                "true_answer": example["best_answer"]
            })

    return pd.DataFrame(results)


# --- Chạy chính ---
if __name__ == "__main__":
    dataset = load_truthfulqa_dataset(sample_size=200)
    results_df = evaluate_cove(dataset, route_llm, chain_llm)

    output_file = "TruthfulQA_200.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✨ Hoàn tất! Kết quả lưu tại: {output_file}")

    # In nhanh 5 kết quả đầu
    try:
        print("\n--- 5 Kết Quả Đầu Tiên ---")
        print(results_df[["question", "cove_answer", "true_answer"]].head().to_markdown(index=False))
    except Exception:
        print(results_df.head())
