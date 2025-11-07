# main_step_by_step_baseline_only_safe.py
import os
import time
import re
import unicodedata
import pandas as pd
from tqdm import tqdm

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.chat_models import ChatOllama


# --- CONFIG ---
INPUT_CSV = "normalized_cove_data.csv"
OUTPUT_CSV = "normalized_cove_data_result_baseline_only.csv"
MODELS = ["gemma3:1b"]
LLM_TEMP = 0.1
SLEEP_BETWEEN_CALLS = 0.2


# --- Utility Functions ---
def clean_text(text: str) -> str:
    try:
        if not isinstance(text, str):
            text = str(text)
    except Exception:
        text = ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7EÀ-ỹĐđ]", " ", text)
    text = " ".join(text.split())
    return text.strip()


# --- LLM Utility ---
def get_single_fact_answer(question: str, llm: ChatOllama):
    prompt_template = ChatPromptTemplate.from_template(
        """
        Answer the below question correctly and concisely in one very short, factual sentence.
Do not provide explanations, context, or multiple answers.
Example question 1: How often does spermatogeneis—the production of sperm—occur?
Example answer 1: 74 days
Example question 2: When was the first remote control tv invented?
Example answer 2: 1950

Question: {question}

Answer:
        """
    )
    chain = prompt_template | llm | StrOutputParser()
    try:
        answer = chain.invoke({"question": question})
        return answer.strip()
    except Exception as e:
        return f"LLM ERROR: {repr(e)}"


def get_normalized_question(question: str, llm: ChatOllama):
    """
    Sử dụng LLM để làm rõ câu hỏi không rõ ràng, giúp chuẩn hóa truy vấn.
    """
    prompt_template = ChatPromptTemplate.from_template(
        """
You are a question clarifier. Given the ambiguous question below, rewrite it into a clear, single-fact question. 
If the question is already clear, return the original question.

Ambiguous Question: {question}
Normalized Question: 
"""
    )
    chain = prompt_template | llm | StrOutputParser()
    try:
        normalized_q = chain.invoke({"question": question}).strip()
        return normalized_q
    except Exception:
        return question


def evaluate_answer(reference_answer: str, generated_answer: str, llm):
    if llm is None:
        return None, "LLM not initialized."
    prompt = ChatPromptTemplate.from_template(
        """
You are an expert evaluator for factual accuracy.

Given:
- Reference answer (trusted): {reference_answer}
- Model-generated answer (to evaluate): {generated_answer}

Task:
Evaluate how factually consistent and correct the generated answer is compared to the reference.
Return ONLY one integer score from 0 to 100 (higher = more accurate).

Output format:
<number>
"""
    )
    parser = StrOutputParser()
    chain = prompt | llm | parser
    try:
        response = chain.invoke({
            "reference_answer": reference_answer,
            "generated_answer": generated_answer
        }).strip()
        nums = re.findall(r'\d+', response)
        if not nums:
            raise ValueError(f"No number found in response: {response}")
        score = int(nums[0])
        score = min(max(score, 0), 100)
        return score, response
    except Exception as e:
        return None, f"ERROR: {repr(e)}"


# --- Core Processing ---
def process_data_with_evaluation_safe(df: pd.DataFrame, chain_llm: ChatOllama, eval_llm: ChatOllama,
                                      output_csv: str, model_name: str):
    df = df.rename(columns={'best_answer': 'reference_answer'}, errors='ignore')

    # Bổ sung các cột cần thiết
    new_cols = ['model', 'baseline_answer', 'normalized_question',
                'normalized_answer', 'baseline_score', 'normalized_score']
    for col in new_cols:
        if col not in df.columns:
            df[col] = ''

    # Gán model
    df["model"] = model_name

    # Xác định hàng cần xử lý
    to_process_idx = [idx for idx, val in df['baseline_score'].items() if pd.isna(val) or val == '']

    if not to_process_idx:
        print("✅ Không có hàng nào cần xử lý hoặc đánh giá.")
        return df

    print(f"🔍 Tìm thấy {len(to_process_idx)} hàng cần xử lý ({model_name}).")

    for i, idx in enumerate(tqdm(to_process_idx, desc=f"Model {model_name}")):
        question = str(df.at[idx, "question"]).strip()
        reference_answer = str(df.at[idx, "reference_answer"]).strip()

        # Baseline answer
        base_ans = get_single_fact_answer(question, chain_llm)
        df.at[idx, 'baseline_answer'] = clean_text(base_ans)

        # Normalized question + answer
        norm_q = get_normalized_question(question, chain_llm)
        df.at[idx, 'normalized_question'] = clean_text(norm_q)

        norm_ans = get_single_fact_answer(norm_q, chain_llm)
        df.at[idx, 'normalized_answer'] = clean_text(norm_ans)

        # Evaluate
        score_base, _ = evaluate_answer(reference_answer, base_ans, eval_llm)
        score_norm, _ = evaluate_answer(reference_answer, norm_ans, eval_llm)

        df.at[idx, 'baseline_score'] = score_base if score_base is not None else -1
        df.at[idx, 'normalized_score'] = score_norm if score_norm is not None else -1

        # --- ✅ Checkpoint: ghi ngay sau mỗi hàng ---
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")

        print(f"[Checkpoint] Row {idx} saved → baseline={score_base}, normalized={score_norm}")
        time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"✨ Hoàn thành model {model_name}.")
    return df


# --- Entry Point ---
def run_batch_safe(input_csv=INPUT_CSV, output_csv=OUTPUT_CSV):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Không tìm thấy file input: {input_csv}")

    df_input = pd.read_csv(input_csv)
    if "question" not in df_input.columns:
        raise ValueError("File input phải có cột 'question'")
    if "best_answer" not in df_input.columns:
        raise ValueError("File input phải có cột 'best_answer' (ground truth)")

    # Nếu file output đã tồn tại → tự phục hồi
    if os.path.exists(output_csv):
        print(f"♻️ Đang khôi phục từ file {output_csv} ...")
        df_output = pd.read_csv(output_csv)
        # Giữ lại các hàng chưa có điểm
        # Hợp nhất dữ liệu đầu vào và đầu ra cũ
        merged = pd.merge(df_input, df_output, how="outer", on="question", suffixes=('', '_old'))

        # Ưu tiên giữ dữ liệu đã có trong file kết quả cũ
        for col in df_output.columns:
            col_old = f"{col}_old"
            if col in merged.columns and col_old in merged.columns:
                merged[col] = merged[col].combine_first(merged[col_old])

        # Xóa cột tạm thời
        merged = merged[[c for c in merged.columns if not c.endswith('_old')]]

        df_input = merged
        print(f"✅ Đã phục hồi {len(df_output)} hàng, tiếp tục phần còn lại.")
    else:
        print("🚀 Bắt đầu mới (chưa có file kết quả).")

    # Chạy lần lượt từng model
    for model_name in MODELS:
        print(f"\n======================")
        print(f"🔹 Running model: {model_name}")
        print(f"======================")
        try:
            chain_llm = ChatOllama(model=model_name, temperature=LLM_TEMP)
            eval_llm = ChatOllama(model=model_name, temperature=0.0)
            df_input = process_data_with_evaluation_safe(df_input, chain_llm, eval_llm,
                                                         output_csv=output_csv, model_name=model_name)
        except Exception as e:
            print(f"❌ Error with model {model_name}: {e}")
            continue

    print(f"\n✅ Hoàn tất toàn bộ. Kết quả lưu tại: {output_csv}")


if __name__ == "__main__":
    run_batch_safe()
