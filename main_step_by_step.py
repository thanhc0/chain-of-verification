# run_cove_batch.py
import os
import time
import json
import pandas as pd
from tqdm import tqdm

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.chat_models import ChatOllama

from src.question_preprocessing import QuestionPreprocessor
from src.route_chain import RouteCOVEChain
from src import prompts

# CONFIG
INPUT_CSV = "normalized_cove_data.csv"
OUTPUT_CSV = "normalized_cove_data_result.csv"
MODELS = ["gpt-oss:20b", "gemma3:1b", "gemma3:12b"]
LLM_TEMP = 0.1
SLEEP_BETWEEN_CALLS = 0.2
SAVE_EVERY = 1  # checkpoint every N processed rows (per model-question pair)


def classify_question_with_router(route_llm, question: str) -> str:
    """
    Use the router prompt to classify the question into a category.
    Returns the detected category string (uppercased, e.g. 'WIKI_CHAIN' or fallback 'SINGLEFACT_CHAIN').
    """
    router_prompt = ChatPromptTemplate.from_template(prompts.ROUTER_CHAIN_PROMPT)
    router_chain = router_prompt | route_llm | StrOutputParser()
    raw = router_chain.invoke({"question": question})
    if raw is None:
        return "SINGLEFACT_CHAIN"
    rs = str(raw).strip().lower()
    if "wiki_chain" in rs:
        return "WIKI_CHAIN"
    if "multi_chain" in rs:
        return "MULTI_CHAIN"
    if "long_chain" in rs:
        return "LONG_CHAIN"
    if "singlefact_chain" in rs or "single_chain" in rs or "single" in rs:
        return "SINGLEFACT_CHAIN"
    return "SINGLEFACT_CHAIN"


def extract_values_from_chain_result(result: dict):
    """
    From the SequentialChain/LLMChain result dict, extract baseline_response, verification_answers (as list of dicts),
    and final_answer. The result shape depends on chain used, so be defensive.
    """
    baseline = ""
    verification_raw = ""
    verification_list = []
    refined = ""

    # baseline_response key common
    if "baseline_response" in result:
        baseline = result["baseline_response"]
    # sometimes baseline may be in other keys
    elif "baseline" in result:
        baseline = result["baseline"]

    # verification answers might be present as a JSON string under 'verification_answers'
    if "verification_answers" in result:
        verification_raw = result["verification_answers"]
        # verification_raw often is a JSON string like: [{ "question": "...", "answer": "..." }, ...]
        try:
            if isinstance(verification_raw, str):
                verification_list = json.loads(verification_raw)
            elif isinstance(verification_raw, list):
                verification_list = verification_raw
            else:
                # fallback: try to parse text lines
                verification_list = []
        except Exception:
            # fallback: try to split lines into Q/A (best-effort)
            verification_list = []
            text = str(verification_raw)
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            # pair lines into QA pairs if possible
            for i in range(0, len(lines), 2):
                q = lines[i]
                a = lines[i+1] if i+1 < len(lines) else ""
                verification_list.append({"question": q, "answer": a})
    else:
        # sometimes chains return a single string of "1. Q\nA\n2. Q2\nA2" under other keys; try to search
        for k in ["verification_questions", "verification_qas", "verification_q_and_a"]:
            if k in result:
                verification_raw = result[k]
                try:
                    verification_list = json.loads(verification_raw) if isinstance(verification_raw, str) else verification_raw
                except Exception:
                    text = str(verification_raw)
                    lines = [l.strip() for l in text.splitlines() if l.strip()]
                    for i in range(0, len(lines), 2):
                        q = lines[i]
                        a = lines[i+1] if i+1 < len(lines) else ""
                        verification_list.append({"question": q, "answer": a})
                break

    # final refined answer
    if "final_answer" in result:
        refined = result["final_answer"]
    elif "final" in result:
        refined = result["final"]
    elif "final_answer_text" in result:
        refined = result["final_answer_text"]
    elif "baseline_response" in result and not verification_list:
        # fallback: if no verification then baseline might be the final
        refined = result.get("baseline_response", "")
    else:
        # last resort: check any value that looks like a big text
        for v in result.values():
            if isinstance(v, str) and len(v) > len(refined):
                refined = v

    return baseline, verification_list, refined


def format_verification_list(verification_list):
    """
    Given list of {"question":..., "answer":...}, produce numbered lines as user requested:
    1. <question>
    <answer>
    2. <question2>
    <answer2>
    """
    if not verification_list:
        return ""
    lines = []
    for i, qa in enumerate(verification_list, start=1):
        q = qa.get("question") if isinstance(qa, dict) else str(qa)
        a = qa.get("answer", "") if isinstance(qa, dict) else ""
        lines.append(f"{i}. {q}\n{a}")
    return "\n".join(lines)


def run_batch(input_csv=INPUT_CSV, output_csv=OUTPUT_CSV):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df_in = pd.read_csv(input_csv)
    if "question" not in df_in.columns:
        raise ValueError("Input CSV must contain column 'question'")

    # prepare output rows
    columns = ["question", "model", "category", "normalized_question",
               "baseline_answer", "verification_question_answer", "refined_answer", "score"]
    if os.path.exists(output_csv):
        df_out = pd.read_csv(output_csv)
    else:
        df_out = pd.DataFrame(columns=columns)

    rows = []

    # iterate each question
    for idx, row in tqdm(df_in.iterrows(), total=len(df_in), desc="Questions"):
        question = str(row["question"]).strip()
        if question == "":
            continue

        # For each model, run the pipeline
        for model_name in MODELS:
            try:
                # init llms for this model
                chain_llm = ChatOllama(model=model_name, temperature=LLM_TEMP)
                route_llm = ChatOllama(model=model_name, temperature=0.0)

                # 1) classify category using router prompt (so we store category even if router is later used)
                category = classify_question_with_router(route_llm, question)

                # 2) normalize question
                preprocessor = QuestionPreprocessor(route_llm, show_steps=False)
                normalized_question = preprocessor(question)

                # 3) build route chain and invoke
                router_cove_chain_instance = RouteCOVEChain(
                    question=normalized_question,
                    route_llm=route_llm,
                    chain_llm=chain_llm,
                    show_intermediate_steps=False
                )
                router_cove_chain = router_cove_chain_instance()
                # invoke runnable
                result = router_cove_chain.invoke({"original_question": normalized_question})

                # 4) extract baseline, verification list, refined
                baseline_answer, verification_list, refined_answer = extract_values_from_chain_result(result)

                # 5) format verification QA as requested
                verification_qa_text = format_verification_list(verification_list)

                # 6) score (placeholder - you can implement your scoring logic)
                score = ""

                # append to df_out and save checkpoint
                out_row = {
                    "question": question,
                    "model": model_name,
                    "category": category,
                    "normalized_question": normalized_question,
                    "baseline_answer": baseline_answer,
                    "verification_question_answer": verification_qa_text,
                    "refined_answer": refined_answer,
                    "score": score
                }
                df_out = pd.concat([df_out, pd.DataFrame([out_row])], ignore_index=True)

                # save checkpoint
                if len(df_out) % SAVE_EVERY == 0:
                    df_out.to_csv(output_csv, index=False, encoding="utf-8-sig")

                # throttle
                time.sleep(SLEEP_BETWEEN_CALLS)

            except Exception as e:
                print(f"[ERROR] model={model_name} question_idx={idx} error={e}")
                # record error row
                err_row = {
                    "question": question,
                    "model": model_name,
                    "category": category if 'category' in locals() else "",
                    "normalized_question": normalized_question if 'normalized_question' in locals() else "",
                    "baseline_answer": "",
                    "verification_question_answer": f"ERROR: {e}",
                    "refined_answer": "",
                    "score": ""
                }
                df_out = pd.concat([df_out, pd.DataFrame([err_row])], ignore_index=True)
                df_out.to_csv(output_csv, index=False, encoding="utf-8-sig")
                time.sleep(1)

    # final save
    df_out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved results to {output_csv}")


if __name__ == "__main__":
    run_batch()
