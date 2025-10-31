# fill_missing_cove_answers_safe.py
import os
import pandas as pd
import unicodedata
import re
import time
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from src.route_chain import RouteCOVEChain

# --- Load environment ---
load_dotenv()

# --- Config ---
INPUT_FILE = "TruthfulQA_200.csv"
OUTPUT_FILE = "TruthfulQA_200.csv"
#OLLAMA_MODEL_NAME = "gpt-oss:20b"
OLLAMA_MODEL_NAME = "gemma3:1b"
LLM_TEMP = 0.1
SAVE_EVERY = 5  # checkpoint every N processed rows
SLEEP_BETWEEN_CALLS = 0.2  # throttle

# --- Init LLMs ---
try:
    chain_llm = ChatOllama(model=OLLAMA_MODEL_NAME, temperature=LLM_TEMP)
    route_llm = ChatOllama(model=OLLAMA_MODEL_NAME, temperature=0.1)
    print(f"✅ Ollama initialized ({OLLAMA_MODEL_NAME})")
except Exception as e:
    print(f"❌ Failed to init Ollama: {e}")
    raise SystemExit(1)


# --- Helpers ---
def clean_text(text: str) -> str:
    """Normalize Unicode and remove non-printable / weird characters; return a single-line string."""
    try:
        if not isinstance(text, str):
            text = str(text)
    except Exception:
        text = ""
    text = unicodedata.normalize("NFC", text)
    # Allow basic Latin and common Latin-extended + Vietnamese characters; replace others with space
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7EÀ-ỹĐđ]", " ", text)
    text = " ".join(text.split())
    return text.strip()


def is_error_response(answer: str) -> bool:
    """Return True if the answer looks like an error/HTTP/exception message."""
    if answer is None:
        return True
    if not isinstance(answer, str):
        return True
    low = answer.lower()
    # common error indicators
    error_keywords = [
        "error", "http", "failed", "timeout", "connection",
        "traceback", "exception", "lỗi", "ollama", "invalid",
        "unavailable", "503", "504", "404", "network", "broken",
        "rate limit", "service not", "service unavailable"
    ]
    # If any keyword occurs within first 200 chars, treat as error
    preview = low[:1000]
    return any(kw in preview for kw in error_keywords)


def generate_cove_answer(question, route_llm, chain_llm):
    """Generate CoVE answer using router + selected chain. Returns (answer_str, raw_response)."""
    try:
        router_chain = RouteCOVEChain(
            question=question,
            route_llm=route_llm,
            chain_llm=chain_llm,
            show_intermediate_steps=False
        )
        selected_chain = router_chain()
        response = selected_chain.invoke({"original_question": question})

        # Normalize how we extract text
        if hasattr(response, "content"):
            raw = response.content
        elif isinstance(response, dict) and "final_answer" in response:
            raw = response["final_answer"]
        else:
            raw = str(response)

        cleaned = clean_text(raw)
        return cleaned, raw

    except Exception as e:
        raw_err = f"EXCEPTION_IN_GENERATION: {repr(e)}"
        return "", raw_err


# --- Load data ---
if not os.path.exists(INPUT_FILE):
    print(f"❌ Input file not found: {INPUT_FILE}")
    raise SystemExit(1)

encodings_to_try = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

# BƯỚC 1: Tải file đầu vào (INPUT_FILE)
df = None
for encoding in encodings_to_try:
    try:
        df = pd.read_csv(INPUT_FILE, encoding=encoding)
        print(f"✅ Input file loaded successfully with encoding: {encoding}")
        break # Thoát vòng lặp nếu thành công
    except Exception as e:
        print(f"❌ Failed to load INPUT_FILE with {encoding}: {e}")
        continue

if df is None:
    raise RuntimeError(f"Failed to load input file {INPUT_FILE} with all tested encodings.")

print(f"📄 Loaded {len(df)} rows from {INPUT_FILE}")

# Ensure required columns, add 'error_info' column if absent
required = {"question", "cove_answer", "true_answer", "score"}
if not required.issubset(df.columns):
    raise ValueError(f"Input CSV must contain columns: {required}. Found: {set(df.columns)}")

#if "error_info" not in df.columns:
#    df["error_info"] = ""

# prepare indices to process (those with empty cove_answer)
def is_empty_cell(val):
    return pd.isna(val) or (isinstance(val, str) and val.strip() == "")

to_process_idx = [idx for idx, val in df["cove_answer"].items() if is_empty_cell(val)]

print(f"🔍 Found {len(to_process_idx)} rows with empty 'cove_answer'.")

if len(to_process_idx) == 0:
    print("✅ Nothing to fill. Exiting.")
    raise SystemExit(0)

# --- Process with safe skip-on-error and periodic checkpointing ---
processed_count = 0
skipped_count = 0

for i, idx in enumerate(tqdm(to_process_idx, desc="Filling missing cove_answer")):
    question = str(df.at[idx, "question"]).strip()
    # Generate
    answer_clean, raw_resp = generate_cove_answer(question, route_llm, chain_llm)

    # If generation returned raw error string starting with EXCEPTION..., treat as error
    if raw_resp.startswith("EXCEPTION_IN_GENERATION") or is_error_response(raw_resp):
        # Print full error/response to console for debugging
        print("\n" + "=" * 80)
        print(f"[ERROR] Generation error for row {idx} (question preview): {question[:120]}")
        print("Full response/raw output from model or exception:")
        print(raw_resp)  # print raw so you see the whole message
        print("=" * 80 + "\n")

        # Record error in DataFrame column, leave cove_answer empty (or set to placeholder)
        #df.at[idx, "error_info"] = raw_resp
        skipped_count += 1

        # continue to next row (do not break)
        # optionally: you could also set df.at[idx, "cove_answer"] = "" or a special tag
        df.at[idx, "cove_answer"] = ""  # keep empty so can revisit later
    else:
        # success: write cleaned answer, clear any previous error_info
        df.at[idx, "cove_answer"] = answer_clean
        #df.at[idx, "error_info"] = raw_resp
        processed_count += 1

    # checkpoint every SAVE_EVERY rows attempted (processed + skipped)
    if (i + 1) % SAVE_EVERY == 0:
        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        print(f"💾 Checkpoint saved after {i+1} attempts (processed {processed_count}, skipped {skipped_count}).")

    time.sleep(SLEEP_BETWEEN_CALLS)

# final save
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"\n✨ Done. Processed: {processed_count}, Skipped (errors): {skipped_count}.")
print(f"Results saved to: {OUTPUT_FILE}")
