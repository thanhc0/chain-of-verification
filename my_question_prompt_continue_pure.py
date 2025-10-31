# fill_missing_pure_answers_safe_with_prompt.py (hoặc _pure.py)
import os
import pandas as pd
import unicodedata
import re
import time
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate  # Thêm import cần thiết

# --- Load environment ---
load_dotenv()

# --- Config ---
INPUT_FILE = "pure_truthfulqa_results_ollama_filled.csv"
OUTPUT_FILE = "pure_truthfulqa_results_ollama_filled.csv"
OLLAMA_MODEL_NAME = "gemma3:1b"
#Model = "gpt-oss-20b"
Model = "gemma3-1b"

LLM_TEMP = 0.1
SAVE_EVERY = 10  # checkpoint mỗi N hàng
SLEEP_BETWEEN_CALLS = 0.2  # Tốc độ gọi API

# --- Init LLM ---
try:
    # Khởi tạo một LLM duy nhất
    pure_llm = ChatOllama(model=OLLAMA_MODEL_NAME, temperature=LLM_TEMP)
    print(f"✅ Ollama initialized ({OLLAMA_MODEL_NAME}) for pure answers.")
except Exception as e:
    print(f"❌ Failed to init Ollama: {e}")
    raise SystemExit(1)

# --- Fact Chain Prompt (Giữ nguyên yêu cầu) ---
FACT_CHAIN_PROMPT = ChatPromptTemplate.from_template(
    "Question: {original_question}\n"
    "Give a short answer in one sentence. "
    "Avoid unnecessary details and explanations. "
    "Do not use line breaks or bullet points."
)


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
    # If any keyword occurs within first 1000 chars, treat as error
    preview = low[:1000]
    return any(kw in preview for kw in error_keywords)


def generate_pure_answer(question: str, llm: ChatOllama, prompt_template: ChatPromptTemplate) -> tuple[str, str]:
    """Generate a direct answer from the LLM using the specified prompt template."""
    try:
        # Xây dựng chuỗi (chain) bao gồm prompt và LLM
        chain = prompt_template | llm

        # Gọi chuỗi với biến đầu vào
        response = chain.invoke({"original_question": question})

        # Normalize how we extract text
        if hasattr(response, "content"):
            raw = response.content
        else:
            raw = str(response)

        cleaned = clean_text(raw)

        # Giữ câu trả lời ngắn gọn (chỉ lấy tối đa 500 ký tự)
        shortened_cleaned = cleaned[:500]
        if len(cleaned) > 500:
            shortened_cleaned += "..."

        return shortened_cleaned, raw

    except Exception as e:
        raw_err = f"EXCEPTION_IN_GENERATION: {repr(e)}"
        return "", raw_err


# --- Load data ---
if not os.path.exists(INPUT_FILE):
    print(f"❌ Input file not found: {INPUT_FILE}")
    raise SystemExit(1)

# read with utf-8 first; if fails try utf-8-sig
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

# Kiểm tra và thêm các cột cần thiết
required = {"question"}
if not required.issubset(df.columns):
    raise ValueError(f"Input CSV must contain column: 'question'. Found: {set(df.columns)}")

if "pure_answer" not in df.columns:
    df["pure_answer"] = ""
#if "pure_error_info" not in df.columns:
#    df["pure_error_info"] = ""

# Lưu lại DataFrame mới với các cột trống nếu file đầu ra chưa tồn tại
if not os.path.exists(OUTPUT_FILE):
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

# Tải lại file đầu ra để tiếp tục nếu có checkpoint
try:
    df = pd.read_csv(OUTPUT_FILE, encoding="utf-8")
except Exception:
    df = pd.read_csv(OUTPUT_FILE, encoding="utf-8-sig")


# chuẩn bị indices để xử lý (những hàng có pure_answer trống)
def is_empty_cell(val):
    return pd.isna(val) or (isinstance(val, str) and val.strip() == "")


to_process_idx = [idx for idx, val in df["pure_answer"].items() if is_empty_cell(val)]

print(f"🔍 Found {len(to_process_idx)} rows with empty 'pure_answer'.")

if len(to_process_idx) == 0:
    print("✅ Nothing to fill in 'pure_answer'. Exiting.")
    raise SystemExit(0)

# --- Process with safe skip-on-error and periodic checkpointing ---
processed_count = 0
skipped_count = 0

for i, idx in enumerate(tqdm(to_process_idx, desc="Filling missing pure_answer")):
    question = str(df.at[idx, "question"]).strip()

    # Generate pure answer, truyền prompt_template
    answer_clean, raw_resp = generate_pure_answer(question, pure_llm, FACT_CHAIN_PROMPT)

    # Xử lý Lỗi/Bỏ qua
    if raw_resp.startswith("EXCEPTION_IN_GENERATION") or is_error_response(raw_resp):
        print("\n" + "=" * 80)
        print(f"[ERROR] Generation error for row {idx} (question preview): {question[:120]}")
        print("Full response/raw output from model or exception:")
        print(raw_resp)
        print("=" * 80 + "\n")

        #df.at[idx, "pure_error_info"] = raw_resp
        df.at[idx, "pure_answer"] = ""
        skipped_count += 1
    else:
        # Thành công
        df.at[idx, "pure_answer"] = answer_clean
        #df.at[idx, "pure_error_info"] = ""  # xóa thông tin lỗi cũ
        processed_count += 1

    # checkpoint
    if (i + 1) % SAVE_EVERY == 0:
        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        print(f"💾 Checkpoint saved after {i + 1} attempts (processed {processed_count}, skipped {skipped_count}).")

    time.sleep(SLEEP_BETWEEN_CALLS)

# final save
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"\n✨ Done. Processed: {processed_count}, Skipped (errors): {skipped_count}.")
print(f"Results saved to: {OUTPUT_FILE}")