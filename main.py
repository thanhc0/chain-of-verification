#import argparse
#from dotenv import load_dotenv
#from pprint import pprint

#from langchain.chat_models import ChatOpenAI
from src.route_chain import RouteCOVEChain
from src.question_preprocessing import QuestionPreprocessor
from langchain_community.chat_models import ChatOllama

import warnings
import logging
import os

# --- TẮT TOÀN BỘ CẢNH BÁO ---
warnings.filterwarnings("ignore")
logging.getLogger("langchain").setLevel(logging.CRITICAL)
logging.getLogger("langchain_core").setLevel(logging.CRITICAL)
logging.getLogger("openai").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)

LLM_TEMP = 0.1
SAVE_EVERY = 5  # checkpoint every N processed rows
SLEEP_BETWEEN_CALLS = 0.2  # throttle
OLLAMA_MODEL_NAME = "gpt-oss:20b"
#OLLAMA_MODEL_NAME = "gemma3:1b"
#OLLAMA_MODEL_NAME = "gemma3:12b"
# --- Init LLMs ---
try:
    chain_llm = ChatOllama(model=OLLAMA_MODEL_NAME, temperature=LLM_TEMP)
    route_llm = ChatOllama(model=OLLAMA_MODEL_NAME, temperature=0.1)
    print(f"✅ Ollama initialized ({OLLAMA_MODEL_NAME})")
except Exception as e:
    print(f"❌ Failed to init Ollama: {e}")
    raise SystemExit(1)


#original_query = """uhm what’s like the name of that thing from google that ai thing"""
original_query = """Lan: Cụ già đi nhanh quá

Trả loi câu hỏi trên: 
Người được đề cập còn sống hay đã chết?
Cụ già, hay cụ, hay người được nói tới, còn khỏe không?

Chỉ trả lời bằng cais gi đơn giản, ngắn gọn, ko giả thích lằng nhằng.
Nếu, có nhiều khung cảnh lẫn lộn, với mỗi bối cảnh: Thêm tiêu đề của bối cảnh, và 2 câu trả lời ngắn gọn kèm theo.
Trả lời bằng tiếng việt."""

if __name__ == "__main__":




    preprocessor = QuestionPreprocessor(route_llm, show_steps=True)

    # B1: Chuẩn hóa câu hỏi
    normalized_query = preprocessor(original_query)
    #original_query = preprocessor(original_query)
    #normalized_query = original_query

    # B2: Route đến chain phù hợp
    router_cove_chain_instance = RouteCOVEChain(
        question=normalized_query,
        route_llm=route_llm,
        chain_llm=chain_llm,
        show_intermediate_steps=False,
        use_search=True
    )
    router_cove_chain = router_cove_chain_instance()
    router_cove_chain_result = router_cove_chain.invoke({"original_question": normalized_query})
    print("Baseline Answer: {}".format(router_cove_chain_result["baseline_response"]))
    print("Final Answer: {}".format(router_cove_chain_result["final_answer"]))
