# src/execute_verification_chain.py
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import DuckDuckGoSearchRun
from langchain_core.output_parsers import StrOutputParser
import json
import time
from langchain.chains.base import Chain
from langchain.prompts import PromptTemplate
from langchain.tools import DuckDuckGoSearchRun
import json
import time
from typing import Any, List, Dict, Optional


class ExecuteVerificationChain(Chain):
    """
    Chain thực thi và kiểm chứng các câu hỏi xác minh (verification questions).
    Có thể dùng search tool (DuckDuckGo) để lấy thông tin thực tế.
    """
    llm: any
    prompt: any
    output_key: str = "verification_answers"
    use_search: bool = False

    search_tool: Optional[DuckDuckGoSearchRun] = None  # 🔹 Khai báo thêm trường này

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.use_search:
            self.search_tool = DuckDuckGoSearchRun()

    @property
    def input_keys(self):
        return ["verification_questions"]

    @property
    def output_keys(self):
        return [self.output_key]

    def _call(self, inputs, run_manager=None):
        verification_questions_text = inputs.get("verification_questions", "")
        if not verification_questions_text:
            return {self.output_key: "No verification questions provided."}

        verification_questions = self._parse_questions(verification_questions_text)
        verification_answers = []

        for q in verification_questions:
            answer = self._execute_single_verification(q)
            verification_answers.append({"question": q, "answer": answer})
            time.sleep(0.5)

        return {self.output_key: json.dumps(verification_answers, ensure_ascii=False, indent=2)}

    # -------------------- #
    # 🔹 Các hàm hỗ trợ
    # -------------------- #

    def _parse_questions(self, text: str):
        """Tách danh sách câu hỏi từ text (từng dòng, hoặc JSON list)."""
        text = text.strip()

        # Trường hợp model trả về JSON list
        if text.startswith("[") and text.endswith("]"):
            try:
                return json.loads(text)
            except Exception:
                pass

        # Trường hợp xuống dòng
        questions = [q.strip("- ").strip() for q in text.split("\n") if q.strip()]
        return questions

    def _execute_single_verification(self, question: str):
        """Trả lời 1 câu hỏi xác minh, có thể qua search hoặc trực tiếp LLM."""
        if self.search_tool:
            try:
                search_result = self.search_tool.run(question)
                # Dùng prompt để tổng hợp kết quả
                search_summary_prompt = f"Question: {question}\n\nSearch Results:\n{search_result}\n\nProvide a concise factual answer:"
                answer = self.llm.invoke(search_summary_prompt)
                #return answer.strip()
                # ✅ Đảm bảo lấy đúng text
                if hasattr(answer, "content"):
                    answer_text = answer.content
                elif isinstance(answer, str):
                    answer_text = answer
                else:
                    answer_text = str(answer)

                print("\n--- 🔍 Verification Step ---")
                print(f"🧠 Question: {question.strip()}")

                print(f"🌐 Search Result (DuckDuckGo): {search_result[:1250].strip()}...")
                print(f"💬 Answer: {answer_text.strip()}")
                print("-----------------------------")
                return answer_text.strip()
            except Exception as e:
                return f"Search failed: {e}"

        # Không có search tool → fallback LLM
        prompt_text = self.prompt.format(verification_questions=question)
        answer = self.llm.invoke(prompt_text)

        # ✅ Đảm bảo lấy đúng text
        if hasattr(answer, "content"):
            answer_text = answer.content
        elif isinstance(answer, str):
            answer_text = answer
        else:
            answer_text = str(answer)

        print("\n--- 🔍 Verification Step ---")
        print(f"🧠 Question: {question.strip()}")
        print(f"💬 Answer: {answer_text.strip()}")
        print("-----------------------------")
        return answer_text.strip()
