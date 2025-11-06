# src/question_preprocess.py

from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_core.output_parsers import StrOutputParser


class QuestionPreprocessor:
    """
    Tiền xử lý và chuẩn hóa câu hỏi đầu vào.
    Mục tiêu: giảm rủi ro hallucination do câu hỏi mơ hồ, thiếu ngữ cảnh.
    """

    def __init__(self, llm, show_steps=False):
        """
        Args:
            llm: Mô hình LLM (ví dụ ChatOpenAI, OpenAI, Ollama, v.v.)
            show_steps: Nếu True, in ra câu hỏi trước/sau khi chuẩn hóa.
        """
        self.llm = llm
        self.show_steps = show_steps

        # Prompt chính: hướng dẫn mô hình chuẩn hóa câu hỏi
        self.preprocess_prompt = PromptTemplate.from_template(
"""You are a question normalization assistant.

Your task is to rewrite the following question so that it is:
- Clear, specific, and unambiguous.
- Free of hallucination-inducing phrasing.
- Retains the *exact same meaning*.
- Grammatically correct and self-contained.

Return only the improved question, nothing else.

Original question:
"{original_question}"
"""
        )

        self.chain = self.preprocess_prompt | self.llm | StrOutputParser()

    def __call__(self, question: str) -> str:
        """Chuẩn hóa và trả về câu hỏi mới."""
        normalized = self.chain.invoke({"original_question": question}).strip()

        if self.show_steps:
            print("🟡 Original question:", question)
            print("🟢 Normalized question:", normalized)

        return normalized
