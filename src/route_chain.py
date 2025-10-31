# route_chain.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda


class RouteCOVEChain:
    """
    Router that decides whether a question requires factual verification or reasoning,
    and constructs the corresponding CoVe (Chain of Verification) sequence.
    """

    def __init__(self, question: str, route_llm, chain_llm, show_intermediate_steps=False):
        self.question = question
        self.route_llm = route_llm
        self.chain_llm = chain_llm
        self.show_steps = show_intermediate_steps

    def __call__(self):
        router_prompt = ChatPromptTemplate.from_template(
            "You are a CoVe router. "
            "Classify the following question:\n\n"
            "\"{question}\"\n\n"
            "Return only one word: `fact` (for factual verification) or `reasoning` (for reasoning)."
        )

        router_chain = router_prompt | self.route_llm | StrOutputParser()
        route = router_chain.invoke({"question": self.question}).strip().lower()

        if self.show_steps:
            print(f"[Router decision]: {route}")

        if "fact" in route:
            return self.build_fact_chain()
        else:
            return self.build_reasoning_chain()

    def build_fact_chain(self):
        """Build the factual verification chain with concise single-sentence output."""
        prompt = ChatPromptTemplate.from_template(
            "Question: {original_question}\n"
            "Give a short, factual, and accurate answer in one sentence. "
            "Avoid unnecessary details and explanations. "
            "Do not use line breaks or bullet points."
        )
        return prompt | self.chain_llm | StrOutputParser() | RunnableLambda(lambda x: {"final_answer": x.strip().replace('\n', ' ')})

    def build_reasoning_chain(self):
        """Build the reasoning chain with concise final conclusion."""
        prompt = ChatPromptTemplate.from_template(
            "Question: {original_question}\n"
            "Provide a concise, logically reasoned answer in one or two sentences. "
            "Avoid step-by-step reasoning or extra explanations. "
            "Keep the answer fluent and on a single line."
        )
        return prompt | self.chain_llm | StrOutputParser() | RunnableLambda(lambda x: {"final_answer": x.strip().replace('\n', ' ')})
