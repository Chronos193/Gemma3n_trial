from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.language_models import BaseChatModel
from gemma3n_trial.schema import DishName


class LLM_Agent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=DishName)

        # Inject format instructions into prompt template
        raw_prompt = PromptTemplate.from_template(
            "Extract the name of the dish from the user query below.\n"
            "Return ONLY valid JSON in the format described.\n"
            "Do not include any extra explanation, markdown, or text.\n\n"
            "{format_instructions}\n\n"
            "Query: {user_query}"
        )

        # Fill in format_instructions using .partial
        self.prompt = raw_prompt.partial(
            format_instructions=self.parser.get_format_instructions()
        )

        # Compose full chain
        self.chain = self.prompt | self.llm | self.parser

    def invoke(self, input: dict) -> DishName:
        # Example input: {"user_query": "How to make butter chicken?"}
        return self.chain.invoke(input)

    async def ainvoke(self, input: dict) -> DishName:
        return await self.chain.ainvoke(input)
