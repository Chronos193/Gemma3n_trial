from typing import List
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.runnables import RunnableSerializable
from gemma3n_trial.schema import DetailedRecipe


class AgentState(BaseModel):
    detailed_recipe: DetailedRecipe
    user_input: str
    response: str = ""


def format_recipe(recipe: DetailedRecipe) -> str:
    return (
        f"Title: {recipe.title}\n"
        f"Summary: {recipe.summary}\n"
        f"Instructions: {recipe.instructions}\n"
        f"Ready in: {recipe.readyInMinutes} minutes\n"
        f"Servings: {recipe.servings}\n"
        f"Ingredients: {', '.join(recipe.ingredients)}"
    )


class CookingGraphAgent:
    def __init__(self, api_key: str, model_name: str = "llama3-8b-8192"):
        self.llm = ChatGroq(
            api_key=api_key,
            model=model_name,
            temperature=0.5,
        )

        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
            output_key="output",
            k=2  # Keep last 2 interactions
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful cooking assistant. Answer user questions about the given recipe."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "Here is the recipe:\n{recipe}\n\n{input}")
        ])

        self.chain: RunnableSerializable = self.prompt | self.llm

    def invoke(self, state: AgentState) -> AgentState:
        user_input = state.user_input.strip()

        # Load memory context
        memory_variables = self.memory.load_memory_variables({"input": user_input})
        chat_history = memory_variables.get("chat_history", [])

        # Prepare input for the chain
        prompt_input = {
            "recipe": format_recipe(state.detailed_recipe),
            "input": user_input,
            "chat_history": chat_history,
        }

        # Invoke the LLM
        try:
            response = self.chain.invoke(prompt_input)
            response_content = (
                response.content.strip()
                if hasattr(response, "content")
                else str(response)
            )
        except Exception:
            response_content = "🤖 Sorry, I couldn't process that right now."

        # Save interaction to memory
        self.memory.save_context(
            {"input": user_input},
            {"output": response_content}
        )

        return AgentState(
            detailed_recipe=state.detailed_recipe,
            user_input=user_input,
            response=response_content
        )
