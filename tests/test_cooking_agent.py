import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph
from pydantic import BaseModel
from gemma3n_trial.schema import DetailedRecipe
from gemma3n_trial.agents import CookingGraphAgent

# -------------------------
# 1. Load API Key
# -------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# -------------------------
# 2. Agent State Definition
# -------------------------
class AgentState(BaseModel):
    detailed_recipe: DetailedRecipe
    user_input: str
    response: str = ""


# -------------------------
# 3. Create Agent and Graph
# -------------------------
agent = CookingGraphAgent(api_key=GROQ_API_KEY)
graph = StateGraph(AgentState)
graph.add_node("ask_agent", agent.invoke)
graph.set_entry_point("ask_agent")
compiled_graph = graph.compile()


# -------------------------
# 4. Dummy Recipe
# -------------------------
sample_recipe = DetailedRecipe(
    id=123,
    title="Spaghetti Carbonara",
    summary="Classic Italian pasta dish with creamy sauce.",
    instructions="1. Cook pasta. 2. Mix eggs and cheese. 3. Combine with pancetta.",
    readyInMinutes=25,
    servings=2,
    ingredients=["spaghetti", "eggs", "parmesan cheese", "pancetta", "black pepper"]
)


# -------------------------
# 5. Interactive Loop
# -------------------------
print("\n🍝 Welcome to the Cooking Assistant!")
print("Ask anything about the dish 'Spaghetti Carbonara'. Type 'exit' to quit.\n")

while True:
    user_input = input("👤 You: ").strip()
    if not user_input:
        print("⚠️  Please enter a message.\n")
        continue

    if user_input.lower() in ["exit", "quit"]:
        print("👋 Exiting Cooking Assistant. Buon appetito!")
        break

    state = {
        "detailed_recipe": sample_recipe,
        "user_input": user_input,
    }

    result = compiled_graph.invoke(state)
    print(f"🤖 Assistant: {result['response']}\n")
