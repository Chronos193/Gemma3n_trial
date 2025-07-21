from langgraph.graph import StateGraph
from gemma3n_trial.schema import RecipeSearchResult, CookingState, DetailedRecipe
from gemma3n_trial.agents import RecipeAgent
import os
from dotenv import load_dotenv
from typing import TypedDict, Optional

load_dotenv()

# --------------------------
# 🧩 0. Typed LangGraph State
# --------------------------
class GraphState(TypedDict, total=False):
    recipe_options: list[RecipeSearchResult]
    selected_recipe: RecipeSearchResult
    detailed_recipe: DetailedRecipe

# --------------------------
# 🍳 1. Initial Input State
# --------------------------
initial_state: GraphState = {
    "selected_recipe": RecipeSearchResult(id=636488, title="Butter Chicken"),
}

# --------------------------
# 🤖 2. Agent Initialization
# --------------------------
agent = RecipeAgent(api_key=os.getenv("SPOONACULAR_API_KEY"))

# --------------------------
# 🧠 3. LangGraph Setup
# --------------------------
graph = StateGraph(GraphState)
graph.add_node("recipe", agent.invoke)
graph.set_entry_point("recipe")
graph.set_finish_point("recipe")  # ✅ Needed for flow completion
cooking_graph = graph.compile()

# --------------------------
# 🚀 4. Run the Graph
# --------------------------
result: GraphState = cooking_graph.invoke(initial_state)

# --------------------------
# 📦 5. Show Result State
# --------------------------
print("=== State After Agent ===")
for k, v in result.items():
    print(f"{k}: {v}")

# --------------------------
# 📝 6. Pretty Print Recipe
# --------------------------
print("\n=== Detailed Recipe ===")
try:
    detailed = result["detailed_recipe"]
    if isinstance(detailed, DetailedRecipe):
        recipe = detailed
    else:
        recipe = DetailedRecipe(**detailed)
    print(recipe.model_dump_json(indent=2))
except KeyError:
    print("❌ No detailed recipe found in state.")
except Exception as e:
    print(f"⚠️ Error creating DetailedRecipe model: {e}")
