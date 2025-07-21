from langgraph.graph import StateGraph
from gemma3n_trial.agents.llm_agent import LLM_Agent
from gemma3n_trial.agents.search_agent import SearchAgent
from gemma3n_trial.agents.interface_agent import InterfaceAgent
from gemma3n_trial.agents.recipe_agent import RecipeAgent
from gemma3n_trial.agents import CookingGraphAgent, AgentState
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from gemma3n_trial.schema import DishName, RecipeSearchResult, DetailedRecipe

class PipelineState(BaseModel):
    user_query: str
    dish_name: Optional[str] = None
    recipes: Optional[List[RecipeSearchResult]] = None
    selected_recipe: Optional[RecipeSearchResult] = None
    detailed_recipe: Optional[Dict[str, Any]] = None

# Initialize LLM and API key
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama3-8b-8192")
spoonacular_api_key = os.getenv("SPOONACULAR_API_KEY")

llm_agent = LLM_Agent(llm)
search_agent = SearchAgent(spoonacular_api_key)
recipe_agent = RecipeAgent(spoonacular_api_key)
cooking_graph_agent = CookingGraphAgent(api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-8b-8192")

def extract_dish_name_node(state: PipelineState) -> dict:
    dish_name_obj = llm_agent.invoke({"user_query": state.user_query})
    return {"dish_name": dish_name_obj.name}

def search_recipes_node(state: PipelineState) -> dict:
    results_obj = search_agent.invoke(state.dish_name)
    print("\nRecipes found:")
    for idx, recipe in enumerate(results_obj.results, 1):
        print(f"{idx}: {recipe.title}")
    return {"recipes": results_obj.results}

def select_recipe_node(state: PipelineState) -> dict:
    while True:
        try:
            user_choice = int(input(f"\nSelect a recipe (1-{len(state.recipes)}): "))
            if 1 <= user_choice <= len(state.recipes):
                break
            print("Invalid choice. Try again.")
        except Exception:
            print("Invalid input. Enter a number.")
    interface_agent = InterfaceAgent(user_choice)
    cooking_state = {
        "recipe_options": state.recipes,
        "selected_recipe": None
    }
    selected = interface_agent.invoke(cooking_state)
    return {"selected_recipe": selected["selected_recipe"]}

def fetch_detailed_recipe_node(state: PipelineState) -> dict:
    cooking_state = {
        "recipe_options": state.recipes,
        "selected_recipe": state.selected_recipe
    }
    new_state = recipe_agent.invoke(cooking_state)
    detailed_recipe = new_state.get("detailed_recipe")
    return {"detailed_recipe": detailed_recipe}

graph = StateGraph(state_schema=PipelineState)
graph.add_node("extract_dish_name", extract_dish_name_node)
graph.add_node("search_recipes", search_recipes_node)
graph.add_node("select_recipe", select_recipe_node)
graph.add_node("fetch_detailed_recipe", fetch_detailed_recipe_node)

graph.add_edge("extract_dish_name", "search_recipes")
graph.add_edge("search_recipes", "select_recipe")
graph.add_edge("select_recipe", "fetch_detailed_recipe")
graph.set_entry_point("extract_dish_name")

app = graph.compile()

def format_recipe_for_print(detailed: Dict[str, Any]):
    print("\n--- Detailed Recipe ---")
    print(f"Title: {detailed.get('title')}")
    print(f"Summary: {detailed.get('summary')}")
    print(f"Instructions: {detailed.get('instructions')}")
    print(f"Ingredients: {', '.join(detailed.get('ingredients', []))}")
    print(f"Ready in: {detailed.get('readyInMinutes')} minutes")
    print(f"Servings: {detailed.get('servings')}")
    print("--- End of Recipe ---\n")


if __name__ == "__main__":
    print("👩‍🍳 Welcome to the Cooking Assistant CLI! 🍽️\n")
    user_query = input("📝 What would you like to cook today? ").strip()

    if not user_query:
        print("⚠️ You need to enter something to begin!")
        exit()

    print("\n⏳ Thinking... Finding the best options for you!\n")
    result = app.invoke({"user_query": user_query})

    selected = result.get("selected_recipe")
    if not selected:
        print("❌ No recipe selected. Please try again with a different query.")
        exit()

    print(f"\n✅ You selected: **{selected.title}** 🎉")

    detailed = result.get("detailed_recipe")
    if not detailed:
        print("⚠️ Couldn't fetch the detailed recipe. Please try again later.")
        exit()

    # Print nicely formatted recipe
    def format_recipe_for_print(detailed: Dict[str, Any]):
        print("\n📖 Here's your detailed recipe:\n")
        print(f"🍲 Title: {detailed.get('title')}")
        print(f"📝 Summary: {detailed.get('summary')}")
        print(f"🧑‍🍳 Instructions:\n{detailed.get('instructions')}\n")
        print(f"🧂 Ingredients: {', '.join(detailed.get('ingredients', []))}")
        print(f"⏱️ Ready in: {detailed.get('readyInMinutes')} minutes")
        print(f"👥 Servings: {detailed.get('servings')}")
        print("\n🍽️ Happy Cooking! 🎉\n")

    #format_recipe_for_print(detailed)

    # Follow-up Q&A loop
    detailed_recipe_obj = DetailedRecipe(**detailed)
    print("🤖 You can now ask me questions about this recipe.")
    print("💬 Type your question or type 'exit' to quit.\n")

    while True:
        followup_input = input("❓ Your question: ").strip()
        if followup_input.lower() == "exit":
            print("👋 Thank you for using the Cooking Assistant. Bon appétit!")
            break
        if not followup_input:
            print("⚠️ Please enter a question or type 'exit' to finish.\n")
            continue

        agent_state = AgentState(
            detailed_recipe=detailed_recipe_obj,
            user_input=followup_input
        )
        agent_state = cooking_graph_agent.invoke(agent_state)
        print(f"\n🤖 Cooking Assistant says: {agent_state.response}\n")
