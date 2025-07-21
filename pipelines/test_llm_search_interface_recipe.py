from langgraph.graph import StateGraph
from gemma3n_trial.agents.llm_agent import LLM_Agent
from gemma3n_trial.agents.search_agent import SearchAgent
from gemma3n_trial.agents.interface_agent import InterfaceAgent
from gemma3n_trial.agents.recipe_agent import RecipeAgent
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
    detailed_recipe: Optional[Dict[str, Any]] = None  # Store as dict for compatibility

# Initialize LLM and API key
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama3-8b-8192")
spoonacular_api_key = os.getenv("SPOONACULAR_API_KEY")

# Instantiate agents
llm_agent = LLM_Agent(llm)
search_agent = SearchAgent(spoonacular_api_key)
recipe_agent = RecipeAgent(spoonacular_api_key)

def extract_dish_name_node(state: PipelineState) -> dict:
    dish_name_obj = llm_agent.invoke({"user_query": state.user_query})
    return {"dish_name": dish_name_obj.name}

def search_recipes_node(state: PipelineState) -> dict:
    results_obj = search_agent.invoke(state.dish_name)
    print("Recipes found:")
    for idx, recipe in enumerate(results_obj.results, 1):
        print(f"{idx}: {recipe.title}")
    return {"recipes": results_obj.results}

def select_recipe_node(state: PipelineState) -> dict:
    while True:
        try:
            user_choice = int(input(f"Select a recipe (1-{len(state.recipes)}): "))
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
    # Prepare CookingState dict for RecipeAgent
    cooking_state = {
        "recipe_options": state.recipes,
        "selected_recipe": state.selected_recipe
    }
    new_state = recipe_agent.invoke(cooking_state)
    detailed_recipe = new_state.get("detailed_recipe")
    return {"detailed_recipe": detailed_recipe}

# Build LangGraph pipeline
graph = StateGraph(state_schema=PipelineState)
graph.add_node("extract_dish_name", extract_dish_name_node)
graph.add_node("search_recipes", search_recipes_node)
graph.add_node("select_recipe", select_recipe_node)
graph.add_node("fetch_detailed_recipe", fetch_detailed_recipe_node)

graph.add_edge("extract_dish_name", "search_recipes")
graph.add_edge("search_recipes", "select_recipe")
graph.add_edge("select_recipe", "fetch_detailed_recipe")
graph.set_entry_point("extract_dish_name")

# Compile the pipeline
app = graph.compile()

if __name__ == "__main__":
    user_query = input("Enter your cooking query: ")
    result = app.invoke({"user_query": user_query})
    selected = result.get("selected_recipe")
    print(f"\nSelected recipe: {selected.title if selected else 'None'}")
    detailed = result.get("detailed_recipe")
    if detailed:
        print("\nDetailed Recipe:")
        print(f"Title: {detailed.get('title')}")
        print(f"Summary: {detailed.get('summary')}")
        print(f"Instructions: {detailed.get('instructions')}")
        print(f"Ingredients: {', '.join(detailed.get('ingredients', []))}")
        print(f"Ready in: {detailed.get('readyInMinutes')} minutes")
        print(f"Servings: {detailed.get('servings')}")
    else:
        print("No detailed recipe found.")