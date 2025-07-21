from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from gemma3n_trial.agents.llm_agent import LLM_Agent
from gemma3n_trial.agents.search_agent import SearchAgent
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel
from typing import Optional, List

from gemma3n_trial.schema import DishName, RecipeSearchResult

class PipelineState(BaseModel):
    user_query: str
    dish_name: Optional[str] = None
    recipes: Optional[List[RecipeSearchResult]] = None

# Initialize LLM and API key
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama3-8b-8192")  # Make sure key is injected
spoonacular_api_key = os.getenv("SPOONACULAR_API_KEY")

# Instantiate agents
llm_agent = LLM_Agent(llm)
search_agent = SearchAgent(spoonacular_api_key)

# Define nodes
def extract_dish_name_node(state):
    dish_name_obj = llm_agent.invoke({"user_query": state.user_query})
    return {"dish_name": dish_name_obj.name}

def search_recipes_node(state):
    results_obj = search_agent.invoke(state.dish_name)
    return {"recipes": results_obj.results}

# Build LangGraph pipeline
graph = StateGraph(state_schema=PipelineState)
graph.add_node("extract_dish_name", extract_dish_name_node)
graph.add_node("search_recipes", search_recipes_node)

graph.add_edge("extract_dish_name", "search_recipes")
graph.set_entry_point("extract_dish_name")

# Compile the pipeline
app = graph.compile()

# Example usage
if __name__ == "__main__":
    user_query = "Pasta?"
    result = app.invoke({"user_query": user_query})
    print(result.get("recipes", "No recipes found"))
