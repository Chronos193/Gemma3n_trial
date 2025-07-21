from pydantic import BaseModel
from typing import Optional
from gemma3n_trial.schema import RecipeSearchResults
from gemma3n_trial.agents import SearchAgent
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
import os
from dotenv import load_dotenv
load_dotenv()

class CookingState(BaseModel):
    dish_name: str
    search_results: Optional[RecipeSearchResults] = None

def search_node(search_agent: SearchAgent):
    def _invoke(state: CookingState) -> dict:
        # Access as attribute, not dict
        dish_name = state.dish_name
        results = search_agent.invoke(dish_name)

        # Return updated dict
        return {
            "dish_name": dish_name,
            "search_results": results
        }
    return _invoke


# Spoonacular API key
API_KEY = os.getenv("SPOONACULAR_API_KEY")

# Init agent
search_agent = SearchAgent(api_key=API_KEY)

# LangGraph node
search_node_fn = search_node(search_agent)
search_node_wrapped = RunnableLambda(search_node_fn)

# Build graph
graph = StateGraph(CookingState)
graph.add_node("search", search_node_wrapped)
graph.set_entry_point("search")
graph.set_finish_point("search")
cooking_graph = graph.compile()

# Run it
state_input = {"dish_name": "creamy butter chicken"}
result = cooking_graph.invoke(state_input)

# Output
print(result["search_results"])
