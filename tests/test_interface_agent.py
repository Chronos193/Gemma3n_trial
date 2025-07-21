from langgraph.graph import StateGraph
from gemma3n_trial.agents import InterfaceAgent
from gemma3n_trial.schema import RecipeSearchResult
from typing import TypedDict
from gemma3n_trial.schema import CookingState

# Sample recipe options
dummy_recipes = [
    RecipeSearchResult(id=1, title="Butter Chicken"),
    RecipeSearchResult(id=2, title="Chicken Tikka"),
    RecipeSearchResult(id=3, title="Paneer Butter Masala"),
    RecipeSearchResult(id=7, title="Chicken Korma"),
    RecipeSearchResult(id=5, title="Biryani"),
]

# Simulated user input: selects option 3 (Paneer Butter Masala)
interface_agent = InterfaceAgent(user_choice=4)

# Build graph
builder = StateGraph(CookingState)
builder.add_node("interface", interface_agent)
builder.set_entry_point("interface")
graph = builder.compile()

# Initial state
initial_state = {"recipe_options": dummy_recipes}

# Run test
result = graph.invoke(initial_state)
print("Selected Recipe:", result["selected_recipe"])
