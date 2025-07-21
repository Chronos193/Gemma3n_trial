from typing import TypedDict
from langchain_core.runnables import Runnable
from gemma3n_trial.schema import CookingState

class InterfaceAgent(Runnable):
    def __init__(self, user_choice: int):
        self.user_choice = user_choice  # Simulated input (1-5)

    def invoke(self, state: CookingState, config=None) -> CookingState:
        recipes = state["recipe_options"]

        if not 1 <= self.user_choice <= len(recipes):
            raise ValueError(f"Invalid choice {self.user_choice}, must be 1-{len(recipes)}")

        selected = recipes[self.user_choice - 1]
        return {**state, "selected_recipe": selected}
