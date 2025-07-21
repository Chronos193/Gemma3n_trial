from typing import TypedDict, Optional
from gemma3n_trial.schema import RecipeSearchResult
class CookingState(TypedDict):
    recipe_options: list[RecipeSearchResult]
    selected_recipe: RecipeSearchResult
