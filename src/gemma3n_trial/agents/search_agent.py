import httpx
from gemma3n_trial.schema import RecipeSearchResults


class SearchAgent:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = "https://api.spoonacular.com/recipes/complexSearch"

    def invoke(self, dish_name: str) -> RecipeSearchResults:
        params = {
            "query": dish_name,
            "number": 5,
            "apiKey": self.api_key,
        }

        response = httpx.get(self.endpoint, params=params)
        response.raise_for_status()
        data = response.json()

        return RecipeSearchResults(results=data.get("results", []))
