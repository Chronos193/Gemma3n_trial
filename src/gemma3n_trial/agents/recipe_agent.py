import httpx
from gemma3n_trial.schema import DetailedRecipe, CookingState, RecipeSearchResult


class RecipeAgent:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = "https://api.spoonacular.com/recipes/{id}/information"

    def invoke(self, state: CookingState) -> CookingState:
        # Handle both Pydantic model and dict for selected_recipe
        selected_raw = state.get("selected_recipe")
        if isinstance(selected_raw, RecipeSearchResult):
            selected = selected_raw
        else:
            selected = RecipeSearchResult(**selected_raw)

        url = self.endpoint.format(id=selected.id)
        params = {"apiKey": self.api_key}

        print(f"Fetching recipe info for ID: {selected.id}")
        print(f"URL: {url}")

        response = httpx.get(url, params=params)
        print(f"Response Status Code: {response.status_code}")

        response.raise_for_status()
        data = response.json()

        print("Response JSON keys:", list(data.keys()))
        print("Sample title:", data.get("title"))

        if "title" not in data or "id" not in data:
            print("⚠️ Incomplete data received. Skipping...")
            return state

        ingredients = [item["original"] for item in data.get("extendedIngredients", [])]

        detailed_recipe = DetailedRecipe(
            id=data["id"],
            title=data["title"],
            summary=data.get("summary"),
            instructions=data.get("instructions"),
            readyInMinutes=data.get("readyInMinutes"),
            servings=data.get("servings"),
            ingredients=ingredients
        )

        # Convert the detailed_recipe to a dict for LangGraph compatibility
        new_state: CookingState = {
            **state,
            "detailed_recipe": detailed_recipe.model_dump()
        }

        print("Returning state with keys:", list(new_state.keys()))
        return new_state
