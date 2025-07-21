from pydantic import BaseModel
from typing import List


class RecipeSearchResult(BaseModel):
    id: int
    title: str


class RecipeSearchResults(BaseModel):
    results: List[RecipeSearchResult]
