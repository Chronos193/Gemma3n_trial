from pydantic import BaseModel
from typing import Optional, List


class DetailedRecipe(BaseModel):
    id: int
    title: str
    summary: Optional[str]
    instructions: Optional[str]
    readyInMinutes: Optional[int]
    servings: Optional[int]
    ingredients: Optional[List[str]] = []  # Will extract from extendedIngredients
