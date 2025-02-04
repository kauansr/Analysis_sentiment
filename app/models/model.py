from pydantic import BaseModel
from typing import List


class TextInput(BaseModel):
    """
    Pydantic model to parse input data.

    Attributes:
    - texto (List[str]): List of texts to analyze.
    """
    texto: List[str]