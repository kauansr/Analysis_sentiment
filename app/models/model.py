from pydantic import BaseModel
from typing import List


class TextInput(BaseModel):
    """
    Pydantic model to parse input data.

    Attributes:
    - texto (List[str]): List of texts to analyze.
    """
    texto: List[str]


class SentimentResponse(BaseModel):
    """
    Pydantic model to return data.
    
    Attributes:
    - predictions: List[str]: List of the predicted texts.
    """
    predictions: List[str]