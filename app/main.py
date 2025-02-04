from dlmodel.model import data_process
from fastapi import FastAPI
from models.model import TextInput

# Initialize FastAPI app
app = FastAPI()


@app.post("/predict")
async def prediction(data: TextInput):
    """
    Predicts the sentiment of input texts by:
    - Preprocessing, tokenizing and predicting the texts

    Args:
    - data (TextInput): The input data containing a list of texts.

    Returns:
    - dict: A dictionary containing the predicted sentiments for each text.
    """

    # A list containing the feelings of the text
    sentiments = data_process(data)

    return {"predictions": sentiments}