from dlmodel.model import data_process
from fastapi import FastAPI
from models.model import TextInput, SentimentResponse
from middlewares.corsmiddle import add_cors_middleware

# Initialize FastAPI app
app = FastAPI()

# Add the middleware
app = add_cors_middleware(app)

@app.post("/predict")
async def prediction(data: TextInput, response_model=SentimentResponse):
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

    return SentimentResponse(predictions=sentiments)