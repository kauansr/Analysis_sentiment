# Sentiment Analysis Project

This is a deep learning project that performs sentiment analysis on Twitter comments using the **Twitter Sentiment Analysis Dataset**. The model is trained to predict whether the sentiment of a given text is **positive**, **negative**, or **neutral**.

## Key Features
- **Text Preprocessing:** Data cleaning, tokenization, lemmatization, and handling of stopwords.
- **Deep Learning Model:** A neural network built using TensorFlow/Keras for sentiment classification.
- **Visualization:** Detailed analysis and visualization of model performance using **Seaborn** and **Matplotlib**.
- **Evaluation:** Evaluation of model accuracy.

## Technologies Used
- **Python**: The programming language used for data manipulation, model building, and evaluation.
- **FastAPI**: Back-end web framework used to build the back-end API.
- **TensorFlow**: A deep learning framework used to build and train the sentiment analysis model.
- **Keras**: High-level neural networks API used for building the deep learning model.
- **spaCy**: NLP library for tokenization, lemmatization, and text preprocessing.
- **Pandas**: Data manipulation and analysis library for handling datasets.
- **NumPy**: Library for numerical operations.
- **Seaborn** & **Matplotlib**: Data visualization libraries for plotting graphs and model performance.
- **regex** & **string**: Python libraries for text cleaning and preprocessing.
- **JavaScript**: JavaScript for the frontend.
- **React.js**: Front-end framework used to build the user interface.

## Installation for Windows

Follow the steps below to set up the project.

1. Clone the repository:
    ```bash
    git clone 

    cd Sentiment_analysis
    ```

2. Backend (FastAPI)
    ```
    python -m venv venv

    venv\Scripts\activate

    pip install -r requirements.txt

    cd app

    uvicorn main:app --reload
    ```

3. Frontend (Reactjs)
    ```
    cd frontend

    npm install

    npm start
    ```