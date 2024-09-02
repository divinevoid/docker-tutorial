from fastapi import FastAPI, HTTPException
from Sentiments import Sentiment
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import uvicorn

nltk.download("stopwords")

app = FastAPI()
model_in = open("sentiment_analysis_model.pkl", "rb")
sa = pickle.load(model_in)

vectorizer = TfidfVectorizer()

ps = PorterStemmer()


def stemming(content):
    stemmed_content = re.sub("[^a-zA-Z]", " ", content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [
        ps.stem(word)
        for word in stemmed_content
        if word not in stopwords.words("english")
    ]
    stemmed_content = " ".join(stemmed_content)
    return stemmed_content


def val_to_category(val):
    category_map = {0: "neutral", 1: "positive", 2: "negative"}
    return category_map.get(val, "unknown")


class TextInput(Sentiment):
    text: str


@app.get("/health")
def read_root():
    return {"message": "Sentiment Analysis API is running!"}


@app.post("/predict")
def predict_sentiment(input_data: TextInput):
    try:
        text = stemming(input_data.text)
        text_vector = vectorizer.transform([text])

        prediction = model.predict(text_vector)
        sentiment = val_to_category(int(prediction[0]))

        return {"text": input_data.text, "sentiment": sentiment}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
