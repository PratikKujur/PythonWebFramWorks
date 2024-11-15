import pickle
from pathlib import Path
import string

__version__="0.1.0"

BASE_DIR=Path(__file__).resolve(strict=True).parent


with open("app//model//trained_pipeline-0.1.0.pkl", "rb") as f:
    model = pickle.load(f)

classes = [
    "Arabic",
    "Danish",
    "Dutch",
    "English",
    "French",
    "German",
    "Greek",
    "Hindi",
    "Italian",
    "Kannada",
    "Malayalam",
    "Portugeese",
    "Russian",
    "Spanish",
    "Sweedish",
    "Tamil",
    "Turkish",
]

def remove_punc(text):
    for char in string.punctuation:
        text = text.replace(char, '')
    return text

def predict_pipleline(text):
    text=remove_punc(text)
    text=text.lower()
    pred=model.predict([text])
    return classes[pred[0]]