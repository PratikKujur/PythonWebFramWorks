from sklearn.base import TransformerMixin
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
import string
from sklearn.metrics import classification_report

class DenseTransformer(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.toarray()  

def sparate_features(dataset_path):
    df = pd.read_csv(dataset_path)
    X, y = df['Text'], df['Language']
    return X, y   

def remove_punc(text):
    for char in string.punctuation:
        text = text.replace(char, '')
    return text

def labelEncoding(y):
    le = LabelEncoder()
    y = le.fit_transform(y)
    return y, le

def split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    dataset_path = "Language Detection.csv"

    X, Y = sparate_features(dataset_path)
    X = X.apply(remove_punc)
    y, label_encoder = labelEncoding(Y)
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    cv = CountVectorizer()
    model = GaussianNB()

   
    pipe = Pipeline([
        ('vectorizer', cv),
        ('to_dense', DenseTransformer()),  
        ('naive_bayes', model)
    ])

    
    pipe.fit(X_train, y_train)

    target_names = label_encoder.inverse_transform(range(len(label_encoder.classes_)))

    
    y_pred = pipe.predict(X_test)
    print(classification_report(y_true=y_test, y_pred=y_pred, target_names=target_names))

  
    with open('trained_pipeline-0.1.0.pkl','wb') as f:
        pickle.dump(pipe, f)