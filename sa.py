import pandas as pd
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.svm import SVC

df = pd.read_csv("sentiment_analysis.csv")
df.head(5)

df.info()
df.shape


def mapping(x):
    map = {"neutral": 0, "positive": 1, "negative": 2}
    return map.get(x)


df["sentiment"] = df["sentiment"].apply(mapping)

word_count = 0
for content in df["text"]:
    word_count += sum(1 for _ in content.split())


ps = PorterStemmer()


def stemming(content):
    stemmed_content = re.sub("[^a-zA-Z]", " ", content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [
        ps.stem(word)
        for word in stemmed_content
        if not word in stopwords.words("english")
    ]
    stemmed_content = " ".join(stemmed_content)
    return stemmed_content


df["text"] = df["text"].apply(stemming)

total_word_count = 0
for content in df["text"]:
    total_word_count += sum(1 for _ in content.split())
print(total_word_count)


X = df["text"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=50
)

vc = TfidfVectorizer()
X_train = vc.fit_transform(X_train)
X_test = vc.transform(X_test)
print(X_test)
model = SVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)
acc = accuracy_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# clas_report = classification_report(y_test, y_pred)
print(f"Model: SVM")
print(f"Accuracy: {acc:.4f}")
# print(f"F1-Score: {f1:.4f}")
# print(f"Classification Report:\n{clas_report}")


def val_to_category(val):
    category_map = {0: "neutral", 1: "positive", 2: "negative"}
    return category_map.get(val, -1)


def make_predictions(text):
    text = stemming(text)
    text = vc.transform([text])
    val = model.predict(text)
    val = val_to_category(int(val[0]))
    print("sentiment is : ", val)


txt = input("Enter a text")
make_predictions(txt)


model_pkl_file = "sentiment_analysis_model.pkl"
with open(model_pkl_file, "wb") as file:
    pickle.dump(model, file)
