import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
clf_pipe = joblib.load('sentiment_pipe.joblib')
df = pd.read_csv("comments_train.csv")

X = df['comment']
y = df['sentiment'].map({"Positive":1,"Negative":0})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
preds = clf_pipe.predict(X_test)

print(roc_auc_score(y_test,preds))