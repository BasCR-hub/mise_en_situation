import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from switcher_class import ClfSwitcher

df = pd.read_csv("comments_train.csv")
X = df['comment']
y = df['sentiment'].map({"Positive":1,"Negative":0})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', ClfSwitcher()),
    ])

parameters = [
    {
        'clf__estimator': [RandomForestClassifier()],
        'tfidf__stop_words': ['french', None],
        'tfidf__ngram_range' : [(1,1),(1,2)],
        'tfidf__max_df': (0.25, 0.5, 0.75, 1.0),
        'clf__estimator__n_estimators': [100,300],
        'clf__estimator__max_depth': [4,6],
        'clf__estimator__min_samples_leaf': [3,5,10],
    },
    {
        'clf__estimator': [LogisticRegression()],
        'tfidf__stop_words': ['french', None],
        'tfidf__ngram_range' : [(1,1),(1,2)],
        'tfidf__max_df': (0.25, 0.5, 0.75, 1.0),
        'clf__penalty': ['l1','l2'],
        'clf__C' : [1,10,100]
    },
    {
        'clf__estimator': [GaussianNB()],
        'tfidf__stop_words': ['french', None],
        'tfidf__ngram_range' : [(1,1),(1,2)],
        'tfidf__max_df': (0.25, 0.5, 0.75, 1.0),
        'clf__var_smoothing': (1e-5, 1e-3, 1e-1),
    },

]

gscv = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=3)
gscv.fit(X_train, y_train)
print(gscv.best_estimator_)