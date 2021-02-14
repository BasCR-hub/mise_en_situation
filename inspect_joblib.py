import joblib

clf_pipe = joblib.load('sentiment_pipe.joblib')
print(clf_pipe)