import build_model
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd
import seaborn as sns
from textblob import TextBlob

# Validating our model

def clean_tweet(tweet):
    tweet = build_model.remove_handler(tweet)
    tweet = build_model.remove_accented_chars(tweet)
    tweet = build_model.expand_contractions(tweet)
    tweet = build_model.remove_url(tweet)
    tweet = re.sub("[^a-zA-Z#]", " ", tweet)
    tweet = build_model.remove_hash_space(tweet)
    tweet = build_model.remove_stopwords(tweet)
    tweet = build_model.stemmer(tweet)
    return tweet


tweet = input("Enter Tweet: ")

norm_tweet = clean_tweet(tweet)

clf = joblib.load('clf_model.pkl')
#tfidf_vect = get_vector.(train_x)
label = clf.predict(build_model.tfidf_vect.transform([norm_tweet]))
if label == 0:
    pred = 'AAP'
elif label == 1:
    pred = 'BJP'
elif label == 2:
    pred = 'Congress'
else:
    pred = 'Others'

df = pd.DataFrame([tweet, pred]).T
df.columns = ['Tweet', 'Party']
print(df)
