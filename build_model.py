import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import nltk
import os
import sys
import unicodedata
from sklearn.externals import joblib
from sklearn import model_selection, preprocessing, metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import decomposition, ensemble


from nltk.tokenize.toktok import ToktokTokenizer
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')



# removing the user handler like "@ bloombergnews"

def remove_handler(input_txt):
    pattern = "@\s?[\w]*"
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)        
    return input_txt


#converting é to e accented characters/letters

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

#Converting each contraction to its expanded, original form helps with text standardization.

CONTRACTION_MAP = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)                                if contraction_mapping.get(match)                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


# Remove http/https url tags

def remove_url(input_txt):
    pattern = 'http[s]?://\S+|pic.twitter.\S+|twitter.\S+'
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = input_txt.replace(i, '', 1)
    return input_txt


# Removing redudant space after hashtag just to make whole hashtag as a single token

def remove_hash_space(input_txt):
    pattern = '#\s+'
    input_txt = re.sub(pattern, '#', input_txt)
    return input_txt
 

def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text


def stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text


def get_target():
	r = re.compile('\w+aap\w+|\w+aap|aap\w+|aap|\w+aam\w+|\w+aam|aam\w+|\w+arvind\w+|\w+arvind|arvind\w+|arvind|\w+kejri\w+|\w+kejri|kejri\w+|kejri|\w+askkejri\w+|\w+askkejri|askkejri\w+|askkejri|akasksmodi|akdelhijansabha|akin\w+|aravindkejiriw|arwind|\w+lokp\w+|\w+lokp|lokp\w+|\w+jhaadu\w+|\w+jhaadu|jhaadu\w+|jhaddu')
	for i in range(0, train_df['tidy_tweet'].count()):
		str = train_df['tidy_tweet'].iloc[i]
		ht = re.findall(r"#(\w+)", str)
		match = list(filter(r.match, ht))
		if not match:
			train_df.at[i, 'target'] = 'others'
		else:
			train_df.at[i, 'target'] = 'aap'
	
	r = re.compile('\w+modi\w+|\w+modi|modi\w+|modi|\w+namo\w+|\w+namo|namo\w+|namo|ach\w+|acch\w+|amitshah|\w+bjp\w+|\w+bjp|bjp\w+|bjp|bjd|askjaitley|\w+chaiwala\w+|\w+chaiwala|chaiwala\w+|chaiwala')
	for i in range(0, train_df['tidy_tweet'].count()):
		if train_df.at[i, 'target'] == 'others':
			str = train_df['tidy_tweet'].iloc[i]
			ht = re.findall(r"#(\w+)", str)
			match = list(filter(r.match, ht))
			if not match:
				pass
			else:
				train_df.at[i, 'target'] = 'bjp'
	
	r = re.compile('\w+congress\w+|\w+congress|congress\w+|congress|gandhi\w+|\w+gandhi|gandhi|\w+rahul\w+|\w+rahul|rahul\w+|rahul|rahulgandhi|pappu|agustawestland|feku\w+|feku|manmohansingh')
	for i in range(0, train_df['tidy_tweet'].count()):
		if train_df.at[i, 'target'] == 'others':
			str = train_df['tidy_tweet'].iloc[i]
			ht = re.findall(r"#(\w+)", str)
			match = list(filter(r.match, ht))
			if not match:
				pass
			else:
				train_df.at[i, 'target'] = 'congress'
	
	r = re.compile('aap\w+|aap|arvind\w+|arvind|kejri\w+|kejri|arwind')
	for i in range(0, train_df['tidy_tweet'].count()):
		if train_df.at[i, 'target'] == 'others':
			str = train_df['tidy_tweet'].iloc[i]
			ht = re.findall(r, str)
			if not ht:
				pass
			else:
				train_df.at[i, 'target'] = 'aap'
	
	r = re.compile('modi|namo|bjp')
	for i in range(0, train_df['tidy_tweet'].count()):
		if train_df.at[i, 'target'] == 'others':
			str = train_df['tidy_tweet'].iloc[i]
			ht = re.findall(r, str)
			if not ht:
				pass
			else:
				train_df.at[i, 'target'] = 'bjp'
	
	r = re.compile('congress|rahul|gandhi|sonia|\binc\b|\bupa\b')
	for i in range(0, train_df['tidy_tweet'].count()):
		if train_df.at[i, 'target'] == 'others':
			str = train_df['tidy_tweet'].iloc[i]
			ht = re.findall(r, str)
			if not ht:
				pass
			else:
				train_df.at[i, 'target'] = 'congress'
	
	

def get_vector(train_x):
	
	# word level tf-idf
	tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'#?\w{1,}', max_features=len(train_x))
	
	return tfidf_vect
	
def fit_model(train_df, tfidf_vect, train_x, test_x):
	tfidf_vect.fit(train_df['tidy_tweet'])
	
	# transform the training and validation data using tfidf vectorizer object
	xtrain_tfidf =  tfidf_vect.transform(train_x)
	xtest_tfidf =  tfidf_vect.transform(test_x)
	
	return xtrain_tfidf, xtest_tfidf
	
	

# Setup training model and make prediction

def train_model(classifier, feature_vector_train, label, feature_vector_test):
    
    # fit the training dataset on the classifier
    
    classifier.fit(feature_vector_train, label)
    joblib.dump(classifier,'clf_model.pkl')
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_test)
    
    return metrics.accuracy_score(predictions, test_y)

# RF on Count Vectors

train_df = pd.read_csv('data.csv')
train_df['tidy_tweet'] = np.vectorize(remove_handler)(train_df['tweet'])
train_df['tidy_tweet'] = np.vectorize(remove_accented_chars)(train_df['tidy_tweet'])
train_df['tidy_tweet'] = np.vectorize(expand_contractions)(train_df['tidy_tweet'])
train_df['tidy_tweet'] = np.vectorize(remove_url)(train_df['tidy_tweet'])
train_df['tidy_tweet'] = np.vectorize(remove_hash_space)(train_df['tidy_tweet'])
train_df['tidy_tweet'] = train_df['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")
train_df['tidy_tweet'] = np.vectorize(remove_stopwords)(train_df['tidy_tweet'])
train_df['tidy_tweet'] = np.vectorize(stemmer)(train_df['tidy_tweet'])
get_target()

train_df.drop(train_df[train_df.target == 'others'].index[:-6000], inplace=True)	

train_df.drop(['favorites', 'user_id', 'tweet_id', 'retweet', 'tweet', 'date'], axis=1, inplace=True)


# split the dataset into training and validation datasets 
train_x, test_x, train_y, test_y = model_selection.train_test_split(train_df['tidy_tweet'], train_df['target'], test_size=0.20, random_state=100, shuffle=True) # 80% Train and 20% Test split


# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
test_y = encoder.fit_transform(test_y)

tfidf_vect = get_vector(train_x)

xtrain_tfidf, xtest_tfidf = fit_model(train_df, tfidf_vect, train_x, test_x)

if __name__ == "__main__":
	
	accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf, train_y, xtest_tfidf)
	print("RF, WordLevel TF-IDF: ", accuracy)
	print()
	print("Saved model as clf_model.pkl in current working directory")
	
	



	
		
