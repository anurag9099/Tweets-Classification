import pandas as pd
import os
import sys
import glob
import json

print("Formatting JSON to CSV...")
favorites = []
user_id = []
tweet_id = []
retweet = []
tweet = []
date = []


directories = ['Corruption/Bfr elec 1_jan_to_6_apr', 'Corruption/Electn time 7_aprl_to_12_may', 'Corruption/Aftr elctn 13_may_to_31_aug',
           'Development/before 1_jan_to_6_april', 'Development/elec 7_apr_to_12_may', 'Development/after 13_may_to_31_aug',
           'India Economy/Bfr elec 1_jan_to_6_apr', 'India Economy/Electn time 7_aprl_to_12_may', 'India Economy/Aftr elctn 13_may_to_31_aug']


def json2csv(data_obj):
    for key in data_obj:
        favorites.append(key['favorites'])
        user_id.append(key['user_id'])
        tweet_id.append(key['tweet_id'])
        retweet.append(key['retweet'])
        tweet.append(key['tweet'])
        date.append(key['date'])
    df = pd.DataFrame({'favorites': favorites, 'user_id': user_id, 'tweet_id': tweet_id, 'retweet': retweet, 'tweet': tweet, 'date': date})
    df.to_csv('data.csv', encoding='utf-8', index=False)


def get_file_path(directories):  
    root_dir = os.path.abspath('.')
    files = []
    
    for dirs in directories:
        data_dir = os.path.join(root_dir, dirs)
        path = os.path.join(data_dir, '*.json')
        files.extend(glob.glob(path))
    return files


def build_csv(directories):
    for file in get_file_path(directories):
        try:
            with open(file, mode='r') as f:
                data_json = f.read()
                data_obj = json.loads(data_json)
                json2csv(data_obj)
        except IOError as exc:
            if exc.errno != errno.EISDIR:
                raise


build_csv(directories)
print("data.csv dataset has been built and saved to current working directory")
