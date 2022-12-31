import tweepy
import pandas as pd
import numpy as np

consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''
bearer = ''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

DATASET = 'US_Election'

data = pd.read_csv('Data/' + DATASET + '/Input.txt', sep = '<\|\|>', header=None, engine = 'python')
data.columns =['id', 'type', 'text']
print(data.shape)

active_count = 0 
suspended_count = 0 

new_data = open('Data/' + DATASET + '/New_Input.txt', 'w+')

# columns = ['user_id', 'user_name', 'tweet_id', 'type', 'text']
unique_users = {} 
for tweet_id in data['id'] :

    try :
        
        tweet = api.get_status(str(tweet_id))

        new_data_frame = data[data['id'] == tweet_id]

        name = str(tweet.user._json['name'])
        new_data.write(str(tweet.user._json['id']) + "<||>" + 
                        str(name) + "<||>" + 
                        str(tweet_id) + "<||>" + 
                        str(data[data['id'] == tweet_id]['type'].to_list()[0]) + "<||>" + 
                        str(data[data['id'] == tweet_id]['text'].to_list()[0]) + "\n")
        
        if (name in unique_users):
            unique_users[name] += 1
        else:
            unique_users[name] = 1

        active_count += 1

    except :
        print("Account Suspended for " , tweet_id)
        suspended_count += 1

print("No. of active accounts " , active_count)
print("No. of suspended accounts " , suspended_count)
print(unique_users)

new_data.close()
