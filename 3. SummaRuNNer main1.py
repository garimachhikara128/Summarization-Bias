import pandas as pd

DATASET = 'US_Election'

data = pd.read_csv('Required_Data/' + DATASET + '/New_Input.txt', sep = '<\|\|>', header=None, engine = 'python')
data.columns = ['user_id', 'user_name', 'tweet_id', 'type', 'text']

file_subset = open('Required_Data/' + DATASET + '/Subset_Input.txt', 'w+')

for sentence in data['text'] :
    file_subset.write(sentence + "\n") 

file_subset.close()