import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import basilica
import numpy as np
import random
from scipy import spatial
from dotenv import load_dotenv
import os

load_dotenv()


basilica_api = os.getenv("basilica_api_key")
c = basilica.Connection(basilica_api)
print('connected to basilica')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print('made train/test df')

def wrangle(df):
    example_disaster_tweet = c.embed_sentence('#reuters Twelve feared killed in Pakistani air ambulance helicopter crash http://t.co/ShzPyIQok5')
    df['keyword'] = df['keyword'].astype('str')
    df['location'] = df['location'].fillna('none given')
    df['has_link'] = df['text'].apply(lambda x: True if ('http' in x.lower()) or ('.com' in x.lower()) else False)
    df['has_keyword'] = df['keyword'].apply(lambda x: True if x.lower() is not 'nan' else False)
    df['keyword'] = df['keyword'].apply(lambda x: x if x.lower() is not 'nan' else 'none')
    df['worldwide'] = df['location'].apply(lambda x: True if 'world' in x.lower() else False)
    df['length'] = df['text'].apply(lambda x: len(x))
    df['exclamatory'] = df['text'].apply(lambda x: True if '!' in x.lower() else False)
    df['questioning'] = df['text'].apply(lambda x: True if "?" in x.lower() else False)
    df['help'] = df['text'].apply(lambda x: True if 'help' in x.lower() else False)
    df['hashtag'] = df['text'].apply(lambda x: True if '#' in x else False)
    print('wrangled data')
    print('getting sentence embeddings...')
    df['embeddings'] = df['text'].apply(lambda x: c.embed_sentence(x, model='twitter'))
    print('created sentence embeddings')
    df['similar_to_real'] = False
    for i in range(len(df)-1):
        distance = spatial.distance.cosine(df['embeddings'].iloc[i], example_disaster_tweet)
        if distance > 0.5:
            df['similar_to_real'].iloc[i] = True
    df['keyword'] = df['keyword'].fillna('none')
    #df['embeddings'] = df['embeddings'].apply(lambda x: x.split(','))
    df['embeddings_sum'] = df['embeddings'].apply(lambda x: sum_func(x))
    df['embeddings_avg'] = df['embeddings'].apply(lambda x: avg(x))
    df['capitals'] = df['text'].apply(lambda x: sum(1 for char in x if char.isupper()))
    df['ellipsis'] = df['text'].apply(lambda x: True if '...' in x else False)
    df['all_caps'] = df['text'].apply(lambda x: True if x.isupper() else False)
    df['all_lowercase'] = df['text'].apply(lambda x: True if x.islower() else False)
    df['crash'] = df['text'].apply(lambda x: True if 'crash' in x.lower() else False)
    df['quake'] = df['text'].apply(lambda x: True if 'quake' in x.lower() else False)

    df = df.drop('embeddings', axis=1)
    return df

def sum_func(list_val):
    sum = 0
    for item in list_val:
        item = item.strip('[]')
        item = float(item)
        sum += item
    return sum
def avg(list_val):
    sum = 0
    for item in list_val:
        item = item.strip('[]')
        item = float(item)
        sum += item
    avg = sum / len(list_val)
    return avg


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train = wrangle(train)
test = wrangle(test)

X_train = train.drop('target', axis=1)
y = train['target']
X_train_encoded = OrdinalEncoder().fit_transform(X=X_train)
test_encoded = OrdinalEncoder().fit_transform(test)
X_train_encoded = pd.DataFrame(data=X_train_encoded, columns=X_train.columns)
test_encoded = pd.DataFrame(data=test_encoded, columns=test.columns)

print('encoded dataframes')
print('fitting model...')
clf = RandomForestClassifier(n_estimators=1000, min_samples_split=2, max_features = 'sqrt', max_depth=60, bootstrap=True)
clf.fit(X_train_encoded, y)
print('classifier fitted')

preds = clf.predict(test_encoded)
preds.to_csv('predictions.csv')