import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import basilica
from scipy import spatial
from dotenv import load_dotenv
import os
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

load_dotenv()

basilica_api = os.getenv("basilica_api_key")
c = basilica.Connection(basilica_api)
train = pd.read_csv('tweets_with_embedding.csv')

y = pd.read_csv('train.csv')['target']

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

train['keyword'] = train['keyword'].fillna('none')
train['embeddings'] = train['embeddings'].apply(lambda x: x.split(','))
train['embeddings_sum'] = train['embeddings'].apply(lambda x: sum_func(x))
train['embeddings_avg'] = train['embeddings'].apply(lambda x: avg(x))
train['capitals'] = train['text'].apply(lambda x: sum(1 for char in x if char.isupper()))
train['ellipsis'] = train['text'].apply(lambda x: True if '...' in x else False)
train['all_caps'] = train['text'].apply(lambda x: True if x.isupper() else False)
train['all_lowercase'] = train['text'].apply(lambda x: True if x.islower() else False)
train['crash'] = train['text'].apply(lambda x: True if 'crash' in x.lower() else False)
train['quake'] = train['text'].apply(lambda x: True if 'quake' in x.lower() else False)

train = train.drop('embeddings', axis=1)

print(train.head().T)
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.33, random_state=42)
X_train_encoded = OrdinalEncoder().fit_transform(X=X_train)
X_train_encoded = pd.DataFrame(data=X_train_encoded, columns=X_train.columns)

X_test_encoded = OrdinalEncoder().fit_transform(X=X_test)
X_test_encoded = pd.DataFrame(data=X_test_encoded, columns=X_train.columns)

print('encoded dataframes')
# clf = RandomForestClassifier(n_estimators=1200, min_samples_split=2, min_samples_leaf=2, max_features='sqrt',
#                             max_depth=20, bootstrap=True)
print('fitting model...')
clf = RandomForestClassifier(n_estimators=1000, min_samples_split=2, max_features = 'sqrt', max_depth=60, bootstrap=True)
clf.fit(X_train_encoded, y_train)
print('classifier fitted')
print(clf.score(X_test_encoded, y_test))
y_pred = clf.predict(X_test_encoded)
print(classification_report(y_test, y_pred))


