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
    example_disaster_tweet = '#reuters Twelve feared killed in Pakistani air ambulance helicopter crash http://t.co/ShzPyIQok5'
    df['keyword'] = df['keyword'].astype('str')
    df['location'] = df['location'].fillna('none given')
    df['has_link'] = df['text'].apply(lambda x: True if 'http' in x.lower() else False)
    df['has_keyword'] = df['keyword'].apply(lambda x: True if x.isalnum() else False)
    df['keyword'] = df['keyword'].apply(lambda x: x if x.isalnum() else 'none')
    df = df.drop('id', axis=1)
    df['worldwide'] = df['location'].apply(lambda x: True if 'world' in x.lower() else False)
    df['length'] = df['text'].apply(lambda x: len(x))
    df['exclamatory'] = df['text'].apply(lambda x: True if '!' in x.lower() else False)
    df['help'] = df['text'].apply(lambda x: True if 'help' in x.lower() else False)
    print('wrangled data...')
    print('getting sentence embeddings...')
    df['embeddings'] = df['text'].apply(lambda x: c.embed_sentence(x, model='twitter'))
    print('created sentence embeddings...')
    for i in range(len(df)-1):
        distance = spatial.distance.cosine(df['text'].iloc[i], example_disaster_tweet)
        if distance > 0.5:
            df['similar_to_real'].iloc[i] = True
        else:
            df['similar_to_real'].iloc[i] = False

    y = train['target']
    df = train.drop('target', axis=1)
    print('created train and target dataframes...')
    return df, y


train, y = wrangle(train)
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.33, random_state=42)
X_train_encoded = OrdinalEncoder().fit_transform(X=X_train)
X_train_encoded = pd.DataFrame(data=X_train_encoded, columns=X_train.columns)

X_test_encoded = OrdinalEncoder().fit_transform(X=X_test)
X_test_encoded = pd.DataFrame(data=X_test_encoded, columns=X_train.columns)

print('encoded dataframes...')
# clf = RandomForestClassifier(n_estimators=1200, min_samples_split=2, min_samples_leaf=2, max_features='sqrt',
#                             max_depth=20, bootstrap=True)
clf = RandomForestClassifier()
print('fitting classifier...')
clf.fit(X_train_encoded, y_train)
print('classifier fitted...')
print(clf.score(X_test_encoded, y_test))
y_pred = clf.predict(X_test_encoded)
print(classification_report(y_test, y_pred))

# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
# # Random search of parameters, using 3 fold cross validation,
# # search across 100 different combinations, and use all available cores
# clf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2,
#                                 random_state=42, n_jobs = -1)
# # Fit the random search model
# clf_random.fit(X_train_encoded, y_train)
#
# print(clf_random.best_params_)