import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train['keyword'] = train['keyword'].astype('str')
train['location'] = train['location'].fillna('none given')
train['has_link'] = train['text'].apply(lambda x: True if 'http' in x.lower() else False)
train['has_keyword'] = train['keyword'].apply(lambda x:True if x.isalnum() else False)
train['keyword'] = train['keyword'].apply(lambda x: x if x.isalnum() else 'none')
train = train.drop('id', axis=1)
y = train['target']
train = train.drop('target', axis=1)
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.33, random_state=42)

X_train_encoded = OrdinalEncoder().fit_transform(X=X_train)
X_train_encoded = pd.DataFrame(data=X_train_encoded, columns=X_train.columns)

X_test_encoded = OrdinalEncoder().fit_transform(X=X_test)
X_test_encoded = pd.DataFrame(data=X_test_encoded, columns=X_train.columns)

cv = CountVectorizer()
X = cv.fit_transform(X_train)

clf = RandomForestClassifier()
clf.fit(X_train_encoded,y_train)
print(clf.score(X_test_encoded,y_test))
y_pred = clf.predict(X_test_encoded)
print(classification_report(y_test, y_pred))