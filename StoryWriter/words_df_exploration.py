import pandas as pd
import numpy as np


words_df = pd.read_csv(r'C:\Users\caino\LambdaSchool\SideProjects\StoryWriter\potter_df.csv', header=None, low_memory=False)

sentences_df = pd.read_csv(r'C:\Users\caino\LambdaSchool\SideProjects\StoryWriter\sentences.csv', header=None, low_memory=False)

new_df = pd.DataFrame()

sentences = []

for column in sentences_df.columns:
    sentences.append(sentences_df[column].apply(lambda x : x if x is not np.NaN else 'skip'))

print(sentences)