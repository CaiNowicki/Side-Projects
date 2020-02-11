import os

import pandas as pd

folder = r'C:\Users\caino\LambdaSchool\SideProjects\StoryWriter\Gutenberg'
files = [f for f in os.listdir(folder) if f.endswith('.csv')]

sentences = pd.DataFrame()

for file in files:
    path = folder +'\\' + file
    print(path)
    book = pd.read_csv(path, sep='.', error_bad_lines=False, quotechar='"', engine='python')
    sentences = pd.concat([sentences, book])
    sentences.to_csv(r'C:\Users\caino\LambdaSchool\SideProjects\StoryWriter\sentences.csv')

