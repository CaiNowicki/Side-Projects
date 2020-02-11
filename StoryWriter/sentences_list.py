import pandas as pd
import numpy as np

folder = r'C:\Users\caino\LambdaSchool\SideProjects\StoryWriter\Gutenberg\\'

file = 'Charles Dickens___Oliver Twist.txt'

file = folder + file
file = open(file, 'r')
lines = file.readlines()
file.close()
lines2 = [x.replace('\n', '') for x in lines]

sentences_df = pd.DataFrame(lines2, columns = ['Lines'])
sentences_df['Lines'] = sentences_df['Lines'].apply(lambda x : str(x))
sentences_df['Lines'] = sentences_df['Lines'].apply(lambda x : np.NaN if x == '' else x)
sentences_df = sentences_df.dropna()
sentences_df['Lines'] = sentences_df['Lines'].apply(lambda x : np.NaN if x.isupper() else x)
sentences_df = sentences_df.dropna()
sentences_df['Lines'] = sentences_df['Lines'].apply(lambda x : np.NaN if x.isnumeric() else x)
sentences_df = sentences_df.dropna()
sentences_df['Lines'] = sentences_df['Lines'].apply(lambda x : np.NaN if x.isspace() else x)
sentences_df = sentences_df.dropna()

sentences_df.to_csv('sentences.csv')