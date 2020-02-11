import pandas as pd
import os
folder = r'C:\Users\caino\LambdaSchool\SideProjects\StoryWriter\Gutenberg'
potter_files = [f for f in os.listdir(folder) if 'Beatrix Potter' in f]

def remove_newlines(fname):
    flist = open(fname).readlines()
    return [s.lstrip('\n') for s in flist]

for file in potter_files:
    path = r"C:/Users/caino/LambdaSchool/SideProjects/StoryWriter/Gutenberg/" + file
    path = remove_newlines(path)

words_lists = []
words = []
for file in potter_files:
    path = r"C:/Users/caino/LambdaSchool/SideProjects/StoryWriter/Gutenberg/" + file
    List = open(path).readlines()
    words_lists.append(List)

for wlist in words_lists:
    for word in wlist:
        words.append(word)

print(words)

