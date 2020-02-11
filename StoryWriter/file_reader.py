import os

folder = r'C:\Users\caino\LambdaSchool\SideProjects\StoryWriter\Gutenberg'
files = [f for f in os.listdir(folder) if f.endswith('.txt')]

lines = []
for file in files:
    path = r'C:/Users/caino/LambdaSchool/SideProjects/StoryWriter/Gutenberg/' + file
    line = path.readline()



