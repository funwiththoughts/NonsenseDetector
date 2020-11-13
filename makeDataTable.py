import pandas as pd
from tabulate import tabulate

texts = []
labels = []

for cat in ['sense','nonsense']:
    file = 'data/' + cat + '.txt'
    f = open(file,'r')
    lines = f.readlines()
    for line in lines:
        texts.append(line)
        labels.append(0 if cat == 'sense' else 1)
    f.close()

df = pd.DataFrame(list(zip(texts,labels)),columns=['sentence','class'])

with open('data/data.tsv','w') as f:
    f.write(df.to_csv(sep='\t',index=False))
