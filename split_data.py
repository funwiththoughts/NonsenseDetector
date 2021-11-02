import pandas as pd
from sklearn.model_selection import train_test_split

#    3.1 Create train/validation/test splits

dataFile = pd.read_table('data/data.tsv',sep='\t')

print(dataFile)

nonsenseData = dataFile[dataFile['class'] == 1]
senseData = dataFile[dataFile['class'] == 0]

nonsenseDataTrain, nonsenseDataTest,senseDataTrain, senseDataTest = train_test_split(nonsenseData,senseData,test_size=0.2,random_state=1)
nonsenseDataTrain, nonsenseDataValid, senseDataTrain, senseDataValid = train_test_split(nonsenseDataTrain,senseDataTrain,test_size=0.2,random_state=1)

dataTrain = pd.concat([nonsenseDataTrain,senseDataTrain])
dataTest = pd.concat([nonsenseDataTest,senseDataTest])
dataValid = pd.concat([nonsenseDataValid,senseDataValid])

for Set in [dataTrain,dataTest,dataValid]:
    print("Number of nonsense sentences is: ",Set['class'].isin([1]).sum())
    print("Number of sense sentences is: ",Set['class'].isin([0]).sum())

with open('data/train.tsv','w') as write_tsv:
    write_tsv.write(dataTrain.to_csv(sep='\t', index=False))

with open('data/validation.tsv','w') as write_tsv:
    write_tsv.write(dataValid.to_csv(sep='\t', index=False))

with open('data/test.tsv','w') as write_tsv:
    write_tsv.write(dataTest.to_csv(sep='\t', index=False))

#for number, sentence, label in dataFile:
 #   if label == 1:
  #      subjectiveData['text'][number] = sentence
   # else:
    #    objectiveData['text'][number] = sentence


#    This script will split the data/data.tsv into train/validation/test.tsv files.
