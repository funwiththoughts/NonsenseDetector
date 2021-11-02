import string
import random
import time
import numpy as np

with open('data/sense.txt','r') as f:
    lines = f.readlines()

words = []

for line in lines:
    spl = line.split(" ")
    for word in spl:
        words.append(word)

with open('data/nonsense.txt','r') as f:
    nonLines = f.readlines()

'''
while len(nonLines) < len(lines):
    sentence = ""
    while True:
        sentence += random.choice(words)
        if sentence[-1] != "\n":
            sentence += " "
        else:
            break
    if 5 < len(sentence.split(' ')) and len(sentence.split(' ')) < 46:
        with open('data/nonsense.txt','a') as f:
            f.write(sentence)
        print(sentence)
    with open('data/nonsense.txt','r') as f:
        nonLines = f.readlines()
    print(len(nonLines))

        
lengthPairs = [(6, 1), (8, 18), (9, 57), (10, 216), (11, 587), (12, 1070), (13, 1764), (14, 2343), (15, 2869), (16, 2977), (17, 3084), (18, 3054), (19, 2958), (20, 2715), (21, 2513), (22, 2197), (23, 1699), (24, 1342), (25, 867), (26, 545), (27, 362), (28, 220), (29, 115), (30, 54), (31, 31), (32, 15), (33, 15), (34, 5), (35, 4), (36, 2), (38, 1), (39, 1), (45, 1)]

for pair in lengthPairs:
    with open('data/nonsense.txt','r') as f:
        flulf = f.readlines()
    flulf = [line for line in flulf if len(line.split(' ')) == pair[0]]
    needed = pair[1] - len(flulf)
    for i in range(needed):
        sentence = ""
        while len(sentence.split(' ')) != pair[0]:
            sentence = ""
            sentence = random.choice(words)
            while sentence[-1] != "\n":
                sentence += " "
                sentence += random.choice(words)
        with open('data/nonsense.txt','a') as f:
            f.write(sentence)
        print(sentence)
        with open('data/nonsense.txt','r') as f:
            nonLines = f.readlines()
        print(len(nonLines))
    
'''
#List of sentence beginnings, arranged by "common-ness"
#Generator uses this to start a sentence
#Dictionary mapping words to words that are most likely to follow them
#Generator uses this to pick subsequent words until it reaches a "\n"
starts = []
comboDict = {}

for line in lines:
    spl = line.split(" ")
    starts.append(spl[0])
    for i in range(0,len(spl)-1):
        if spl[i] not in comboDict:
            comboDict[spl[i]] = [spl[i+1]]
        else:
            comboDict[spl[i]].append(spl[i+1])

foo = []

while len(foo) < len(lines):
    sentence = np.random.choice(starts)
    while sentence[-1] != "\n":
        spl = sentence.split(" ")
        sentence += " "
        sentence += np.random.choice(comboDict[spl[-1]])
    print(sentence)
    if len(sentence.split(" ")) > 4:
        foo.append(sentence)
        with open('data/nonsense.txt','a') as f:
            f.write(sentence)
    with open('data/nonsense.txt','r') as f:
        foo = f.readlines()
    print(len(foo))

