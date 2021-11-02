import matplotlib.pyplot as plt
import statistics

with open('data/sense.txt','r') as f:
    lines = f.readlines()

lenToFreq = {}
leng = []
endToFreq = {}

for line in lines:
    spl = line.split(' ')
    length = len(spl)
    leng.append(length)
    if length not in lenToFreq:
        lenToFreq[length] = 0
    if line[-2] not in endToFreq:
        endToFreq[line[-2]] = 0
    lenToFreq[length] += 1
    endToFreq[line[-2]] += 1

for key in lenToFreq.keys():
    lenToFreq[key] = lenToFreq[key]/len(lines)

for key in endToFreq.keys():
    endToFreq[key] = endToFreq[key]/len(lines)

lenToFreq = sorted(lenToFreq.items())
print(lenToFreq)

print("mean length: ",sum(leng)/len(leng))
print("standard deviation: ",statistics.stdev(leng))

print(endToFreq)

with open('data/nonsense.txt','r') as f:
    lines = f.readlines()
    try:
        lines.remove("--START HERE--\n")
    except:
        pass

lenToFreq = {}
leng = []
endToFreq = {}

for line in lines:
    spl = line.split(' ')
    length = len(spl)
    leng.append(length)
    if length not in lenToFreq:
        lenToFreq[length] = 0
    if line[-2] not in endToFreq:
        endToFreq[line[-2]] = 0
    lenToFreq[length] += 1
    endToFreq[line[-2]] += 1

for key in lenToFreq.keys():
    lenToFreq[key] = lenToFreq[key]/len(lines)

for key in endToFreq.keys():
    endToFreq[key] = endToFreq[key]/len(lines)

lenToFreq = sorted(lenToFreq.items())
print(lenToFreq)

print("mean length: ",sum(leng)/len(leng))
print("standard deviation: ",statistics.stdev(leng))

print(endToFreq)
