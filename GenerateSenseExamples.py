import string
import random
from urllib.request import urlopen,Request

word_site = "http://www.mieliestronk.com/corncob_lowercase.txt"
response = urlopen(word_site)
txt = response.read()
WORDS = txt.splitlines()

words = []
for i in range(0,len(WORDS)):
    word = str(WORDS[i])
    word = word[2:len(word)-1]
    words.append(word)

print(len(words))
print(fuck)

with open('wordsToRemove.txt','r') as f:
    toRemove = [x[0:-1] for x in f.readlines()]

words = [word for word in words if word not in toRemove]

with open('data/sense.txt','r') as f:
    existing = f.readlines()

for line in existing:
    for word in line.split(' '):
        if word in words:
            words.remove(word)
        elif word.lower() in words:
            words.remove(word.lower())
        elif word[-1] in string.punctuation and word[0:len(word)-1] in words:
            words.remove(word[0:len(word)-1])
        elif word[-1] == '\n' and word[-2] in string.punctuation and word[0:len(word)-2] in words:
            words.remove(word[0:len(word)-2])
        elif word[-1] in string.punctuation and word[0:len(word)-1].lower() in words:
            words.remove(word[0:len(word)-1].lower())
        elif word[-1] == '\n' and word[-2] in string.punctuation and word[0:len(word)-2].lower() in words:
            words.remove(word[0:len(word)-2].lower())
        elif "-" in word and word.replace("-","") in words:
            words.remove(word.replace("-",""))
        elif "-" in word and word[-1] in string.punctuation and word.replace("-","")[0:len(word.replace("-",""))-1] in words:
            words.remove(word.replace("-","")[0:len(word.replace("-",""))-1])
        elif "-" in word and word[-1] == "\n" and word[-2] in string.punctuation and word.replace("-","")[0:len(word.replace("-",""))-2] in words:
            words.remove(word.replace("-","")[0:len(word.replace("-",""))-2])

skip = False
while words != []:
    if word == words[0]:
        with open('wordsToRemove.txt','a') as f:
            f.write(word)
            f.write('\n')
        words.remove(word)
    word = words[0]
    print("word = ",word)
    with urlopen('https://www.wordhippo.com/what-is/sentences-with-the-word/'+word+'.html') as url:
        txt = url.read()
    lines = txt.splitlines()
    try:
        backupWord = str(word)
        try:
            index = lines.index(b'<tr class="exv2row1"  id="gexv2row1">')
        except:
            index = lines.index(b'<tr class="exv2row1"  id="hexv2row1">')
        x = str(lines[index+1])
        ind = x.index(">")
        x = x.replace(x[0:ind+1],'')
        x = x.replace("</td>'","")
        x = x.replace("<b>","")
        x = x.replace("</b>","")
        with open('data/sense.txt','r') as f:
            skip = (x+'\n' in f.readlines())
        if not skip:
            print(x)
            with open('data/sense.txt','a') as f:
                f.write('\n')
                f.write(x)
            for word in x.split(' '):
                if word in words:
                    words.remove(word)
                elif word.lower() in words:
                    words.remove(word.lower())
                elif word[-1] in string.punctuation and word[0:len(word)-1] in words:
                    words.remove(word[0:len(word)-1])
                elif word[-1] in string.punctuation and word[0:len(word)-1].lower() in words:
                    words.remove(word[0:len(word)-1].lower())
                elif "-" in word and word.replace("-","") in words:
                    words.remove(word.replace("-",""))
                elif "-" in word and word[-1] in string.punctuation and word.replace("-","")[0:len(word.replace("-",""))-1] in words:
                    words.remove(word.replace("-","")[0:len(word.replace("-",""))-1])
        word = backupWord
        skip = False
    except:
        with open('wordsToRemove.txt','r') as f:
            skip = (word+'\n' in f.readlines())
        if not skip:
            with open('wordsToRemove.txt','a') as f:
                f.write(word)
                f.write('\n')
        words.remove(word)
    print(len(words))

'''
with urlopen('https://www.wordhippo.com/what-is/sentences-with-the-word/abash.html') as url:
    txt = url.read()

lines = txt.splitlines()
print(str(lines[4148]))
#print(str(lines[3975])[25:len(str(lines[3975]))-6])

#print(lines.index(b'<tr class="exv2row1"  id="hexv2row1">'))
'''
        
'''
for line in lines:
    split = line.split(' ')
    for word in words:
        if (word in split) or (word.capitalize() in split):
            words.remove(word)
        else:
            for punct in string.punctuation:
                if word+punct in split or word.capitalize()+punct in split:
                    words.remove(word)
                elif split[-1] in [word+punct+'\n',word.capitalize()+punct+'\n']:
                    words.remove(word)

print(len(words))
print(len(lines)+len(words))
print(words)
print(time.time()-start)


wordsPlusPunct = WORDS + string.punctuation.split()
everythingExceptPunct = lettersPlusDigits.split() + WORDS

options = [uppercase,lowercase,letters,lettersPlusDigits,lettersPlusPunct,lettersPlusDigitsPlusPunct,everything]

f = open('data/nonsense.txt','w')

for i in range(2999):
    sequence = ''
    for j in range(random.choice(range(1,40))):
        word = str(random.choice(wordsPlusPunct))
        if word[0:2] == "b'":
            word = word[2:len(word)-1]
        if j > 0 and word not in string.punctuation:
            sequence += ' '
        sequence += word
        print(word)
        print('\n')
        if word in ['.','?','!']:
            break
    f.write('\n')
    f.write(sequence)

f.close()
'''
