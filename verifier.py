import string
import random
import time
import numpy as np

with open('data/sense.txt','r') as f:
    lines = f.readlines()

with open('data/verifier.txt','w') as f:
    for i in range(len(lines)):
        f.write(lines[i])
        print(lines[i])
        print(i," completed")
        print(len(lines)-i," remaining")
