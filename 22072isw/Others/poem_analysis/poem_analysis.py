import os
from test import sampling

directory = './poems'
L = []

while True :
    poem_name = input('>> ') + ".txt"
    fpath = os.path.join(directory, poem_name)
    if os.path.exists(fpath) :
        L = open(fpath, 'r', encoding='utf-8').readlines()
        break
    print("Poem does not exist!")

h, y, s = 0, 0, 1

L_sampled, Y = [], []
n = len(L)

for i in range(n) :
    if L[i][-1] == '\n' :
        L[i] = L[i][:-1]
    if len(L[i]) == 0 :
        s += 1
        if s < 2 :
            L_sampled.append(Y)
            Y = []
    else :
        Y.append(sampling(L[i]))
        if s != 0 :
            y += 1
            s = 0
        h += 1

print(h, y)
print(L_sampled)