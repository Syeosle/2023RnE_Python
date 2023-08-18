def substring(s) :
    L = []
    n = len(s)
    for i in range(n) :
        for j in range(n - i) :
            L.append(s[j:i+j+1])
    return L

def sampling(s) :
    L = s.split(' ')
    for i in range(len(L)) :
        L[i] = substring(L[i])
    return L

if __name__=="__main__" :
    s = input(">>") 
    print(sampling(s))    