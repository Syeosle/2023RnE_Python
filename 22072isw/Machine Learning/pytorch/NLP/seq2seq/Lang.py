import os
import re
import pandas as pd
import torch

from konlpy.tag import Hannanum
from nltk import word_tokenize


KO = Hannanum()
D = [1, 5, 8, 2, 2, 2]

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 35


def readFromLegacy(type) :
    data_dir = './data/KoEnData'
    ko_path = os.path.join(data_dir + '/lang', 'ko.txt')
    en_path = os.path.join(data_dir + '/lang', 'en.txt')
    type_path = os.path.join(data_dir + '/parsed', str(type))
    if not (os.path.exists(ko_path) and os.path.exists(en_path) and os.path.exists(type_path)) :
        return None
    
    ko_lang, en_lang = Lang('ko'), Lang('en')
    ko_lang.readFromFile(ko_path)
    en_lang.readFromFile(en_path)
    
    file_list = os.listdir(type_path)
    n = len(file_list)
    R = []
    for p in range(n) :
        print("Reading %d%s fragment! (%.2lf%s)" % \
                ((p + 1), "st" if (p + 1)%10 == 1 else ("nd" if (p + 1)%10 == 2 else ("rd" if (p + 1)%10 == 3 else "th")), (p + 1)/n*100, '%'))
        file = open(os.path.join(type_path, file_list[p]), 'r', encoding='utf-8')
        data = file.readlines()
        R.extend(list(map(lambda x : list(map(
                lambda y : torch.tensor(list(map(int, y.split(',')))).view(-1, 1).to('cuda'),
                x.strip().split(' '))), data)))

    return R, ko_lang, en_lang


def readData(path, type, ko_lang, en_lang) :
    if not os.path.exists(path) :
        return False 
    else :
        if checkDir(type) :
            return False
        type_path = os.path.join('./data/KoEnData/parsed', str(type))
        start = D[type - 1]
        data = pd.read_excel(path)
        data = data.iloc[:, start:start+2]
        print("Number of sequences : %d" % (len(data)))
        R= []
        f_que, f_idx = [], len(os.listdir(type_path))
        for p in range(len(data)) :     
            if (p + 1) % 1000 == 0 or p == len(data) - 1 :
                f_idx += 1
                frag = open(os.path.join(type_path, str(f_idx) + '.txt'), 'w', encoding='utf-8')
                frag.write('\n'.join(f_que))
                frag.close()
                print("├ %dth sequence parsed! (%.2lf%s)" % (p + 1, (p + 1) / len(data) * 100, '%'))
            ko_sent, en_sent = data.iloc[p, 0], data.iloc[p, 1]
            ko_vec = sentenceParse(ko_sent, ko_lang, koreanTag)
            en_vec = sentenceParse(en_sent, en_lang, englishTag)
            if max(len(ko_vec), len(en_vec)) <= MAX_LENGTH :
                data_line = (ko_vec, en_vec)
                f_que.append(' '.join((list(map(
                    lambda x : ','.join(list(map(lambda y : str(y.item()), x))), data_line))
                )))
                R.append(data_line)
        frag.close()
        
        ko_lang.saveToFile('./data/KoEnData/lang/ko.txt')
        en_lang.saveToFile('./data/KoEnData/lang/en.txt')
                
        return R


def sentenceParse(sentence, lang, tagfunc) :
    sentence = tagfunc(sentence)
    lang.addWords(sentence)
    vec = [lang.word2index[word] for word in sentence]
    vec.append(EOS_token)
    vec = torch.tensor(vec).view(-1, 1).to('cuda')
    return vec


def koreanTag(sentence: str) :
    s = normalizeString(sentence)
    return KO.morphs(s)

def englishTag(sentence: str) :
    s = normalizeString(sentence)
    return word_tokenize(s)

def normalizeString(s: str):
    s = s.lower()
    s = re.sub(r"[^a-zA-Z가-힣.!?]+", r" ", s)
    return s

def mappingZero(tensor: torch.Tensor, length: int) :
    n, type = len(tensor), tensor.dtype
    if n < length :
        return
    zeros = torch.zeros(length - n, type=type).view(-1, 1)
    tensor.expand(zeros)
    


def checkDir(type) :
    path = os.path.join('./data/KoEnData/parsed', str(type))
    if os.path.exists(path) :
        return True
    os.makedirs(path)
    return False

class Lang :
    def __init__(self, name) :
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1:"EOS"}
        self.n_words = 2
            
    def addWord(self, word) :
        if word not in self.word2index :
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else :
            self.word2count[word] += 1
    
    def addWords(self, wordlist) :
        for word in wordlist :
            self.addWord(word)
            
    def readFromFile(self, path) :
        if not os.path.exists(path) :
            return None
        file = open(path, 'r', encoding='utf-8')
        L = file.readlines()
        self.n_words = len(L)
        for i in L :
            word_data = i.strip().split('\t')
            word = word_data[0]
            index, count = int(word_data[1]), int(word_data[2])
            self.word2count[word] = count
            self.word2index[word] = index
            self.index2word[index] = word
        file.close()
        
    def saveToFile(self, path) :
        file = open(path, 'w', encoding='utf-8')
        R = []
        for word in self.word2index :
            index, count = self.word2index[word], self.word2count[word]
            index, count = str(index), str(count)
            R.append('\t'.join((word, index, count)))
        file.write('\n'.join(R))
        file.close()
        

def dataPrepare(type) :
    data_dir = './data/KoEnData'
    data_L = os.listdir(data_dir)
    if type not in range(1, 7) :
        assert False
    pairs = []
    ko_lang, en_lang = Lang('ko'), Lang('en')
    if os.path.exists(os.path.join(data_dir + '/parsed', str(type))) :
        print("Reading from legacy parsed data...")
        pairs, ko_lang, en_lang = readFromLegacy(type)
    else :
        for f in data_L :
            if f[0] == str(type) :
                print("Find new file! : %s" % (f))
                file_data = readData(os.path.join(data_dir, f), type, ko_lang, en_lang)
                if file_data :
                    for datas in file_data :
                        pairs.append(datas)
    print("\nSentences : %d\nWords KO : %d\nWords EN : %d\n" % (len(pairs), ko_lang.n_words, en_lang.n_words))
    return pairs, ko_lang, en_lang