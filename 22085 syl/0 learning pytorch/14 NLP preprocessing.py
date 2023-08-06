en_text = "Be extremely subtle, even to the point of formlessness. Be extremely mysterious, even to the point of soundlessness. Thereby you can be the director of the opponent's fate."

import spacy
spacy_en = spacy.load('en_core_web_sm')

print([tok.text for tok in spacy_en.tokenizer(en_text)])

import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize
print(word_tokenize(en_text))

kor_text = "사과의 놀라운 효능이라는 글을 봤어. 그래서 오늘 사과를 먹으려고 했는데 사과가 썩어서 슈퍼에 가서 사과랑 오렌지 사왔어"

# something wrong with here
from konlpy.tag import Mecab
tokenizer = Mecab()
print(tokenizer.morphs(kor_text))
