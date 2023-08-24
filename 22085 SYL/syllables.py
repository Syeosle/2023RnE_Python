import string as st
EXCEPTION_ENG = {'cafe': 2}
VOWEL_ENG = 'aeiouy'

# for Eng
# remove punctuations
# exception counting first
# syllables counting for other words:
# 1. 연속된 모음 덩어리 수 셈
# 2. 단어 끝이 e인 단어 수 뺌
# 3. (자음)le로 끝나는 단어 수 셈

# for Kr
# remove punctuations
# remove spaces
# len = syllables


def syllEng(string):  # 글자만 남은 lowercase 단어 -> syllables
    if string in EXCEPTION_ENG: return EXCEPTION_ENG[string]
    cnt = 0
    string = ' ' + string
    for i in range(1, len(string)):
        if string[i-1] not in VOWEL_ENG and string[i] in VOWEL_ENG: cnt += 1
    if string[-1] == 'e' and string[-2] not in 'l' + VOWEL_ENG: cnt -= 1
    return cnt


def syllRom(string):  # 글자만 남은 lowercase 단어 -> syllables
    cnt = 0
    string = ' ' + string
    for i in range(1, len(string)):
        if string[i-1] not in VOWEL_ENG and string[i] in VOWEL_ENG: cnt += 1
    return cnt


def splitPoints(string, isRoman=False):  # 글자만 남은 lowercase 문장 -> 띄어 쓰기의 음절 위치 (마지막은 총 음절)
    words = string.split()
    L = []  # result
    for word in words:
        if word.encode().isalpha():
            if isRoman: L.append(syllRom(word))
            else: L.append(syllEng(word))
        else: L.append(len(word))

    for i in range(1, len(L)):
        L[i] += L[i-1]
    return L


def onlyLower(string):
    for i in st.punctuation: string = string.replace(i, '')
    string = string.lower()
    return string


def bestTranslate(strings, original, isRoman=False):  # 번역문, 원문 -> 제일 맞는 문장
    # strings: list of translated sentences ordered by probability
    strings_keep = [i for i in strings]
    for i in range(len(strings)):
        strings[i] = onlyLower(strings[i])
    original = onlyLower(original)

    info_str = [(splitPoints(strings[i], isRoman), i) for i in range(len(strings))]
    info_original = splitPoints(original, isRoman)

    # filter 1: by length
    info_str.sort(key=lambda x: abs(x[0][-1] - info_original[-1]))
    for i in range(len(info_str)):
        if info_str[i][0][-1] != info_str[0][0][-1]:
            info_str = info_str[:i]
            break

    # filter 2: by fit
    score = [0] * len(info_str)
    for i in info_original:
        for j in range(len(info_str)):
            if i in info_str[j][0]: score[j] += 1

    return strings_keep[info_str[score.index(max(score))][1]]
