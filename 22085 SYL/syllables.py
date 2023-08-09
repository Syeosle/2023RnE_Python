EXCEPTION_ENG = frozenset(['cafe'])


def syllableCountEn(string):
    cnt = 0
    string = string.lower()

    # remove punctuations

    # exception counting first

    # syllables counting for other words
    # 1. 연속된 모음 덩어리 수 셈
    # 2. 단어 끝이 e인 단어 수 뺌
    # 3. (자음)le로 끝나는 단어 수 셈

    return -1


def syllableCountKr(string):
    # remove punctuations
    # remove spaces
    return len(string)


def sentenceSplit(string):
    L = []
    # index 늘리며 공백 아닌 경우 영어인지 확인
    # 바뀔때 마다 잘라 (str, lang)의 튜플 리스트 반환
    return L


def syllablesOf(string, isWord=False):
    L = sentenceSplit(string)
    cnt = 0
    for item, lang in L:
        cnt += syllableCountEn(item) if lang == 'en' else syllableCountKr(item)
    return cnt


def splitPointsOf(string):
    L = [syllablesOf(item, isWord=True) for item in string.split()]


def bestTranslate(strings, syllables):
    # strings: list of translated sentences
    # syllables: syllables to fit
    L = []  # shortest strings
    sylls = [abs(syllablesOf(item)-syllables) for item in strings]
    minDiff = min(sylls)
    for i in range(len(strings)):
        if sylls[i] == minDiff: L.append(strings[i])



    L = []

    return -1
