EXCEPTION_ENG = frozenset([' cafe '])


def syllableCountEng(string):
    cnt = 0
    for i in EXCEPTION_ENG:
        string.count(i)