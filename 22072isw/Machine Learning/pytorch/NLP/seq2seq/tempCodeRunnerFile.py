def pairFix(ko, en) :
    ko_sentences = kss.split_sentences(ko)
    en_sentences = sent_tokenize(en)
    if len(ko_sentences) != len(en_sentences) :
        return None
    ko_sentences = pd.DataFrame(ko_sentences)
    en_sentences = pd.DataFrame(en_sentences)
    return pd.concat(ko_sentences, en_sentences, axis=1)