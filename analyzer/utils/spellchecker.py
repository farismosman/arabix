# original implementation http://norvig.com/spell-correct.html


import re
from collections import Counter
from cltk.corpus.arabic.alphabet import HAMZAT, LETTERS


class SpellChecker:
    
    def __init__(self, text_df):
        document = self.content(text_df)
        self.document = document

    def arabic_alphabet(self):
        result = []
        result.extend(HAMZAT)
        result.extend(LETTERS)
        return result

    def words(self, text):
        return re.findall(r'\w+', text.lower())

    def content(self, df):
        _all_tweets = ' '.join('{}'.format(val) for _, val in df.to_dict().items())  
        return Counter(self.words(_all_tweets))

    def P(self, word):
        "Probability of `word`."
        total_no_of_words = sum(self.document.values())
        return self.document[word] / total_no_of_words

    def suggest(self, sentence):
        result = []
        _words = sentence.split()
        for word in _words:
            result.append(self.correction(word))
        return ' '.join(w for w in result)

    def correction(self, word):
        "Most probable spelling correction for word."
        return max(self.candidates(word), key=self.P)

    def candidates(self, word):
        "Generate possible spelling corrections for word."
        return (self.known([word]) or self.known(self.first_correction(word)) or self.known(self.second_correction(word)) or [word])

    def known(self, words):
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in self.document)

    def first_correction(self, word):
        "All edits that are one edit away from `word`."
        letters    = self.arabic_alphabet()
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def second_correction(self, word): 
        "All edits that are two edits away from `word`."
        return (_second_edit for _first_edit in self.first_correction(word) for _second_edit in self.first_correction(_first_edit))