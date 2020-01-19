class Tokenizer:

    def __init__(self, dictionary):
        self.dictionary = dictionary
    
    def tokenize(self, sentence):
        result = []
        _words = sentence.split()
        for word in _words:
            if word in self.dictionary:
                result.append(self.dictionary[word])
            else:
                result.append(word)

        return ' '.join(w for w in result)