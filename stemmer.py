from nltk import stem

class Stemmer:
    def __init__(self):
        self.stemmer = stem.SnowballStemmer('english')
        self.dict = {}

    def stem_term(self, token):
        """
        This function stem a token
        :param token: string of a token
        :return: stemmed token
        """

        if token in self.dict:
             return self.dict[token]
        term = self.stemmer.stem(token)
        self.dict[token] = term
        return term