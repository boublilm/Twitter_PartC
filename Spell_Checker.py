from spellchecker import SpellChecker

class SpellCheck:
    def __init__(self):
        self.sc = SpellChecker(distance=1)
        self.dict = {}

    def correction(self, token):
        """
        This function stem a token
        :param token: string of a token
        :return: stemmed token
        """

        if token in self.dict:
             return self.dict[token]
        term = self.sc.correction(token)
        self.dict[token] = term
        return term