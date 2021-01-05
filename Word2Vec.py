from gensim.models import KeyedVectors
import numpy as np

MODEL_PATH = r"C:\Users\maorb\model"

class Word2Vec:
    def _init_(self):
        self.wv = KeyedVectors.load(MODEL_PATH, mmap='r')

    def expand_query(self,query):
        query_list = [term for term in query.split(" ")]
        q_added = []
        # Get the first synset with the same name as term
        for term in query_list:
            try:
                similar = self.wv.similar_by_word(term)
            except:
                continue
            for sim in similar:
                if sim[1] > 0.3 and sim[0] not in q_added and sim[0] not in query_list:
                    q_added.append(sim[0])
                else:
                    break

        return ' '.join(query_list + q_added)

    def get_vector(self, doc):
        words = [word for word in doc if word in self.wv.vocab]
        if len(words) >= 1:
            return np.mean(self.wv[words], axis=0)
        else:
            return []