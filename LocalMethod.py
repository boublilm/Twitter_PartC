import numpy as np

MIN_SIMILARITY = 0.2

class LocalMethod:

    @staticmethod
    def expand_query(query, doc_set):
        matrix, term_map = LocalMethod.create_matrix(doc_set)
        inverse_map = {inx: term for term, inx in term_map.items()}
        new_terms = []
        all_terms = query.split(' ')
        for term in all_terms:
            if term not in term_map:
                continue
            t_index = term_map[term]
            terms_vector = matrix[t_index]
            new_term_index = np.argsort(terms_vector)[-2]
            best_term = inverse_map[new_term_index]
            if best_term not in all_terms:
                new_terms.append(best_term)
        new_terms = LocalMethod.filter_new_words(query, new_terms, matrix, term_map)

        new_query = query + ' ' + ' '.join(new_terms)
        return new_query

    @staticmethod
    def filter_new_words(query, words_to_add, matrix, term_map):
        to_add = set()
        for word in words_to_add:
            w_index = term_map[word]
            terms_vector = matrix[w_index]
            counter = 0
            for term in query.split(' '):
                if term not in term_map:
                    continue
                t_index = term_map[term]
                similarity = terms_vector[t_index]
                if similarity >= MIN_SIMILARITY:
                    counter += 1
                    if counter > 1:
                        to_add.add(word)
                        break
        return to_add

    @staticmethod
    def create_matrix(doc_set):
        terms = set()
        for doc in doc_set:
            terms.update(doc_set[doc].keys())

        term_map = {term: inx for inx, term in enumerate(terms)}
        N = len(terms)
        matrix = np.zeros((N, N))
        for doc in doc_set:
            terms_in_doc = doc_set[doc]
            for t1 in terms_in_doc:
                t1_index = term_map[t1]
                t1_tf = terms_in_doc[t1]
                for t2 in terms_in_doc:
                    t2_index = term_map[t2]
                    t2_tf = terms_in_doc[t2]
                    matrix[t1_index, t2_index] += t1_tf*t2_tf

        normalized_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                c_ij= matrix[i, j]
                c_ii = matrix[i, i]
                c_jj = matrix[j, j]
                normalized_matrix[i,j] = c_ij / (c_ii + c_jj - c_ij)

        return normalized_matrix, term_map

