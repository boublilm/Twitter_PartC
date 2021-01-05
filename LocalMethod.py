import numpy as np

class LocalMethod:

    @staticmethod
    def expand_query(query, doc_set):
        matrix, term_map = LocalMethod.create_matrix(doc_set)
        new_query = query

        return new_query

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

