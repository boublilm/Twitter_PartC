import numpy as np
# you can change whatever you want in this module, just make sure it doesn't
# break the searcher module
from Word2Vec import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity


class Ranker:
    def __init__(self):
        self.w2v = Word2Vec()

    def rank_relevant_docs(self,relevant_docs, k=None):
        """
        This function provides rank for each relevant document and sorts them by their scores.
        The current score considers solely the number of terms shared by the tweet (full_text) and query.
        :param k: number of most relevant docs to return, default to everything.
        :param relevant_docs: dictionary of documents that contains at least one term from the query.
        :return: sorted list of documents by score
        """
        posting_files, doc_set, query_terms, term_dict = relevant_docs

        # Creating Matrix
        doc_map = {tweet_id: inx for inx, tweet_id in enumerate(doc_set)}
        term_map = {term: inx for inx, term in enumerate(query_terms)}
        query_len = len(query_terms)
        number_of_docs = len(doc_set)
        matrix = np.zeros((number_of_docs, query_len))
        query_vector = np.zeros((query_len, 1))
        query_weight = 0

        # Inset values to matrix and calculating query_weight (scalar) and query_vector
        for term in query_terms:
            docs = term_dict[term][0]
            j = term_map[term]
            w_iq = term_dict[term][3] * query_terms[term]
            query_weight += w_iq ** 2
            query_vector[j, 0] = w_iq
            for tweet_id in docs:
                w_ij = posting_files[(term, tweet_id)][1]
                i = doc_map[tweet_id]
                matrix[i, j] = w_ij

        # Math calculation for
        document_vector = np.dot(matrix, query_vector)  # MONE
        size_vector = np.zeros((number_of_docs, 1))  # MECHANE
        query_weight = query_weight ** 0.5
        for tweet_id, indx in doc_map.items():
            size_vector[indx, 0] = (doc_set[tweet_id] ** 0.5) * query_weight
        ranking = np.transpose(np.divide(document_vector, size_vector)).tolist()[0]

        ranked_tweets = [(x, y) for x, y in zip(doc_map, ranking)]
        sorted_tweets = sorted(ranked_tweets, key=lambda x: x[1], reverse=True)
        return [x[0] for x in sorted_tweets][:k]

    def rank_relevant_docs_1(self,relevant_docs, k=None):
        """
        This function provides rank for each relevant document and sorts them by their scores.
        The current score considers solely the number of terms shared by the tweet (full_text) and query.
        :param k: number of most relevant docs to return, default to everything.
        :param relevant_docs: dictionary of documents that contains at least one term from the query.
        :return: sorted list of documents by score
        """
        posting_files, doc_set, query_terms, term_dict = relevant_docs

        # Creating Matrix
        doc_map = {tweet_id: inx for inx, tweet_id in enumerate(doc_set)}
        term_map = {term: inx for inx, term in enumerate(query_terms)}
        query_len = len(query_terms)
        number_of_docs = len(doc_set)
        matrix = np.zeros((number_of_docs, query_len))
        query_vector = np.zeros((query_len, 1))
        query_weight = 0

        # Inset values to matrix and calculating query_weight (scalar) and query_vector
        for term in query_terms:
            docs = term_dict[term][0]
            j = term_map[term]
            #w_iq = term_dict[term][3] * query_terms[term]
            w_iq = 1
            query_weight += w_iq ** 2
            query_vector[j, 0] = w_iq
            for tweet_id in docs:
                w_ij = posting_files[(term, tweet_id)][1]
                i = doc_map[tweet_id]
                matrix[i, j] = w_ij

        # Math calculation for
        document_vector = np.dot(matrix, query_vector)  # MONE
        size_vector = np.zeros((number_of_docs, 1))  # MECHANE
        query_weight = query_weight ** 0.5
        for tweet_id, indx in doc_map.items():
            size_vector[indx, 0] = (doc_set[tweet_id] ** 0.5) * query_weight
        ranking = np.transpose(np.divide(document_vector, size_vector)).tolist()[0]

        ranked_tweets = [(x, y) for x, y in zip(doc_map, ranking)]
        sorted_tweets = sorted(ranked_tweets, key=lambda x: x[1], reverse=True)
        return [x[0] for x in sorted_tweets]

    def get_cosine_similarity(self,feature_vec_1, feature_vec_2):
        return cosine_similarity(feature_vec_1.reshape(1, -1), feature_vec_2.reshape(1, -1))[0][0]

    def rank_relevant_docs_by_w2v(self,relevant_docs, k=None):
        """
        This function provides rank for each relevant document and sorts them by their scores.
        The current score considers solely the number of terms shared by the tweet (full_text) and query.
        :param k: number of most relevant docs to return, default to everything.
        :param relevant_docs: dictionary of documents that contains at least one term from the query.
        :return: sorted list of documents by score
        """
        posting_files, doc_set, query_terms, term_dict = relevant_docs
        query_vector = self.w2v.get_vector(query_terms)
        normed_query_vector = np.linalg.norm(query_vector)
        ranks = []
        for doc in doc_set:
            vector = self.w2v.get_vector(doc_set[doc])
            if vector == []: continue
            #cosine_w2v = get_cosine_similarity(query_vector,vector)
            cosine_w2v = np.dot(vector,query_vector) / np.linalg.norm(vector) * normed_query_vector
            ranks.append((doc,cosine_w2v))
        sorted_tweets = sorted(ranks, key=lambda x: x[1], reverse=True)
        return [x[0] for x in sorted_tweets][:800]