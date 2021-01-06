import configuration
from Word2Vec import Word2Vec
from ranker import Ranker
import utils
from WordNet import WordNet
from LocalMethod import LocalMethod

# DO NOT MODIFY CLASS NAME
class Searcher:
    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit. The model 
    # parameter allows you to pass in a precomputed model that is already in 
    # memory for the searcher to use such as LSI, LDA, Word2vec models. 
    # MAKE SURE YOU DON'T LOAD A MODEL INTO MEMORY HERE AS THIS IS RUN AT QUERY TIME.
    def __init__(self, parser, indexer, model=None):
        self._parser = parser
        self._indexer = indexer
        self._ranker = Ranker()
        self._model = model
        self.w2v = None

    def load_w2v(self):
        self.w2v = Word2Vec()
        self._ranker = Ranker(self.w2v)

    def search_reg(self, query, k=None):
        """
        Executes a query over an existing index and returns the number of
        relevant docs and an ordered list of search results (tweet ids).
        Input:
            query - string.
            k - number of top results to return, default to everything.
        Output:
            A tuple containing the number of relevant search results, and
            a list of tweet_ids where the first element is the most relavant
            and the last is the least relevant result.
        """
        query = self._parser.remove_stopwords(query)
        parsed_query, parsed_entities = self._parser.parse_query(query)
        relevant_docs = self._relevant_docs_from_posting(parsed_query, parsed_entities)
        n_relevant = len(relevant_docs[1])
        ranked_doc_ids = self._ranker.rank_relevant_docs(relevant_docs)
        return n_relevant, ranked_doc_ids

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def search(self, query, k=None):
        """ 
        Executes a query over an existing index and returns the number of 
        relevant docs and an ordered list of search results (tweet ids).
        Input:
            query - string.
            k - number of top results to return, default to everything.
        Output:
            A tuple containing the number of relevant search results, and 
            a list of tweet_ids where the first element is the most relavant 
            and the last is the least relevant result.
        """
        query = WordNet.expand_query(self._parser.remove_stopwords(query))
        parsed_query, parsed_entities = self._parser.parse_query(query)
        relevant_docs = self._relevant_docs_from_posting(parsed_query, parsed_entities)
        n_relevant = len(relevant_docs[1])
        ranked_doc_ids = self._ranker.rank_relevant_docs(relevant_docs)
        return n_relevant, ranked_doc_ids

    def search_w2v(self, query, k=None):
        """
        Executes a query over an existing index and returns the number of
        relevant docs and an ordered list of search results (tweet ids).
        Input:
            query - string.
            k - number of top results to return, default to everything.
        Output:
            A tuple containing the number of relevant search results, and
            a list of tweet_ids where the first element is the most relavant
            and the last is the least relevant result.
        """
        query = self._parser.remove_stopwords(query)
        parsed_query, parsed_entities = self._parser.parse_query(query)
        posting_files, doc_set, query_terms, term_dict = self._relevant_docs_from_posting(parsed_query, parsed_entities)
        n_relevant = len(doc_set)
        doc_set = self._indexer.get_doc_list(doc_set)
        ranked_doc_ids = self._ranker.rank_relevant_docs_by_w2v((posting_files, doc_set, query_terms, term_dict))
        return n_relevant, ranked_doc_ids

    def search_local(self, query, k=None):
        """
        Executes a query over an existing index and returns the number of
        relevant docs and an ordered list of search results (tweet ids).
        Input:
            query - string.
            k - number of top results to return, default to everything.
        Output:
            A tuple containing the number of relevant search results, and
            a list of tweet_ids where the first element is the most relavant
            and the last is the least relevant result.
        """
        NUM_OF_DOCS = 200
        k = 800
        # Get all relevant docs
        query = self._parser.remove_stopwords(query)
        parsed_query, parsed_entities = self._parser.parse_query(query)
        relevant_docs = self._relevant_docs_from_posting(parsed_query, parsed_entities)
        ranked_doc_ids = self._ranker.rank_relevant_docs_1(relevant_docs)
        # Expand query
        doc_dict = self._indexer.get_doc_list(ranked_doc_ids[:NUM_OF_DOCS])
        new_query = LocalMethod.expand_query(query, doc_dict)
        parsed_query, parsed_entities = self._parser.parse_query(new_query)
        # Run new query
        relevant_docs = self._relevant_docs_from_posting(parsed_query, parsed_entities)
        n_relevant = len(relevant_docs[1])
        ranked_doc_ids = self._ranker.rank_relevant_docs(relevant_docs,k)
        return n_relevant, ranked_doc_ids

    # feel free to change the signature and/or implementation of this function 
    # or drop altogether.
    def _relevant_docs_from_posting(self, parsed_query, parsed_entities):
        """
        This function loads the posting list and count the amount of relevant documents per term.
        :param query_as_list: parsed query tokens
        :return: dictionary of relevant documents mapping doc_id to document frequency.
        """
        query_terms = {term: parsed_query[term] for term in parsed_query}  # query terms is {term:tf}

        # Preparing query terms as appear in dictionary
        terms = list(query_terms)
        for term in terms: # fixing words in query to match the words in dictionary (lower\upper)
            if not self._indexer._is_term_exist(term):
                if term.isupper() and self._indexer._is_term_exist(term.lower()):
                    query_terms[term.lower()] = query_terms[term]
                    query_terms.pop(term)
                elif term.islower() and self._indexer._is_term_exist(term.upper()):
                    query_terms[term.upper()] = query_terms[term]
                    query_terms.pop(term)
                else:
                    query_terms.pop(term)

        # Adding entities to query terms
        for entity in parsed_entities:
            if self._indexer._is_term_exist(entity):
                query_terms[entity] = parsed_entities[entity]

        # Creating docs set and preparing bucked_id list
        doc_set = {}
        return_postings = {}
        return_term_dict = {}
        for term in query_terms:
            doc_set.update(self._indexer.get_term_tweets_list(term))
            return_postings.update(self._indexer.get_term_posting_list(term))
            return_term_dict[term] = self._indexer.get_term_records(term)

        return return_postings, doc_set, query_terms, return_term_dict
