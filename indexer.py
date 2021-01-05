from utils import save_obj, load_obj
from cmath import log10
# DO NOT MODIFY CLASS NAME
class Indexer:
    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def __init__(self, config):
        self.inverted_idx = {}  # { term: [[tweet_id], df, cf, idf]}
        self.postingDict = {}  # {(term, tweet_id): [normalized tf, tf-idf]}
        self.config = config
        self.document_dict = {}  # {tweet_id : [set(words), |d|]}
        self.upper_terms = set()
        self.suspected_entities = {}  # ENTITY: (TWEETID, tf, max tf)

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def add_new_doc(self, document):
        """
        This function perform indexing process for a document object.
        Saved information is captures via two dictionaries ('inverted index' and 'posting')
        :param document: a document need to be indexed.
        :return: -
        """
        terms_in_document = document.term_doc_dictionary
        self.document_dict[document.tweet_id] = [terms_in_document, 0]
        tweet_id = document.tweet_id

        # Handle entities - verify only entities appear in 2+ tweets will be in our corpus
        for entity in document.entities:
            if entity in self.inverted_idx:
                terms_in_document[entity] = document.entities[entity]
            elif entity in self.suspected_entities:
                # Add entity as term
                prev_tweet_id = self.suspected_entities[entity][0]
                prev_tf = self.suspected_entities[entity][1]
                prev_max_tf = self.suspected_entities[entity][2]
                self.inverted_idx[entity] = [[prev_tweet_id], 1, prev_tf, None]
                self.postingDict.update({(entity, prev_tweet_id): [prev_tf / prev_max_tf, 0]})
                # Add to document term list to process, and remove from suspected
                terms_in_document[entity] = document.entities[entity]
                self.suspected_entities.pop(entity)
            else:  # new entity
                self.suspected_entities[entity] = (tweet_id, document.entities[entity], document.max_tf)

        # Go over each term in the doc - add to posting file and update term dictionary
        for term in terms_in_document.keys():
            tf = terms_in_document[term]

            old_term = term
            try:
                # Updating term dictionary
                if term not in self.inverted_idx:
                    # We want to make sure lower term and upper term are in the same bucket
                    if term.upper() in self.inverted_idx:  # we have lower and upper is inside - this is a new term
                        self.upper_terms.add(term.upper()) # add to fix list
                        self.inverted_idx[term] = [[tweet_id], 1, tf, None]
                    elif term.lower() in self.inverted_idx:  # we have upper and lower is inside - add to existing lower
                        term = term.lower()
                        term_rec = self.inverted_idx[term]
                        term_rec[0].append(tweet_id)
                        term_rec[1] += 1  # df
                        term_rec[2] += tf  # cf
                        self.inverted_idx[term] = term_rec
                    else:  # new word - new term
                        self.inverted_idx[term] = [[tweet_id], 1, tf, None]

                else:  # existing term - update term parameters
                    if term.lower() in self.inverted_idx:
                        term = term.lower()
                    term_rec = self.inverted_idx[term]
                    term_rec[0].append(tweet_id)
                    term_rec[1] += 1  # df
                    term_rec[2] += tf  # cf
                    self.inverted_idx[term] = term_rec

                # Add to posting file
                self.postingDict.update(
                    {(term, tweet_id): [tf / document.max_tf, 0]})  # TODO: remove indices

            except:
                print('problem with the following key {}'.format(term))

    def finish_index(self):
        # Collect cf=1 terms, will be removed from our corpus
        delete_list = filter(lambda term : self.inverted_idx[term][2] <= 1,self.inverted_idx)
        to_delete = set() # set((term,tweetID))
        for term in delete_list:
            if term.upper() not in self.upper_terms:
                tweet_id = self.inverted_idx[term][0][0]
                to_delete.add((term, tweet_id))

        N = len(self.document_dict)

        # Clean all cf=1 terms from posting files and term dict
        for term,tweet_id in to_delete:
            self.inverted_idx.pop(term)
            self.postingDict.pop((term,tweet_id))

        # Fix upper terms in fix list
        for upper_term in self.upper_terms:
            # Updating term dictionary for lower term
            lower_term = upper_term.lower()
            upper_record = self.inverted_idx[upper_term]
            lower_record = self.inverted_idx[lower_term]
            for i in range(3):
                lower_record[i] += upper_record[i]
            self.inverted_idx[lower_term] = lower_record
            # Updating posting files to lower term
            for tweet in self.inverted_idx[upper_term][0]:
                if (upper_term, tweet) not in self.postingDict: continue
                self.postingDict[(lower_term, tweet)] = self.postingDict[(upper_term, tweet)]
                self.postingDict.pop((upper_term, tweet))
            # Remove from term dictionary
            self.inverted_idx.pop(upper_term)

        # Calculate document |d| by wij in each posting for ranking
        for key in self.postingDict:
            term, tweet_id = key
            idf = self.inverted_idx[term][3]
            if idf is None:
                idf = (log10(N / self.inverted_idx[term][1])).real
                self.inverted_idx[term][3] = idf
            tf_ij = self.postingDict[key][0]
            w_ij = tf_ij * idf
            self.postingDict[key][1] = w_ij
            self.document_dict[tweet_id][1] += w_ij ** 2

        #save_obj(self.postingDict, file_path) # TODO: Where we save posting dict

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def load_index(self, fn):
        """
        Loads a pre-computed index (or indices) so we can answer queries.
        Input:
            fn - file name of pickled index.
        """
        fn = fn.replace('.pkl', '')
        self.inverted_idx,self.postingDict,self.document_dict = load_obj(fn)

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def save_index(self, fn):
        """
        Saves a pre-computed index (or indices) so we can save our work.
        Input:
              fn - file name of pickled index.
        """
        fn = fn.replace('.pkl', '')
        save_obj((self.inverted_idx, self.postingDict, self.document_dict), fn)

    # feel free to change the signature and/or implementation of this function 
    # or drop altogether.
    def _is_term_exist(self, term):
        """
        Checks if a term exist in the dictionary.
        """
        return term in self.inverted_idx

    # feel free to change the signature and/or implementation of this function 
    # or drop altogether.
    def get_term_posting_list(self, term):
        """
        Return the posting list from the index for a term.
        """
        tweet_ids = self.inverted_idx[term][0]
        all_postings = {}
        for tweetid in tweet_ids:
            all_postings[(term, tweetid)] = self.postingDict[(term, tweetid)]
        return all_postings

    def get_term_tweets_list(self, term):
        """
        Return the document list from the index for a term.
        {tweet_id: |d|}
        """
        doc_dict = {}
        tweet_ids = self.inverted_idx[term][0]
        for tweet in tweet_ids:
            doc_dict[tweet] = self.document_dict[tweet][1]
        return doc_dict

    def get_doc_list(self, docs):
        """
        Return the document dict {tweet_id: {term:tf}}
        """
        doc_dict = {}
        for tweet_id in docs:
            doc_dict[tweet_id] = self.document_dict[tweet_id][0]
        return doc_dict

    def get_term_records(self,term):
        return self.inverted_idx[term]