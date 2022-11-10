from modules.Word2Vec import Word2Vec_
from modules.VectoriserTools import VectoriserTools


class Vectoriser(Word2Vec_, VectoriserTools):
    def __init__(self, stop_words, verbose, phrase_lens, 
                word2vec_vars, vocab_extractor_vars):
        super().__init__(stop_words=stop_words, 
                        verbose=verbose,
                        phrase_lens=phrase_lens,
                        vocab_size=vocab_extractor_vars['keep_top_n'],
                        word2vec_vars=word2vec_vars)
        self.vocab_extractor_vars = vocab_extractor_vars
        self.min_df = vocab_extractor_vars['min_df']
        self.max_df = vocab_extractor_vars['max_df']
        self.keep_top_n = vocab_extractor_vars['keep_top_n']


    def vectorise(self, cleaned_reviews, reviews, star_ratings, vectorise_method):

        # vectorises reviews using chosen vectorise method #
        # returns vectorised reviews with id dicts, embeddings and vocab #
        # extract vocab using doc frequency stats

        print('')
        print('... vectorising ...')

        vocab, df_counts, _, _, word_to_id_dict = self.extract_vocab(cleaned_reviews, 
                                                                    self.min_df,
                                                                    self.max_df,
                                                                    self.keep_top_n
                                                                    )

        print('')
        print(f'Size of vocab: {len(vocab)}')

        # create binary vectors using vocab
        binarised_reviews, aggregated_tfs, star_ratings = self.binarise(cleaned_reviews, star_ratings, vocab)

        # get word embeddings
        if vectorise_method == 'word2vec' or vectorise_method == 'glove':
            word2vec_vectoriser = Word2Vec_(stop_words=self.stop_words, 
                                            verbose=self.verbose,  
                                            phrase_lens=self.phrase_lens,
                                            vocab_size=self.keep_top_n,
                                            word2vec_vars=self.word2vec_vars
                                            )
            #if using custom word2vec
            if vectorise_method == 'word2vec':
                word_embs = self.create_embeddings_gensim(reviews, vocab)
            
            # if using pretrained glove
            elif vectorise_method == 'glove':
                word_embs = self.get_embeddings(vocab, self.emb_size)
            
            # vectorise reviews
            vectorised_reviews = word2vec_vectoriser.vectorise_reviews(word_embs, binarised_reviews)

        # if using tfidf vectors
        elif vectorise_method == 'TFIDF':
            vectorised_reviews = self.tfidf_calc(aggregated_tfs, vocab, df_counts)
            word_embs = None
        
        info_dict = {
                    'word_to_ids' : word_to_id_dict, 'embs' : word_embs, 
                    'df_counts' : df_counts, 'vocab' : vocab
                    }
        
        return vectorised_reviews, star_ratings, info_dict