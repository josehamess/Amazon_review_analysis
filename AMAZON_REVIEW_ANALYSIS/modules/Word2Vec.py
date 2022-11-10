import numpy as np
from modules.TextCleaner import TextClean
from modules.VectoriserTools import VectoriserTools
from gensim.models import Word2Vec, Phrases


class Word2Vec_(TextClean, VectoriserTools):
    def __init__(self, stop_words, verbose, phrase_lens, vocab_size, word2vec_vars, **kwargs):
        super().__init__(stop_words, verbose, phrase_lens, **kwargs)
        self.stop_words = stop_words
        self.verbose = verbose
        self.phrase_lens = phrase_lens
        self.vocab_size = vocab_size
        self.word2vec_vars = word2vec_vars
        self.min_count = word2vec_vars['min_count']
        self.window_size = word2vec_vars['window_size']
        self.emb_size = word2vec_vars['emb_size']
    

    def create_sentences(self, text_list):

        # cleans and creates sentences from texts #
        # returns list of sentences #


        text_list = self.formatting_cleaner(text_list)
        text_list = self.bracket_removal(text_list)
        sentence_list = self.sentence_splitter(text_list)
        sentence_list = self.punctuation_removal(sentence_list)
        sentence_list = self.decapitalise(sentence_list)
        sentence_list, _, _ = self.stop_word_removal(sentence_list)
        
        return sentence_list


    def create_embeddings_gensim(self, texts, vocab):

        # create word embeddings using gensim #
        # returns embeddings array #

        print('')
        print('... Creating embeddings ...')
        print('')
        sentences = self.create_sentences(texts)
        if sum(self.phrase_lens) > 1.5:
            bigram_transformer = Phrases(sentences)
            sentences = bigram_transformer[sentences]
        model = Word2Vec(sentences=sentences, 
                        vector_size=self.emb_size, 
                        window=self.window_size, 
                        min_count=self.min_count, 
                        workers=4)
        all_embeddings = model.wv
        embeddings = np.zeros((len(vocab), all_embeddings[0].shape[0]))
        for word in model.wv.key_to_index:
            if word in vocab:
                embeddings[np.where(vocab == word)[0][0], :] = all_embeddings[word]

        return embeddings
    

    def vectorise_reviews(self, embeddings, binarised_reviews):

        # vectorises binarised texts using embeddings #
        # returns vectorised reviews #

        vectorised_reviews = np.zeros((binarised_reviews.shape))
        for i in range(binarised_reviews.shape[0]):
            review = binarised_reviews[i, :]
            vectorised_reviews[i, :] = binarised_reviews[i, :] / len(review[review == 1])
        
        vectorised_reviews = np.dot(vectorised_reviews, embeddings)
        print(f'Size of embeddings: {embeddings[0].shape[0]}')
        print(f'Number of vectorised reviews after vectorisation: {vectorised_reviews.shape[0]}')

        return vectorised_reviews






        

