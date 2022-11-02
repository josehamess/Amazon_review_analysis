import numpy as np
import zipfile
from collections import Counter


class LanguageVectoriser():
    def __init__(self, total_texts):
        self.total_texts = total_texts
    

    def rating_chooser(self, vectorised_texts, binary_texts, classifier, ratings_to_analyse):

        # selects only chosen ratings #
        # returns ratings and text list for specified ratings #

        if len(ratings_to_analyse) < 5:
            inds = np.array([], dtype=int)
            for rating in ratings_to_analyse:
              inds = np.append(inds, np.where(classifier[:, 0] == rating))
            classifier = classifier[inds, :]
            binary_texts = binary_texts[inds, :]
            vectorised_texts = vectorised_texts[inds, :]
        
        return vectorised_texts, binary_texts, classifier


    def n_gram_counter(self, texts):

        # counts how many times a specific ngram appears #
        # returns a dictionary of ngram counts #

        ngram_counts = {}
        for text in texts:
            for ngram in text:
                if ngram in ngram_counts:
                    ngram_counts[ngram] += 1
                else:
                    ngram_counts[ngram] = 1

        return ngram_counts


    def doc_freq_counter(self, texts):

        # counts the numbers of documents a specific ngram appears in #
        # returns a dictionary of counts for each ngram #

        df_counts = {}
        for text in texts:
            ngram_set = list(set(text))
            for ngram in ngram_set:
                if ngram in df_counts:
                    df_counts[ngram] += 1
                else:
                    df_counts[ngram] = 1
        
        return df_counts
    

    def dictionary_creation(self, vocab):

        # creates a id to ngram and word to id reference dictionary #
        # returns two dictioaries with reference ids for ngrams #

        id_to_word = dict(enumerate(vocab))
        word_to_id = dict()
        count = 0
        for ngram in vocab:
            word_to_id[ngram] = count
            count += 1

        return id_to_word, word_to_id


    def extract_vocab(self, texts, min_df, max_df, keep_top_n):

        # determines a vocab set that best represents the texts #
        # returns a list of ngrams #

        df_counts = self.doc_freq_counter(texts)
        ngram_counts = self.n_gram_counter(texts)
        vocab_dict = {}
        for ngram in df_counts:
            if df_counts[ngram] < max_df and df_counts[ngram] > min_df:
              if ' ' in ngram:
                vocab_dict[ngram] = ngram_counts[ngram]
        vocab = np.array(list(vocab_dict.keys()))[np.argsort(list(vocab_dict.values()))]
        id_to_word, word_to_id = self.dictionary_creation(vocab)
        
        return vocab[len(vocab) - keep_top_n:], df_counts, ngram_counts, id_to_word, word_to_id
    

    def idf_calc(self, vocab, df_counts):

        # calculates inverse document frequencies for vocab #
        # this is a measure of how well an ngram differentaites a text from the rest #
        # returns vector of idfs for vocab #

        idfs = np.zeros((1, len(vocab)))
        for i, ngram in enumerate(vocab):
            idfs[:, i] = np.log10(self.total_texts / df_counts[ngram])
        
        return idfs
    

    def tfidf_calc(self, vectorised_texts, idfs):

        # calcuates tfidf vectors for each text #
        # returns an array containing tfidf vectors for each text #

        tfidf_vectors = np.zeros(vectorised_texts.shape)
        for i in range(self.total_texts):
            tfidf_vectors[i, :] = vectorised_texts[i, :] * idfs

        return tfidf_vectors
    

    def remove_blanks(self, vectorised_texts, binary_texts, classifier):

        # removes texts with no vocab in #
        # returns cleaned vectorised text and film classifier vector #

        inds = []
        for i in range(vectorised_texts.shape[0]):
            text = vectorised_texts[i, :]
            if len(text[text != 0]) > 0:
                inds.append(i)
        vectorised_texts = vectorised_texts[inds, :]
        binary_texts = binary_texts[inds, :]
        classifier = classifier[inds, :]

        return vectorised_texts, binary_texts, classifier
    

    def get_embeddings(self, emb_size, vocab):

        # using premade glove embeddings for words #
        # returns vectorised texts and classifer #

        word_embs = np.zeros((len(vocab), emb_size))
        with zipfile.ZipFile("glove.840B.300d.zip") as z: 
            with z.open("glove.840B.300d.txt") as f:
                for line in f:
                    line = line.decode('utf-8') 
                    word = line.split()[0]
                    if word in vocab:
                        emb = np.array(line.strip('\n').split()[1:]).astype(np.float32)
                        word_embs[np.where(vocab==word), :] += emb
        
        return word_embs

    
    def vectorise(self, texts, classifier, vocab, df_counts, ngram_counts, TFIDF, glove, emb_size, ratings_to_analyse):

        # creates vectors of actors speech based of vocab list #
        # returns array of vectorised texts and character classification vector and dictionary #

        vectorised_texts = np.zeros((self.total_texts, len(vocab)))
        binary_texts = np.zeros((self.total_texts, len(vocab)))
        text_count = 0
        for text in texts:
            ngram_counts = Counter(text)
            for j, ngram in enumerate(vocab):
                if ngram in ngram_counts:
                    vectorised_texts[text_count, j] = ngram_counts[ngram]
                    binary_texts[text_count, j] = 1
            text_count += 1
        
        if TFIDF == True:
            idfs = self.idf_calc(vocab, df_counts)
            vectorised_texts = self.tfidf_calc(vectorised_texts, idfs)
        
        vectorised_texts, binary_texts, classifier = self.remove_blanks(vectorised_texts, binary_texts, classifier)
        vectorised_texts, binary_texts, classifier = self.rating_chooser(vectorised_texts, binary_texts, 
                                                                         classifier, ratings_to_analyse)

        if glove == True:
            word_embs = self.get_embeddings(emb_size, vocab)
            vectorised_texts = np.dot(binary_texts, word_embs)
        
        return vectorised_texts, classifier