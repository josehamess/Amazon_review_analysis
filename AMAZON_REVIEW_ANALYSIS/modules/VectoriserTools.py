import numpy as np
import zipfile
from collections import Counter


class VectoriserTools():
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    

    def rating_chooser(self, vectorised_texts, aggregated_tfs, classifier, ratings_to_analyse):

        # selects only chosen ratings #
        # returns ratings and text list for specified ratings #

        if len(ratings_to_analyse) < 5:
            inds = np.array([], dtype=int)
            for rating in ratings_to_analyse:
              inds = np.append(inds, np.where(classifier[:, 0] == rating))
            classifier = classifier[inds, :]
            binary_texts = binary_texts[inds, :]
            vectorised_texts = vectorised_texts[inds, :]
            aggregated_tfs = aggregated_tfs[inds, :]
        
        return vectorised_texts, aggregated_tfs, classifier


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
                vocab_dict[ngram] = ngram_counts[ngram]
        vocab = np.array(list(vocab_dict.keys()))[np.argsort(list(vocab_dict.values()))]
        vocab = vocab[-keep_top_n:]
        id_to_word, word_to_id = self.dictionary_creation(vocab)
        
        return vocab, df_counts, ngram_counts, id_to_word, word_to_id
    

    def idf_calc(self, vocab, df_counts, num_texts):

        # calculates inverse document frequencies for vocab #
        # this is a measure of how well an ngram differentaites a text from the rest #
        # returns vector of idfs for vocab #

        idfs = np.zeros((1, len(vocab)))
        for i, ngram in enumerate(vocab):
            idfs[:, i] = np.log10(num_texts / df_counts[ngram])
        
        return idfs
    

    def tfidf_calc(self, aggregated_tfs, vocab, df_counts):

        # calcuates tfidf vectors for each text #
        # returns an array containing tfidf vectors for each text #

        idfs = self.idf_calc(vocab, df_counts, aggregated_tfs.shape[0])
        tfidf_vectors = np.zeros(aggregated_tfs.shape)
        for i in range(aggregated_tfs.shape[0]):
            tfidf_vectors[i, :] = aggregated_tfs[i, :] * idfs
        print(f'Number of vectorised reviews after vectorisation: {tfidf_vectors.shape[0]}')

        return tfidf_vectors
    

    def remove_blanks(self, vectorised_texts, aggregated_tfs, classifier):

        # removes texts with no vocab in #
        # returns cleaned vectorised text and classifier #

        inds = []
        for i in range(vectorised_texts.shape[0]):
            text = vectorised_texts[i, :]
            if len(text[text != 0]) > 0:
                inds.append(i)
        vectorised_texts = vectorised_texts[inds, :]
        aggregated_tfs = aggregated_tfs[inds, :]
        classifier = classifier[inds, :]

        return vectorised_texts, aggregated_tfs, classifier
    

    def get_embeddings(self, vocab, emb_size=300):

        # using premade glove embeddings for words #
        # returns vectorised texts and classifer #

        word_embs = np.zeros((len(vocab), emb_size))
        with zipfile.ZipFile("glove.840B.300d.zip") as z: 
            with z.open("glove.840B.300d.txt") as f:
                for line in f:
                    line = line.decode('utf-8') 
                    word = line.split()[0]
                    if word in vocab:
                        emb = np.array(line.strip('\n').split()[1:])
                        if 'name@domain.com' not in emb:
                            word_embs[np.where(vocab==word), :] += emb.astype(np.float32)
        
        return word_embs

    
    def binarise(self, texts, classifier, vocab):

        # creates binary of vectors for each text based off vocab #
        # returns array of vectorised texts and classification vector and dictionary #

        binarised_texts = np.zeros((len(texts), len(vocab)))
        aggregated_tfs = np.zeros((len(texts), len(vocab)))
        
        text_count = 0
        for text in texts:
            ngram_counts = Counter(text)
            for ngram in ngram_counts.keys():
                if ngram in vocab:
                    binarised_texts[text_count, np.where(vocab == ngram)[0][0]] = 1
                    aggregated_tfs[text_count, np.where(vocab == ngram)[0][0]] = ngram_counts[ngram] / len(text)
            text_count += 1
        
        binarised_texts, aggregated_tfs, classifier = self.remove_blanks(binarised_texts, 
                                                                        aggregated_tfs,
                                                                        classifier)

        return binarised_texts, aggregated_tfs, classifier
