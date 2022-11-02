import regex as re


class TextClean():
    def __init__(self, stop_words, verbose, use_phrases, phrase_lens):
        self.stop_words = stop_words
        self.verbose = verbose
        self.use_phrases = use_phrases
        self.phrase_lens = phrase_lens


    def formatting_cleaner(self, text_list):

        # removes the new line token from speech #
        # returns the cleaned list of speech #

        cleaned_text_list = []
        for text in text_list:
            cleaned_text = ''
            for i in range(len(text)):
                if text[i] in ['\n', '<br>', '\t']:
                    cleaned_text += ' '
                else:
                    cleaned_text += text[i]
            cleaned_text_list.append(cleaned_text)
        
        return cleaned_text_list


    def bracket_removal(self, text_list):

        # removes all acting queues surrounded by brackets #
        # returns list of speech cleaned of bracketed text #

        cleaned_text_list = []
        for text in text_list:
            cleaned_text_list.append(re.sub(r'\([^()]*\)', '', text))
        
        return cleaned_text_list


    def punctuation_removal(self, text_list):

        # removes punctuation from actors speech #
        # returns new list of actors speech #

        cleaned_text_list = []
        for text in text_list:
            new_text = ''
            for i in range(len(text)):
                if text[i] not in ['"', ',', '.', '?', '!', '(', ')', '*', "'", '-', ':', '/', '£']:
                    new_text += text[i]
            cleaned_text_list.append(new_text)
        
        return cleaned_text_list


    def decapitalise(self, text_list):

        # removes capital letters from speech #
        # returns clean list of actors speech #

        decap_dict = {'A':'a', 'B':'b', 'C':'c', 'D':'d', 'E':'e', 'F':'f', 'G':'g', 'H':'h', 'I':'i', 'J':'j',
                        'K':'k', 'L':'l', 'M':'m', 'N':'n', 'O':'o', 'P':'p', 'Q':'q', 'R':'r', 'S':'s', 'T':'t',
                        'U':'u', 'V':'v', 'W':'w', 'X':'x', 'Y':'y', 'Z':'z'}
        cleaned_text_list = []
        for text in text_list:
            new_text = ''
            for i in range(len(text)):
                if re.match(r'[A-Z]', text[i]):
                    new_text += decap_dict[text[i]]
                else:
                    new_text += text[i]
            cleaned_text_list.append(new_text)
        
        return cleaned_text_list
    

    def phrase_splitter(self, words, phrase_len):

        # takes words and turns into phrases #
        # returns a list of phrases #

        phrases = []
        for i in range(len(words) - phrase_len):
            if i >= 0:
                phrase = ''
                for j in range(phrase_len):
                    #if words[i + j] not in self.stop_words:
                    if j == (phrase_len - 1):
                        phrase += words[i + j]
                    else:
                        phrase += f'{words[i + j]} '
                if len(phrase) > 0 and phrase not in self.stop_words:
                    if phrase[-1] == ' ':
                        phrase = phrase[0:-1]
                    elif phrase[0] == ' ':
                        phrase = phrase[1:]
                    phrases.append(phrase)
        
        return phrases


    def stop_word_removal(self, stop_words, text_list):

        # removes stop words from text #
        # returns text in list format without stop words present #

        split_text_list = []
        word_count = 0
        stop_word_count = 0
        for text in text_list:
            if len(text) > 0:
                ngrams = text.split(' ')
                ngrams_no_stop = []
                for ngram in ngrams:
                    if ngram not in self.stop_words:
                        ngrams_no_stop.append(ngram)
                if self.use_phrases == True:
                    all_phrases = []
                    for phrase_len in self.phrase_lens:
                        all_phrases += self.phrase_splitter(ngrams_no_stop, phrase_len)
                new_split_text = []
                for ngram in all_phrases:
                    word_count += 1
                    if ngram not in self.stop_words and len(ngram) > 0:
                        new_split_text.append(ngram)
                    else:
                        stop_word_count += 1
                split_text_list.append(new_split_text)

        return split_text_list, word_count, stop_word_count


    def clean_up(self, text_list, classifier, ratings_to_analyse, verbose):

        # runs groups of functions for cleaning up text #
        # returns the cleaned up list of actor speech #

        text_list = self.formatting_cleaner(text_list)
        text_list = self.bracket_removal(text_list)
        text_list = self.punctuation_removal(text_list)
        text_list = self.decapitalise(text_list)
        text_list, word_count, stop_word_count = self.stop_word_removal(self.stop_words, text_list)

        if verbose == True:
            print(f'Number of lines from character: {len(text_list)}')
            print(f'Percentage of words removed using stop word list: {round(100 * (stop_word_count / word_count), 1)}%')
            print(f'Average length of line after cleaning: {round(word_count / len(text_list), 1)}')

        return text_list, classifier