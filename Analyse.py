from Autoencoder import Autoencoder
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


class Analyse(Autoencoder):
    def __init__(self, vocab, encodings, scaler, classifier, vectorised_texts, df_count, word_to_id):
        super(Analyse, self).__init__(vector_len=len(vocab))
        self.vocab = vocab
        self.encodings = encodings
        self.scaler = scaler
        self.classifier = classifier
        self.vectorised_texts = vectorised_texts
        self.df_count = df_count
        self.word_to_id = word_to_id
    

    def encoding_constrainer(self, classified_encodings, class_num, x_lower, x_upper, y_lower, y_upper):

        # finds the number of encodings for a class within a square area #
        # returns the number of encodings for a class within a square area #

        spec_class = classified_encodings[classified_encodings[:, -1] == class_num]
        x_constrained_1 = spec_class[spec_class[:, 0] <= x_upper]
        x_constrained_2 = x_constrained_1[x_constrained_1[:, 0] > x_lower]
        xy_constrained_1 = x_constrained_2[x_constrained_2[:, 1] <= y_upper]
        xy_constrained_2 = xy_constrained_1[xy_constrained_1[:, 1] > y_lower]
        
        return xy_constrained_2.shape[0]
    

    def cluster_size_calc(self, granularity):

        # determines the number of texts that fall inside areas in the latent space #
        # returns an array of the number of encodings for each c;ass in each square of the 2D latent space #
        
        x_dir = np.arange(np.floor(np.min(self.encodings[:, 0])), 
                            np.ceil(np.max(self.encodings[:, 0]) + 1), granularity)
        y_dir = np.arange(np.floor(np.min(self.encodings[:, 1])), 
                            np.ceil(np.max(self.encodings[:, 1]) + 1), granularity)
        classified_encodings = np.append(self.encodings, self.classifier, axis=1)
        cluster_info_array = np.zeros((len(x_dir), len(y_dir), 7))

        for i, x_val in enumerate(x_dir[0:-1]):
            cluster_info_array[i, :, 0] = ((x_dir[i] + x_dir[i + 1]) / 2)
            for j, y_val in enumerate(y_dir[0:-1]):
                cluster_info_array[i, j, 1] = ((y_dir[j] + y_dir[j + 1]) / 2)
                for k in range(1, 6):
                    cluster_info_array[i, j, k + 1] = self.encoding_constrainer(classified_encodings, 
                                                                                k, 
                                                                                x_dir[i], 
                                                                                x_dir[i + 1], 
                                                                                y_dir[j], 
                                                                                y_dir[j + 1])
        
        return cluster_info_array
    

    def cluster_size_heatmap(self, cluster_info_array):

        # plots a 3D plot of the number of encodings in each square of the latent space #

        fig = plt.figure(figsize=(15, 15))
        plt.imshow(np.sum(cluster_info_array[:, :, 2:], axis=2).T, cmap='hot', interpolation='nearest')
        plt.show()
    

    def topic_extractor(self, encoding, num_topics):

        # extracts a number of the most common topics from an encoding #
        # returns list of words #

        decoding = self.create_decoding(self.scaler, encoding)
        top_words = np.squeeze(self.vocab[np.argsort(decoding)][:, -num_topics:])

        return top_words
      
    
    def get_largest_clusters(self, cluster_info_array, num_clusters):

      # get the clusters with most data points #
      # returns encodings of where these clusters are #

      largest_cluster_encodings = []
      summed_ratings = np.sum(cluster_info_array[:, :, 2:], axis=2)
      for i in range(num_clusters):
        ind = np.unravel_index(summed_ratings.argmax(), summed_ratings.shape)
        inds = [ind[0], ind[1]]
        encoding = [cluster_info_array[inds[0], inds[1], 0], cluster_info_array[inds[0], inds[1], 1]]
        largest_cluster_encodings.append((inds, encoding))
        summed_ratings[ind[0], ind[1]] = 0

      return largest_cluster_encodings
    

    def analyse_all_topics(self, num_topics, num_clusters, granularity):

      # returns information about topics in different clusters #
      # prints information and returns a list of all topics found in clusters #

      topic_list = np.array([])
      cluster_info_array = self.cluster_size_calc(granularity)
      largest_cluster_encodings = self.get_largest_clusters(cluster_info_array, num_clusters)
      total_reviews = np.sum(cluster_info_array[:, :, 2:])
      for i, encoding in enumerate(largest_cluster_encodings):
        topics = self.topic_extractor(encoding[1], num_topics)
        topic_list = np.append(topic_list, topics)
        total_in_cluster = np.sum(cluster_info_array[encoding[0][0], encoding[0][1], 2:])
        print('')
        print(f'Cluster {i} information:')
        print(f'Encoding:{encoding[1]}')
        print(f'Topics: {topics}')
        percentage = round(100 * (total_in_cluster / total_reviews), 4)
        print(f'Percentage of total reviews in cluster: {percentage}%')
        for j in range(1, 6):
          num_of_ratings = np.sum(cluster_info_array[encoding[0][0], encoding[0][1], j + 1])
          print(f'Percent {j} star: {round(100 * (num_of_ratings / total_in_cluster), 4)}%')
      
      return topic_list


    def topic_popularity_display(self, topic_list):

      # finds the number of topics that have appeared in clusters #
      # plots ba chart of topic popularity and returns topic counts #

      topics = np.unique(topic_list)
      popularity_dict = {}
      for topic in topics:
          popularity_dict[topic] = 100 * (self.df_count[topic] / len(self.classifier))
      topics = np.array(list(popularity_dict.keys()))[np.argsort(list(popularity_dict.values()))]
      topic_values = np.sort(np.array(list(popularity_dict.values())))
      plt.figure(figsize = (20, 6))
      plt.bar(topics[::-1], topic_values[::-1])
      plt.xlabel('Topics')
      plt.ylabel('Percentage of reviews with topic in')
      plt.xticks(rotation='vertical')
      plt.grid()
      plt.show()
    

    def topic_rating_distribution(self, topic_list):
      
      # calculates proportion of ratings for topics in extracted topic list #
      # returns pie charts of the proportion of ratings for a particular topic #
      print('Star rating distribution for each topic (rated 1 star to 5 star)')
      print('')
      colour_dict = {'1':'r', '2':'tab:orange', '3':'y', '4':'m', '5':'g'}
      grid_edge_len_x = 4
      grid_edge_len_y = int(np.ceil(len(topic_list) / 4))
      fig, axs = plt.subplots(grid_edge_len_y, grid_edge_len_x, figsize=(20, 20), facecolor='white')
      axs_vals = [0, 0]
      for i, topic in enumerate(topic_list):
        if i % grid_edge_len_x == 0 and i != 0:
          axs_vals[0] += 1
          axs_vals[1] = 0
        ngram_idx = np.where(self.vocab == topic)[0]
        review_inds = np.where(self.vectorised_texts[:, ngram_idx] > 0)
        class_counts = Counter(self.classifier[review_inds])
        counts = list(class_counts.values())
        classes = list(class_counts.keys())
        colours = []
        for class_ in classes:
          colours.append(colour_dict[str(class_)])
        axs[axs_vals[0], axs_vals[1]].set_title(f'"{topic}" in {review_inds[0].shape[0]} reviews')
        axs[axs_vals[0], axs_vals[1]].pie(counts, labels=classes, colors=colours)
        axs_vals[1] += 1
      plt.show()