import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler


class Autoencoder(nn.Module):
    def __init__(self, vector_len):
        super(Autoencoder, self).__init__()
        self.vector_len = vector_len
        self.encoder = nn.Sequential(
            nn.Linear(self.vector_len, int(self.vector_len / 2)),
            nn.LeakyReLU(),
            nn.Dropout(0.15),
            nn.Linear(int(self.vector_len / 2), 100),
            nn.Tanh(),
            nn.Linear(100, 2)
            #nn.Tanh(),
            #nn.Linear(20, 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 100),
            #nn.Tanh(),
            #nn.Linear(20, 100),
            nn.Tanh(),
            nn.Linear(100, int(self.vector_len / 2)),
            nn.Dropout(0.15),
            nn.LeakyReLU(),
            nn.Linear(int(self.vector_len / 2), self.vector_len)
        )


    def forward(self, x):

        # runs forward through network #
        # returns output from network #

        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

    def scaler(self, vectorised_text):

        # scale data #
        # return scaled array of texts #

        scaler = StandardScaler().fit(vectorised_text)
        scaled_text = scaler.transform(vectorised_text)

        return scaled_text, scaler
    

    def create_encodings(self, vectorised_text):

        # create encodings for all texts using encoder #
        # returns encodings in array #

        encodeloader = torch.utils.data.DataLoader(vectorised_text, batch_size=vectorised_text.shape[0], shuffle=False)
        with torch.no_grad():
            for batch in encodeloader:
                encodings = self.encoder(batch.clone().detach().requires_grad_(True).float()).numpy()
        
        return encodings
    

    def plot_encodings(self, encodings, classifier):

        # plots encoded texts #

        classified_encodings = np.append(encodings, classifier, axis=1)
        colours = ['c', 'k', 'r', 'b', 'y', 'g', 'm', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:orange']
        plt.figure(figsize=(10, 10))
        plt.grid()
        for class_ in np.unique(classifier):
            char_encodings = classified_encodings[classified_encodings[:, -1] == class_]
            plt.scatter(char_encodings[:, 0], char_encodings[:, 1], s=4, c=colours[class_], label=f'{class_} stars')
        plt.xlabel('Encoding 1')
        plt.ylabel('Encoding 2')
        plt.legend()
        plt.show()
    

    def create_decoding(self, scaler, encoding):

        # extracts words associated with particular encoding #
        # returns topics related to encoding and percentage of texts from each film #

        encoding = np.reshape(np.array(encoding), (1, 2))
        decodeloader = torch.utils.data.DataLoader(encoding, batch_size=1, shuffle=False)
        with torch.no_grad():
            for batch in decodeloader:
                scaled_decoding = self.decoder(batch.clone().detach().float()).numpy()
        decoding = scaler.inverse_transform(scaled_decoding)
        
        return decoding