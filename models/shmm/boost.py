import numpy as np
from datetime import datetime
from .lsa import LatentSemanticAnalysis
from .model import SentimentalHMM
from . import utils


class AdaBoost:

    def __init__(self):
        self.M = None
        self.G_M = []
        self.alphas = []

    def fit(self, X, y, M=10, texts=None, clusters=None):
        self.M = M
        self.G_M = []
        self.alphas = []

        if not clusters:
            clusters = np.random.randint(low=50, high=125, size=M)

        lsa = LatentSemanticAnalysis(texts if texts else X)

        X = [utils.word_tokenize(sentence) for sentence in X]

        for m in range(self.M):
            print(f'\n[{datetime.now().strftime("%H:%M:%S")}] fit: iteration: {m+1} | {self.M}')

            if m == 0:
                w_i = np.ones(len(y)) * 1 / len(y)
            else:
                w_i = self.update_weights(w_i, alpha_m, y, y_pred)

            train = lsa.fit_transform(X, method='minibatch', n_clusters=clusters[m])
            X_positive, y_positive, X_negative, y_negative = utils.polar_split(train, y)

            G_m = SentimentalHMM(order=2)
            G_m.fit(X_positive, y_positive, X_negative, y_negative, smoothing='one-count')
            self.G_M.append(G_m)

            y_pred = G_m.predict(X)
            error_m = self.compute_error(y, y_pred, w_i)

            print(f'\n[{datetime.now().strftime("%H:%M:%S")}] fit: predicting error: {np.around(error_m, 4)}')

            alpha_m = self.compute_alpha(error_m)
            self.alphas.append(alpha_m)

        print(f'\n[{datetime.now().strftime("%H:%M:%S")}] Ensemble weights: {np.around(self.alphas, 4)}')
        print(f'\n[{datetime.now().strftime("%H:%M:%S")}] Ensemble clusters: {clusters}')

    def predict(self, X):
        weak_pred = np.zeros((2, len(X)))
        for m in range(self.M):
            print(f'\n[{datetime.now().strftime("%H:%M:%S")}] predict: iteration: {m+1} | {self.M}')
            weak_pred += np.tile(self.alphas[m], (2, 1)) * self.G_M[m].predict(X, return_probs=True)
        y_pred = (weak_pred[0, :] >= weak_pred[1, :]) * 2 - 1
        return y_pred.tolist()

    @staticmethod
    def compute_error(y, y_pred, w_i):
        return sum(w_i * np.not_equal(y, y_pred).astype(int))

    @staticmethod
    def compute_alpha(error):
        return 0.5 * np.log((1 - error) / error)

    @staticmethod
    def update_weights(w_i, alpha, y, y_pred):
        w_i = w_i * np.exp(-alpha * y_pred * y)
        return w_i / sum(w_i)
