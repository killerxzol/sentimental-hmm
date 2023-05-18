import numpy as np
import multiprocessing
from datetime import datetime
from . import utils
from ..hmm.model import HiddenMarkovModel


class SentimentalHMM:

    def __init__(self, order):
        self.p_model = HiddenMarkovModel(order=order)
        self.n_model = HiddenMarkovModel(order=order)

    def fit(self, X_positive, y_positive, X_negative, y_negative, smoothing=None):
        self.p_model.fit(X_positive, y_positive, smoothing=smoothing)
        self.n_model.fit(X_negative, y_negative, smoothing=smoothing)

    def predict(self, X, n_process=8, return_probs=False):
        print(f'\n[{datetime.now().strftime("%H:%M:%S")}] SHMM: predicting started (size: {len(X)})')
        positive_probs = self._multi_predict(X, self.p_model, n_process)
        negative_probs = self._multi_predict(X, self.n_model, n_process)
        print(f'[{datetime.now().strftime("%H:%M:%S")}] SHMM: predicting completed')
        if not return_probs:
            return (positive_probs >= negative_probs).astype(int) * 2 - 1
        return positive_probs, negative_probs

    def _multi_predict(self, X, model, n_process):
        X = utils.chunks(X, n_process)
        manager = multiprocessing.Manager()
        notepad = manager.dict()
        processes = []
        for k in range(n_process):
            processes.append(multiprocessing.Process(target=self._predict, args=(next(X), model, notepad, k)))
        for process in processes:
            process.start()
        for process in processes:
            process.join()
        probs = []
        for k in range(n_process):
            probs.extend(notepad[k])
        return np.array(probs)

    @staticmethod
    def _predict(X, model, notepad, k):
        notepad[k] = model.forward_probability(X, model.predict(X))
