from datetime import datetime
from nltk.corpus import stopwords
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.utils.extmath import randomized_svd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


RANDOM_STATE = 22


class LatentSemanticAnalysis:

    def __init__(self, texts):
        self.vocabulary = None
        self._u, self._features = self._vectorize(texts)

    def fit(self, method, n_clusters):
        print(f'\n[{datetime.now().strftime("%H:%M:%S")}] LSA: clustering started (method={method}, clusters={n_clusters})')
        if method == 'standard':
            kmeans = KMeans(n_clusters=n_clusters, max_iter=500, random_state=RANDOM_STATE, n_init=10, verbose=False)
            clusters = kmeans.fit_predict(self._u)
        elif method == 'minibatch':
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, max_iter=500, random_state=RANDOM_STATE, n_init=10, verbose=False)
            clusters = kmeans.fit_predict(self._u)
        else:
            raise ValueError
        print(f'[{datetime.now().strftime("%H:%M:%S")}] LSA: clustering completed')
        self.vocabulary = dict(zip(self._features, clusters))

    def transform(self, X):
        X_trans = []
        for i, sentence in enumerate(X):
            X_trans.append([])
            for word in sentence:
                if word in self.vocabulary:
                    X_trans[i].append((word, self.vocabulary[word]))
        return X_trans

    def fit_transform(self, X, method='standard', n_clusters=45):
        self.fit(method, n_clusters=n_clusters)
        return self.transform(X)

    @staticmethod
    def _vectorize(texts):
        tdm_pipe = make_pipeline(
            CountVectorizer(min_df=5, max_df=0.9, stop_words=stopwords.words('english')),
            TfidfTransformer(norm='l2', use_idf=True)
        )
        tdm = tdm_pipe.fit_transform(texts)
        u, s, vt = randomized_svd(tdm.transpose(), n_components=800, random_state=RANDOM_STATE)
        normalizer = Normalizer(copy=False)
        normalizer.fit_transform(u)
        features = tdm_pipe[0].get_feature_names_out()
        return u, features
