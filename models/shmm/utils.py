from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer


def word_tokenize(text, stop_words=True):
    analyzer = CountVectorizer(
        min_df=0.0, max_df=1.0, stop_words=stopwords.words('english') if stop_words else None
    ).build_analyzer()
    return analyzer(text)


def polar_split(X, targets):
    X_positive, y_positive, X_negative, y_negative = [], [], [], []
    for sentence, target in zip(X, targets):
        if sentence:
            x, y = zip(*sentence)
        else:
            x, y = [], []
        if target == 1:
            X_positive.append(x)
            y_positive.append(y)
        else:
            X_negative.append(x)
            y_negative.append(y)
    return X_positive, y_positive, X_negative, y_negative


def chunks(lst, n):
    k, m = divmod(len(lst), n)
    return (lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
