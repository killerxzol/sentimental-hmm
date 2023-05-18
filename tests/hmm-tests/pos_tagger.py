from models.hmm.model import HiddenMarkovModel
from nltk.corpus import treebank
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score


def flatten(lst):
    return [p for s in lst for p in s]


def tag_split(lst_of_lst):
    lst_of_words, lst_of_tags = [], []
    for lst in lst_of_lst:
        words, tags = zip(*lst)
        lst_of_words.append(words)
        lst_of_tags.append(tags)
    return lst_of_words, lst_of_tags


def treebank_train_test_split():
    sentences = treebank.tagged_sents(tagset='universal')
    train, test = train_test_split(sentences, train_size=0.8, test_size=0.2)
    return [*tag_split(train), *tag_split(test)]


def metrics(y_true, y_pred):
    y_true, y_pred = flatten(y_true), flatten(y_pred)
    print(f'f1-score: {f1_score(y_true, y_pred, average=None, labels=list(set(y_true)))}')
    print(f'Accuracy: {accuracy_score(y_true, y_pred)}')


def main():
    X_train, y_train, X_test, y_test = treebank_train_test_split()

    model = HiddenMarkovModel(order=2)
    model.fit(X_train, y_train, smoothing='one-count')
    y_pred = model.predict(X_test)

    metrics(y_test, y_pred)


if __name__ == '__main__':
    main()
