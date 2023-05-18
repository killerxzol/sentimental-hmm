import pandas as pd
from sklearn.model_selection import StratifiedKFold
from models.shmm.boost import AdaBoost
from models.shmm import utils


DATASETS = {
    0: '../../../data/prepared/polarity-dataset-v1.csv',
    1: '../../../data/prepared/rotten-imdb.csv'
}


def k_fold(data, k=3, clusters=None):
    scores = []
    kFold = StratifiedKFold(n_splits=k)
    for i, (train_index, test_index) in enumerate(kFold.split(data.iloc[:, 0], data.iloc[:, 1])):

        X_train, y_train = data.iloc[train_index, 0].to_list(), data.iloc[train_index, 1].to_list()

        ab = AdaBoost()
        ab.fit(X_train, y_train, M=len(clusters), clusters=clusters)

        X_test, y_test = data.iloc[test_index, 0].to_list(), data.iloc[test_index, 1].to_list()
        y_pred = ab.predict([utils.word_tokenize(x) for x in X_test])

        scores.append(utils.accuracy(y_test, y_pred))
        print(f'\n[{i+1}/{k}] accuracy: {scores[i]}')

    print(f'Clusters: {clusters}')
    print(f'Min accuracy: {min(scores)}')
    print(f'Max accuracy: {max(scores)}')
    print(f'\nCross validation: {sum(scores) / k}')


def main(dataset):
    data = pd.read_csv(dataset).sample(frac=1, random_state=22).reset_index(drop=True)
    k_fold(data, clusters=[50])


if __name__ == '__main__':
    main(dataset=DATASETS[0])
