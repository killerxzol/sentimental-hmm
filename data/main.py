from utils import utils


DATASETS = {
    0: 'rotten-imdb',
    1: 'polarity-dataset-v1',
    2: 'polarity-dataset-v2',
    3: 'acl-imdb'
}


def main(dataset):
    data = utils.load_raw(dataset)
    data.to_csv(f'prepared/{dataset}.csv', encoding='utf-8', index=False)


if __name__ == '__main__':
    main(dataset=DATASETS[0])
