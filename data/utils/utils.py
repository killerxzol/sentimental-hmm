import pandas as pd
from os import listdir


SENTENCE_BASED = {
    'rotten-imdb': ('raw/rotten-imdb/', ('plot.tok.gt9.5000', 'quote.tok.gt9.5000')),
    'polarity-dataset-v1': ('raw/polarity-dataset/v1/', ('rt-polarity.neg', 'rt-polarity.pos'))
}

TEXT_BASED = {
    'polarity-dataset-v2': 'raw/polarity-dataset/v2/',
    'acl-imdb': 'raw/acl-imdb/train/'
}


def load_raw(dataset):
    if dataset in SENTENCE_BASED:
        folder, files = SENTENCE_BASED[dataset]
        return _load_sentences(folder, files)
    elif dataset in TEXT_BASED:
        folder = TEXT_BASED[dataset]
        return _load_texts(folder)
    else:
        raise ValueError


def _load_sentences(folder_path, file_names):
    data = pd.DataFrame(columns=['sentence', 'label'])
    for ind, file_name in enumerate(file_names):
        with open(folder_path + file_name, 'r', encoding='utf-8') as file:
            for line in file:
                data.loc[len(data.index)] = [line.rstrip(), -1 if ind == 0 else 1]
    return data


def _load_texts(folder_path):
    data = pd.DataFrame(columns=['text', 'label'])
    for ind, label in enumerate(['neg', 'pos']):
        for file_name in listdir(folder_path + label):
            with open(folder_path + label + '/' + file_name, 'r', encoding='utf-8') as file:
                data.loc[len(data.index)] = [' '.join([line.strip() for line in file]), -1 if ind == 0 else 1]
    return data
