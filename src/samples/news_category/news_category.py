import pandas as pd

train = pd.read_csv('train_data.csv')
test = pd.read_csv('test_data.csv')

from konlpy.tag import Okt

tokenizer = Okt()


def get_length(title):
    return len(title)


train['title_len'] = train['title'].apply(get_length)
