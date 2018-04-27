# usi news
path = '../data/uci-news-aggregator.csv'

table = {
    'b': 0,
    't': 1,
    'e': 2,
    'm': 3,
}

dataset_config = {
    'lowercase'     : True,
    'length'        : 100,
    'word_size'     : 50,

    'path'          : path,

    'table'         : table,

    'key_label'     : 4,
    'key_text'      : 1,

    'regex'         : False,

    'embedding_path': '../glove/glove.twitter.27B.50d.txt',
    'embedding_dim' : 50,
}