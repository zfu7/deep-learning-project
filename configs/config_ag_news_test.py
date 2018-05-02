# usi news
test = '../data/ag-test.csv'

table = {
    '1': 0,
    '2': 1,
    '3': 2,
    '4': 3,
}

dataset_config = {
    'lowercase'     : True,
    'length'        : 100,
    'word_size'     : 50,

    'path'          : test,

    'table'         : table,

    'key_label'     : 0,
    'key_text'      : 2,

    'regex'         : False,

    'embedding_path': '../glove/glove.twitter.27B.50d.txt',
    'embedding_dim' : 50,
}