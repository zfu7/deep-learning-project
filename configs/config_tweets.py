# tweets
path = '../data/tweets.csv'

table = {
    'realDonaldTrump': 0,
    'HillaryClinton': 1
}

dataset_config = {
    'lowercase'     : True,
    'length'        : 100,
    'word_size'     : 10,

    'path'          : path,

    'table'         : table,

    'key_label'     : 1,
    'key_text'      : 2,

    'regex'         : True,

    'embedding_path': '../glove/glove.twitter.27B.50d.txt',
    'embedding_dim' : 50,
}
