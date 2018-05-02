# tweets
path = '../data/tweets_test.csv'

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

    'key_label'     : 2,
    'key_text'      : 5,

    'regex'         : True,

    'embedding_path': '../glove/glove.twitter.27B.50d.txt',
    'embedding_dim' : 50,
}
