dataset_config = {
    'path': '../data/tweets.csv',
    'lowercase': True,
    'length': 100,
    'class_size': 2
}

model_config = {
    'feature_num': 70,
    'class_num':   2,
    'dropout': 0.5
}

train_config = {
    'lr': 0.05,
    'epochs': 10,
    'batch': 1
}
