model = 'rcnn'

model_config = {
    'embedding_dim' : 50,
    'hidden_dim'    : 50,
    'context_dim'   : 50,
    'label_dim'     : 2,
    'lstm_layers'   : 2
}

dataset_config = {
    'data_path'     : '../data/tweets.csv',
    'embedding_path': '../glove/glove.twitter.27B.50d.txt',
    
    'embedding_dim' : 50
}

train_config = {
    'lr'            : 0.01,
    'momentum'      : 0.9,
    'epochs'        : 100,
    'batch'         : 100,
}

file_config = {
    'pretrained'    : False,

    'model'         : 'pretrained/' + model,
    'loss'          : 'results/loss_' + model,
    'acc'           : 'results/acc_' + model,
}