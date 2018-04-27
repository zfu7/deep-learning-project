dataset_config = {
    'path'          : '../data/tweets.csv',
    'lowercase'     : True,
    'length'        : 100,
    'class_size'    : 2
}

# input channels, output channels, kernel size, batch normalization, max pooling
conv_config = [
    [70, 32, 7, True, True],
    [32, 32, 7, True, False],
    [32, 32, 3, True, False],
    [32, 32, 3, True, False],
    [32, 32, 3, True, True],
]

# input features, output features, if activate and dropout
fc_config = [
    [192, 256, True],
    [256, 256, True],
    [256, 2,   False],
]

model_config = {
    'dropout'       : 0.5,
    'conv_layers'   : conv_config,
    'fc_layers'     : fc_config,
}

train_config = {
    'lr'            : 0.1,
    'momentum'      : 0.9,
    'epochs'        : 100,
    'batch'         : 100
}