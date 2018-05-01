# input channels, output channels, kernel size, batch normalization, max pooling
res_config = [
    [4,  8,  2, 1],
    [8,  16, 2, 2],
    [16, 32, 2, 2],
    [32, 64, 2, 1],
]

fc_config = [
    [192, 0, False],
]

model_config = {
    'input'         : 70,
    'residual'      : res_config,
    'fc_layers'     : fc_config,
}

train_config = {
    'lr'            : 0.01,
    'momentum'      : 0.9,
    'epochs'        : 100,
    'batch'         : 100
}