# input channels, output channels, kernel size, batch normalization, max pooling
model = 'char_res'

res_config = [
    [128,64, 2, 1],
    [64, 32, 2, 2],
    [32, 16, 2, 2],
    [16, 8, 2, 1],
]

fc_config = [
    [24, 0, False],
]


model_config = {
    'input'         : 70,
    'residual'      : res_config,
    'fc_layers'     : fc_config,
}

train_config = {
    'lr'            : 0.05,
    'momentum'      : 0.9,
    'epochs'        : 50,
    'batch'         : 100
}

file_config = {
    'pretrained'    : False,

    'model'         : 'pretrained/' + model,
    'loss'          : 'results/loss_' + model,
    'acc'           : 'results/acc_' + model,
}
