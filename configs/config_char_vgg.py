# input channels, output channels, kernel size, batch normalization, max pooling
model = 'char_vgg'

conv_config = [
    [70, 128,3],
    [128,64, 3],
    [64, 64, 3],
    [64, 32, 3],
    [32, 32, 3],
]

# input features, output features, if activate and dropout
fc_config = [
    [96,  256, True],
    [256, 256, True],
    [256, 0,   False],
]

model_config = {
    'dropout'       : 0.5,
    'conv_layers'   : conv_config,
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
