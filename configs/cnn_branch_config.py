from functools import partial
import torch
import torch.nn as nn



cnn_experiment_1 = dict(
    name = 'CNN_Experiment_1',

    model_args = dict(
        input_size = 3,
        num_classes = 10,
        hidden_layers = [128, 512, 512],
        activation = nn.ReLU,
        norm_layer = False,
        drop_prob = 0.4,
    ),
)