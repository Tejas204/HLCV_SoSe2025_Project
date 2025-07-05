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
        max_pooling = False,
        model_name = "google/vit-base-patch16-224"

    ),
)

cnn_experiment_2 = dict(
    name = 'CNN_Experiment_2',

    model_args = dict(
        input_size = 3,
        num_classes = 10,
        hidden_layers = [128, 512, 512],
        activation = nn.ReLU,
        norm_layer = False,
        drop_prob = 0.4,
        max_pooling = True,
        model_name = "google/vit-base-patch16-224"
    ),
)