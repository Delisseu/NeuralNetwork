from neuralnet.Features import Relu, Sigmoid, BCE, MSE
from neuralnet.Layers import Conv2D, SelfAttention, MultiAttentionWO, ConvAttention, MultiHead, MultiConvAttentionWO
from neuralnet.Layers_Features import LayerNorm
from neuralnet.Optimizers import Adam, InverseSqrtScheduler
from neuralnet.core_gpu import NeuralNetwork

CNN = [
    {"input_dim": (100, 100, 1), "out_channels": 8, "layer": Conv2D, 'act': Relu, "lr": 0.0001, "norm": "bn",
     "pooling_shape": (2, 2), "pooling_stride": 1, "bias": False},
    {"out_channels": 16, "layer": Conv2D, 'act': Relu, "lr": 0.0001, "norm": "bn", "pooling_shape": (2, 2),
     "pooling_stride": 1, "bias": False},
    {"out_channels": 1, "layer": Conv2D, 'act': Relu, "lr": 0.0005, "kernel_size": (1, 1), "norm": "bn",
     "bias": False},
    {'neurons': 512, 'act': Relu, "lr": 0.0005, "norm": "bn", "bias": False},
    {'neurons': 64, 'act': Relu, "lr": 0.0005, "norm": "bn", "bias": False},
    {'neurons': 8, 'act': Relu, "lr": 0.0005},
    {'neurons': 1, 'act': Sigmoid, "lr": 0.0005}]

net1 = NeuralNetwork(CNN, BCE(), Adam(scheduler=InverseSqrtScheduler()))

selfAttention = [
    {"input_dim": (100, 100, 1), "out_channels": 8, "layer": Conv2D, 'act': Relu, "lr": 0.0001, "norm": "bn",
     "pooling_shape": (2, 2), "pooling_stride": 1},
    {"out_channels": 8, "layer": Conv2D, 'act': Relu, "lr": 0.0001, "norm": "bn"},
    {"out_channels": 8, "layer": Conv2D, 'act': Relu, "lr": 0.0001, "norm": "bn"},
    {"out_channels": 4, "layer": Conv2D, 'act': Relu, "lr": 0.0001, "kernel_size": (1, 1), "norm": "bn"},
    {"layer": SelfAttention, "lr": 0.0001},
    {"layer": LayerNorm, "n_lr": 0.0001, "norm": "ln"},
    {'neurons': 128, 'act': Relu, "lr": 0.0001, "norm": "bn"},
    {'neurons': 64, 'act': Relu, "lr": 0.0001, "norm": "bn"},
    {'neurons': 8, 'act': Relu, "lr": 0.0001},
    {'neurons': 1, 'act': Sigmoid, "lr": 0.0001}]

net2 = NeuralNetwork(selfAttention, BCE(), Adam(scheduler=InverseSqrtScheduler()))

MultiSelfAttention = [
    {"input_dim": (100, 100, 1), "out_channels": 8, "layer": Conv2D, 'act': Relu, "lr": 0.0001, "norm": "bn"},
    {"out_channels": 8, "layer": Conv2D, 'act': Relu, "lr": 0.0001, "norm": "bn"},
    {"out_channels": 8, "layer": Conv2D, 'act': Relu, "lr": 0.0001, "norm": "bn"},
    {"out_channels": 4, "layer": Conv2D, 'act': Relu, "lr": 0.0001, "kernel_size": (1, 1), "norm": "bn"},
    {"layer": MultiHead, "concat_axis": -1, "heads": [[{"layer": SelfAttention, "lr": 0.0001, "d_need_head": 4}],
                                                      [{"layer": SelfAttention, "lr": 0.0001, "d_need_head": 4}]]},
    {"lr": 0.0001, "d_need_head": 4, "layer": MultiAttentionWO},
    {"layer": LayerNorm, "n_lr": 0.0001, "norm": "ln"},
    {'neurons': 128, 'act': Relu, "lr": 0.0001, "norm": "bn"},
    {'neurons': 16, 'act': Relu, "lr": 0.0001},
    {'neurons': 1, 'act': Sigmoid, "lr": 0.0001}]

net3 = NeuralNetwork(MultiSelfAttention, BCE(), Adam(scheduler=InverseSqrtScheduler()))

CBAM_CNN = [
    {"input_dim": (100, 100, 1), "out_channels": 8, "layer": Conv2D, 'act': Relu, "lr": 0.0001, "norm": "bn",
     "pooling_shape": (2, 2), "pooling_stride": 1, "bias": False},
    {"out_channels": 16, "layer": Conv2D, 'act': Relu, "lr": 0.0001, "norm": "bn", "pooling_shape": (2, 2),
     "pooling_stride": 1, "bias": False},
    {"layer": ConvAttention, "lr": 0.0005, "mode": "Channel", "agg_mode": "GAP+GMP"},
    {"layer": ConvAttention, "lr": 0.0005, "mode": "Spatial", "agg_mode": "GAP+GMP"},
    {"out_channels": 1, "layer": Conv2D, 'act': Relu, "lr": 0.0005, "kernel_size": (1, 1), "norm": "bn",
     "bias": False},
    {'neurons': 512, 'act': Relu, "lr": 0.0005, "norm": "bn", "bias": False},
    {'neurons': 64, 'act': Relu, "lr": 0.0005, "norm": "bn", "bias": False},
    {'neurons': 8, 'act': Relu, "lr": 0.0005},
    {'neurons': 1, 'act': Sigmoid, "lr": 0.0005}]

net4 = NeuralNetwork(CBAM_CNN, BCE(), Adam(scheduler=InverseSqrtScheduler()))

MultiConvAttention = [
    {"input_dim": (100, 100, 1), "out_channels": 8, "layer": Conv2D, 'act': Relu, "lr": 0.0001, "norm": "bn",
     "pooling_shape": (2, 2), "pooling_stride": 1, "bias": False},
    {"out_channels": 16, "layer": Conv2D, 'act': Relu, "lr": 0.0001, "norm": "bn", "pooling_shape": (2, 2),
     "pooling_stride": 1, "bias": False},

    {"layer": MultiHead, "concat_axis": -1,
     "heads": [
         [{"layer": ConvAttention, "lr": 0.0005, "forward_weight": True, "mode": "Channel", "agg_mode": "GAP+GMP"}],
         [{"layer": ConvAttention, "lr": 0.0005, "forward_weight": True, "mode": "Channel", "agg_mode": "GAP+GMP"}]]},
    {"lr": 0.0005, "d_need_head": 16, "layer": MultiConvAttentionWO, "mode": "Channel"},

    {"layer": MultiHead, "concat_axis": -1,
     "heads": [
         [{"layer": ConvAttention, "lr": 0.0005, "forward_weight": True, "mode": "Spatial", "agg_mode": "GAP+GMP"}],
         [{"layer": ConvAttention, "lr": 0.0005, "forward_weight": True, "mode": "Spatial", "agg_mode": "GAP+GMP"}]]},
    {"lr": 0.0005, "d_need_head": 1, "layer": MultiConvAttentionWO, "mode": "Spatial", "kernel_size": (1, 1)},

    {"out_channels": 1, "layer": Conv2D, 'act': Relu, "lr": 0.0005, "kernel_size": (1, 1), "norm": "bn",
     "bias": False},

    {'neurons': 512, 'act': Relu, "lr": 0.0005, "norm": "bn", "bias": False},
    {'neurons': 64, 'act': Relu, "lr": 0.0005, "norm": "bn", "bias": False},
    {'neurons': 8, 'act': Relu, "lr": 0.0005},
    {'neurons': 1, 'act': Sigmoid, "lr": 0.0005}]

net5 = NeuralNetwork(MultiConvAttention, BCE(), Adam(scheduler=InverseSqrtScheduler()))

cls_head = [{'neurons': 64, 'act': Relu, "lr": 0.0005, "norm": "bn", "bias": False},
            {'neurons': 8, 'act': Relu, "lr": 0.0005},
            {'neurons': 1, 'act': Sigmoid, "lr": 0.0005}]

reg_head = [{'neurons': 64, 'act': Relu, "lr": 0.0005, "norm": "bn", "bias": False},
            {'neurons': 8, 'act': Relu, "lr": 0.0005},
            {'neurons': 2, "lr": 0.0005}]

multi_head = [
    {"input_dim": (100, 100, 1), "out_channels": 8, "layer": Conv2D, 'act': Relu, "lr": 0.0001, "norm": "bn",
     "pooling_shape": (2, 2), "pooling_stride": 1, "bias": False},
    {"out_channels": 16, "layer": Conv2D, 'act': Relu, "lr": 0.0001, "norm": "bn", "pooling_shape": (2, 2),
     "pooling_stride": 1, "bias": False},
    {"out_channels": 1, "layer": Conv2D, 'act': Relu, "lr": 0.0005, "kernel_size": (1, 1), "norm": "bn",
     "bias": False},
    {'neurons': 512, 'act': Relu, "lr": 0.0005, "norm": "bn", "bias": False},
    {"layer": MultiHead, "heads": [cls_head, reg_head]}]

net6 = NeuralNetwork(multi_head, [BCE(), MSE()], Adam(scheduler=InverseSqrtScheduler()))
