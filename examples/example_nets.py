from neuralnet.Features import Relu, Sigmoid, BCE, MSE
from neuralnet.Layers import Conv2D, SelfAttention, MultiAttentionWO, ConvAttention, MultiHead, MultiConvAttentionWO
from neuralnet.Layers_Features import LayerNorm, BatchNorm, Pooling, XavierUniform
from neuralnet.Optimizers import Adam, InverseSqrtScheduler
from neuralnet.core_gpu import NeuralNetwork

CNN = [
    {"input_dim": (100, 100, 1), "out_channels": 8, "layer": Conv2D, "learn_params": {"lr": 0.0001}, "bias": False},
    {"layer": BatchNorm, "learn_params": {"lr": 0.0001}}, {"layer": Relu}, {"layer": Pooling},

    {"out_channels": 16, "layer": Conv2D, "learn_params": {"lr": 0.0001}, "bias": False},
    {"layer": BatchNorm, "learn_params": {"lr": 0.0001}}, {"layer": Relu},

    {"out_channels": 1, "layer": Conv2D, "learn_params": {"lr": 0.0005}, "kernel_size": (1, 1), "bias": False},
    {"layer": BatchNorm, "learn_params": {"lr": 0.0005}}, {"layer": Relu},

    {'neurons': 512, "bias": False, "learn_params": {"l1": 0.1, "l2": 0.01, "wd": 0.001, "lr": 0.0001}},
    {"layer": BatchNorm, "learn_params": {"lr": 0.0005}}, {"layer": Relu},
    {'neurons': 64, "bias": False}, {"layer": BatchNorm, "learn_params": {"lr": 0.0005}}, {"layer": Relu},
    {'neurons': 8, "learn_params": {"lr": 0.0005}}, {"layer": Relu},
    {'neurons': 1, "learn_params": {"lr": 0.0005}, "init_dict": {"init_cls": XavierUniform}}, {"layer": Sigmoid}]

net1 = NeuralNetwork(CNN, BCE(), Adam(scheduler=InverseSqrtScheduler()))

selfAttention = [
    {"input_dim": (100, 100, 1), "out_channels": 8, "layer": Conv2D, "learn_params": {"lr": 0.0001}, "bias": False},
    {"layer": BatchNorm, "learn_params": {"lr": 0.0001}}, {"layer": Relu}, {"layer": Pooling},

    {"out_channels": 8, "layer": Conv2D, "learn_params": {"lr": 0.0001}, "bias": False},
    {"layer": BatchNorm, "learn_params": {"lr": 0.0001}}, {"layer": Relu}, {"layer": Pooling},

    {"out_channels": 8, "layer": Conv2D, "learn_params": {"lr": 0.0005}, "bias": False},
    {"layer": BatchNorm, "learn_params": {"lr": 0.0005}}, {"layer": Relu}, {"layer": Pooling},

    {"out_channels": 4, "layer": Conv2D, "learn_params": {"lr": 0.0005}, "kernel_size": (1, 1), "bias": False},
    {"layer": BatchNorm, "learn_params": {"lr": 0.0005}}, {"layer": Relu}, {"layer": Pooling},

    {"layer": SelfAttention, "learn_params": {"lr": 0.0005}},
    {"layer": LayerNorm, "learn_params": {"lr": 0.0005}},
    {'neurons': 128, "learn_params": {"l1": 0.1, "l2": 0.01, "wd": 0.001, "lr": 0.0001}, "bias": False},
    {"layer": BatchNorm, "learn_params": {"lr": 0.0005}}, {"layer": Relu},
    {'neurons': 64, "learn_params": {"lr": 0.0005}, "bias": False},
    {"layer": BatchNorm, "learn_params": {"lr": 0.0005}}, {"layer": Relu},
    {'neurons': 8, "learn_params": {"lr": 0.0005}}, {"layer": Relu},
    {'neurons': 1, "learn_params": {"lr": 0.0005}, "init_dict": {"init_cls": XavierUniform}}, {"layer": Sigmoid}]

net2 = NeuralNetwork(selfAttention, BCE(), Adam(scheduler=InverseSqrtScheduler()))

MultiSelfAttention = [
    {"input_dim": (100, 100, 1), "out_channels": 8, "layer": Conv2D, "learn_params": {"lr": 0.0001}, "bias": False},
    {"layer": BatchNorm, "learn_params": {"lr": 0.0001}}, {"layer": Relu}, {"layer": Pooling},

    {"out_channels": 8, "layer": Conv2D, "learn_params": {"lr": 0.0001}, "bias": False},
    {"layer": BatchNorm, "learn_params": {"lr": 0.0001}}, {"layer": Relu}, {"layer": Pooling},

    {"out_channels": 8, "layer": Conv2D, "learn_params": {"lr": 0.0005}, "bias": False},
    {"layer": BatchNorm, "learn_params": {"lr": 0.0005}}, {"layer": Relu}, {"layer": Pooling},

    {"out_channels": 4, "layer": Conv2D, "learn_params": {"lr": 0.0005}, "kernel_size": (1, 1), "bias": False},
    {"layer": BatchNorm, "learn_params": {"lr": 0.0005}}, {"layer": Relu}, {"layer": Pooling},

    {"layer": MultiHead, "concat_axis": -1,
     "heads": [[{"layer": SelfAttention, "learn_params": {"lr": 0.0005}, "d_need_head": 4}],
               [{"layer": SelfAttention, "learn_params": {"lr": 0.0005}, "d_need_head": 4}]]},
    {"learn_params": {"lr": 0.0005}, "d_need_head": 4, "layer": MultiAttentionWO},
    {"layer": LayerNorm, "learn_params": {"lr": 0.0005}},
    {'neurons': 128, "learn_params": {"l1": 0.1, "l2": 0.01, "wd": 0.001, "lr": 0.0001}, "bias": False},
    {"layer": BatchNorm, "learn_params": {"lr": 0.0005}}, {"layer": Relu},
    {'neurons': 64, "learn_params": {"lr": 0.0005}, "bias": False},
    {"layer": BatchNorm, "learn_params": {"lr": 0.0005}}, {"layer": Relu},
    {'neurons': 8, "learn_params": {"lr": 0.0005}}, {"layer": Relu},
    {'neurons': 1, "learn_params": {"lr": 0.0005}, "init_dict": {"init_cls": XavierUniform}}, {"layer": Sigmoid}]

net3 = NeuralNetwork(MultiSelfAttention, BCE(), Adam(scheduler=InverseSqrtScheduler()))

CBAM_CNN = [
    {"input_dim": (100, 100, 1), "out_channels": 8, "layer": Conv2D, "learn_params": {"lr": 0.0001}, "bias": False},
    {"layer": BatchNorm, "learn_params": {"lr": 0.0001}}, {"layer": Relu}, {"layer": Pooling},

    {"out_channels": 16, "layer": Conv2D, "learn_params": {"lr": 0.0005}, "bias": False},
    {"layer": BatchNorm, "learn_params": {"lr": 0.0005}}, {"layer": Relu},

    {"layer": ConvAttention, "learn_params": {"lr": 0.0005}, "mode": "Channel", "agg_mode": "GAP+GMP"},
    {"layer": ConvAttention, "learn_params": {"lr": 0.0005}, "mode": "Spatial", "agg_mode": "GAP+GMP"},

    {"out_channels": 1, "layer": Conv2D, "learn_params": {"lr": 0.0005}, "kernel_size": (1, 1), "bias": False},
    {"layer": BatchNorm, "learn_params": {"lr": 0.0005}}, {"layer": Relu},

    {'neurons': 512, "bias": False, "learn_params": {"l1": 0.1, "l2": 0.01, "wd": 0.001, "lr": 0.0001}},
    {"layer": BatchNorm, "learn_params": {"lr": 0.0005}}, {"layer": Relu},
    {'neurons': 64, "learn_params": {"lr": 0.0005}, "bias": False},
    {"layer": BatchNorm, "learn_params": {"lr": 0.0005}}, {"layer": Relu},
    {'neurons': 8, "learn_params": {"lr": 0.0005}}, {"layer": Relu},
    {'neurons': 1, "learn_params": {"lr": 0.0005}, "init_dict": {"init_cls": XavierUniform}}, {"layer": Sigmoid}]

net4 = NeuralNetwork(CBAM_CNN, BCE(), Adam(scheduler=InverseSqrtScheduler()))

MultiConvAttention = [
    {"input_dim": (100, 100, 1), "out_channels": 8, "layer": Conv2D, "learn_params": {"lr": 0.0001}, "bias": False},
    {"layer": BatchNorm, "learn_params": {"lr": 0.0001}}, {"layer": Relu}, {"layer": Pooling},

    {"out_channels": 16, "layer": Conv2D, "learn_params": {"lr": 0.0001}, "bias": False},
    {"layer": BatchNorm, "learn_params": {"lr": 0.0001}}, {"layer": Relu},

    {"layer": MultiHead, "concat_axis": -1,
     "heads": [
         [{"layer": ConvAttention, "learn_params": {"lr": 0.0005}, "forward_weight": True, "mode": "Channel",
           "agg_mode": "GAP+GMP"}],
         [{"layer": ConvAttention, "learn_params": {"lr": 0.0005}, "forward_weight": True, "mode": "Channel",
           "agg_mode": "GAP+GMP"}]]},
    {"learn_params": {"lr": 0.0005}, "d_need_head": 16, "layer": MultiConvAttentionWO, "mode": "Channel"},

    {"layer": MultiHead, "concat_axis": -1,
     "heads": [
         [{"layer": ConvAttention, "learn_params": {"lr": 0.0005}, "forward_weight": True, "mode": "Spatial",
           "agg_mode": "GAP+GMP"}],
         [{"layer": ConvAttention, "learn_params": {"lr": 0.0005}, "forward_weight": True, "mode": "Spatial",
           "agg_mode": "GAP+GMP"}]]},
    {"lr": 0.0005, "d_need_head": 1, "layer": MultiConvAttentionWO, "mode": "Spatial", "kernel_size": (1, 1)},

    {"out_channels": 1, "layer": Conv2D, "learn_params": {"lr": 0.0005}, "kernel_size": (1, 1), "bias": False},
    {"layer": BatchNorm, "learn_params": {"lr": 0.0005}}, {"layer": Relu},

    {'neurons': 512, "bias": False, "learn_params": {"l1": 0.1, "l2": 0.01, "wd": 0.001, "lr": 0.0001}},
    {"layer": BatchNorm, "learn_params": {"lr": 0.0005}}, {"layer": Relu},
    {'neurons': 64, "learn_params": {"lr": 0.0005}, "bias": False},
    {"layer": BatchNorm, "learn_params": {"lr": 0.0005}}, {"layer": Relu},
    {'neurons': 8, "learn_params": {"lr": 0.0005}}, {"layer": Relu},
    {'neurons': 1, "learn_params": {"lr": 0.0005}, "init_dict": {"init_cls": XavierUniform}}, {"layer": Sigmoid}]

net5 = NeuralNetwork(MultiConvAttention, BCE(), Adam(scheduler=InverseSqrtScheduler()))

cls_head = [{'neurons': 64, "learn_params": {"lr": 0.0005}, "bias": False},
            {"layer": BatchNorm, "learn_params": {"lr": 0.0005}}, {"layer": Relu},
            {'neurons': 8, "learn_params": {"lr": 0.0005}}, {"layer": Relu},
            {'neurons': 1, "learn_params": {"lr": 0.0005}, "init_dict": {"init_cls": XavierUniform}},
            {"layer": Sigmoid}]

reg_head = [{'neurons': 64, "learn_params": {"lr": 0.0005}, "bias": False},
            {"layer": BatchNorm, "learn_params": {"lr": 0.0005}}, {"layer": Relu},
            {'neurons': 8, "learn_params": {"lr": 0.0005}}, {"layer": Relu},
            {'neurons': 2, "learn_params": {"lr": 0.0005}, "init_dict": {"init_cls": XavierUniform}}]

multi_head = [
    {"input_dim": (100, 100, 1), "out_channels": 8, "layer": Conv2D, "learn_params": {"lr": 0.0001}, "bias": False},
    {"layer": BatchNorm, "learn_params": {"lr": 0.0001}}, {"layer": Relu}, {"layer": Pooling},

    {"out_channels": 16, "layer": Conv2D, "learn_params": {"lr": 0.0001}, "bias": False},
    {"layer": BatchNorm, "learn_params": {"lr": 0.0001}}, {"layer": Relu},

    {"out_channels": 1, "layer": Conv2D, "learn_params": {"lr": 0.0005}, "kernel_size": (1, 1), "bias": False},
    {"layer": BatchNorm, "learn_params": {"lr": 0.0005}}, {"layer": Relu},

    {'neurons': 512, "bias": False, "learn_params": {"l1": 0.1, "l2": 0.01, "wd": 0.001, "lr": 0.0001}},
    {"layer": BatchNorm, "learn_params": {"lr": 0.0005}}, {"layer": Relu},

    {"layer": MultiHead, "heads": [cls_head, reg_head]}]

net6 = NeuralNetwork(multi_head, [BCE(), MSE()], Adam(scheduler=InverseSqrtScheduler()))
