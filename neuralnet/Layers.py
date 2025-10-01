import cupy as cp
import numpy as np

from neuralnet.Features import Sigmoid, Softmax, Relu
from neuralnet.Layers_Features import Padding, PatchExtractor, Aggregation, KaimingUniform, XavierUniform


class Layer:
    def backward(self, loss_grad, optimizer):
        pass

    def forward(self, x, train):
        pass

    def init_weights(self):
        pass

    def export(self):
        pass


class MetaLayer(Layer):
    """
    Маркерный класс для слоёв, которые содержат вложенные подмодули.
    """
    pass


class Conv2D(Layer):
    def __init__(self, out_channels, input_dim, kernel_size=(3, 3), init_dict=None, trainable=True,
                 learn_params=None, prev=None, w=None, bias=None, input_need_shape=None, **kwargs):

        if not isinstance(init_dict, dict):
            self.init_dict = {"init_cls": KaimingUniform}
        else:
            self.init_dict = init_dict

        self.init_cls = self.init_dict["init_cls"](**self.init_dict)

        if isinstance(learn_params, dict):
            self.learn_params = {key: cp.asarray(value, dtype=cp.float32) for key, value in learn_params.items()}
        else:
            self.learn_params = {"lr": cp.asarray(0.001, dtype=cp.float32)}

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = Padding(kernel_size=kernel_size, **kwargs)
        self.patcher = PatchExtractor(kernel_size=kernel_size, **kwargs)
        self.input_dim = input_dim
        self.input_need_shape = input_need_shape

        if self.input_need_shape is None:
            self.input_need_shape = self.input_dim

        self.input = None
        self.next = None
        self.patches = None
        self.prev = prev
        self.trainable = trainable
        self.W = w
        self.bias = bias
        self.init_weights()

    def forward(self, x, train=False):

        x = x.reshape(x.shape[0], *self.input_need_shape)

        x = self.padding.forward(x)

        patches = self.patcher.forward(x, train)

        need_shape = self.patcher.patch_shape
        lin = patches @ self.W.T

        if self.bias is not False:
            lin += self.bias
        lin = lin.reshape(x.shape[0], need_shape[0], need_shape[1], self.out_channels)

        if train:
            self.patches = patches
            self.input = x

        return lin

    def backward(self, loss_grad, optimizer):

        grad_output = loss_grad.reshape(-1, self.out_channels)

        lin_grad = grad_output @ self.W  # (B*H*W, input_dim * kH * kW)

        patcher_grad = self.patcher.backward(lin_grad)

        grad_input = self.padding.backward(patcher_grad)

        grad_input = grad_input.reshape(grad_input.shape[0], *self.input_dim)

        if self.trainable:
            dW = grad_output.T @ self.patches

            optimizer.step(self.W, dW, self.learn_params)
            if self.bias is not False:
                db = grad_output.sum(axis=0)
                optimizer.step(self.bias, db, self.learn_params, is_bias=True)
        return grad_input

    def init_weights(self):

        if self.W is None:
            kH, kW = self.kernel_size
            fan_in = self.input_need_shape[-1] * kH * kW
            self.W = self.init_cls(fan_in, self.out_channels)
            self.W = self.W.T

        if self.bias is None:
            self.bias = cp.zeros((self.out_channels,))

        if self.bias is not False:
            self.bias = cp.array(self.bias, dtype=cp.float32)

        self.W = cp.array(self.W, dtype=cp.float32)

    def export(self):
        dicti = {"out_channels": self.out_channels, "trainable": self.trainable, "kernel_size": self.kernel_size,
                 "input_dim": self.input_dim, "input_need_shape": self.input_need_shape, "layer": Conv2D,
                 "learn_params": {key: value.get() for key, value in self.learn_params.items()}, "w": self.W.copy(),
                 "init_dict": self.init_dict}

        if self.bias is not False:
            dicti["bias"] = self.bias.copy()
        else:
            dicti["bias"] = False

        dicti = self.padding.export() | dicti

        return dicti


class Dense(Layer):

    def __init__(self, neurons, input_dim, learn_params=None, init_dict=None, trainable=True, alpha=0.0,
                 w=None, bias=None, prev=None, input_need_shape=None, **kwargs):

        if not isinstance(init_dict, dict):
            self.init_dict = {"init_cls": KaimingUniform}
        else:
            self.init_dict = init_dict

        self.init_cls = self.init_dict["init_cls"](**self.init_dict)

        if isinstance(learn_params, dict):
            self.learn_params = {key: cp.asarray(value, dtype=cp.float32) for key, value in learn_params.items()}
        else:
            self.learn_params = {"lr": cp.asarray(0.001, dtype=cp.float32)}

        self.alpha = cp.asarray(alpha, dtype=cp.float32)
        self.input = None
        self.next = None
        self.prev = prev
        self.input_dim = input_dim

        if isinstance(self.input_dim, int):
            self.input_dim = (self.input_dim,)

        if input_need_shape is None:
            self.input_need_shape = (np.prod(self.input_dim),)  # Разворачиваем
        else:
            if isinstance(input_need_shape, int):
                self.input_need_shape = (input_need_shape,)
            else:
                self.input_need_shape = input_need_shape

        self.count_neurons = neurons
        self.trainable = trainable
        self.W = w
        self.bias = bias
        self.init_weights()

    def forward(self, x, train=False):
        x = x.reshape(x.shape[0], *self.input_need_shape)
        lin = x @ self.W

        if self.bias is not False:
            lin += self.bias

        if train:
            self.input = x

        return lin

    def backward(self, loss_grad, optimizer):

        grad_input = (loss_grad @ self.W.T).reshape(loss_grad.shape[0], *self.input_dim)

        if self.trainable:
            dW = cp.dot(self.input.T, loss_grad)

            optimizer.step(self.W, dW, self.learn_params)
            if self.bias is not False:
                db = loss_grad.sum(axis=0)
                optimizer.step(self.bias, db, self.learn_params, is_bias=True)
        return grad_input

    def init_weights(self):
        fan_in = self.input_need_shape[0]
        if self.W is None:
            self.W = self.init_cls(fan_in, self.count_neurons)

        if self.bias is None:
            self.bias = cp.zeros((1, self.count_neurons))

        if self.bias is not False:
            self.bias = cp.array(self.bias, dtype=cp.float32)
        self.W = cp.array(self.W, dtype=cp.float32)

    def export(self):
        dicti = {"neurons": self.count_neurons, "trainable": self.trainable, "init_dict": self.init_dict,
                 "input_need_shape": self.input_need_shape, "layer": Dense, "input_dim": self.input_dim,
                 "learn_params": {key: value.get() for key, value in self.learn_params.items()}, "w": self.W.copy()}

        if self.bias is not False:
            dicti["bias"] = self.bias.copy()
        else:
            dicti["bias"] = False

        return dicti


class MultiHead(MetaLayer):
    def __init__(self, input_dim, heads, nn_class, optimizer, prev, concat_axis=False, **kwargs):
        init_heads = []

        self.input_dim = input_dim
        for head in heads:
            head[0]["input_dim"] = input_dim
            init_heads.append(nn_class(head, optimizer=optimizer, loss_func=None))

        self.heads = init_heads
        self.len_heads = cp.asarray(len(self.heads), dtype=cp.float32)
        self.concat_axis = concat_axis
        self.next = None
        self.prev = prev

    def forward(self, x, train=False):
        result = []
        for head in self.heads:
            result.append(head.forward(x, train=train))
            head.last_output_shape = result[-1].shape
        if self.concat_axis is not False:
            result = cp.concatenate(result, axis=self.concat_axis)

        return result

    def backward(self, loss_grad, optimizer=None):
        if self.concat_axis is not False:
            # Нужно "разрезать" loss_grad на части для каждой головы
            grads = []
            idx = 0
            for head in self.heads:
                head_out_shape = head.last_output_shape[self.concat_axis]
                # slice градиента для этой головы
                slices = [slice(None)] * loss_grad.ndim
                slices[self.concat_axis] = slice(idx, idx + head_out_shape)
                grads.append(loss_grad[tuple(slices)])
                idx += head_out_shape
        else:
            # Без конкатенации
            grads = loss_grad

        grad_sum = 0

        for i, head in enumerate(self.heads):
            grad_sum += head.backward(grads[i])
        return grad_sum / self.len_heads

    def export(self):
        dicti = {"heads": [nn.export() for nn in self.heads], "layer": MultiHead, "concat_axis": self.concat_axis,
                 "input_dim": self.input_dim}
        return dicti


class MultiAttentionWO(Layer):
    def __init__(self, input_dim, d_need_head, learn_params=None, trainable=True, w=None, prev=None,
                 input_need_shape=None, bias=False, init_dict=None, **kwargs):

        if not isinstance(init_dict, dict):
            self.init_dict = {"init_cls": XavierUniform}
        else:
            self.init_dict = init_dict

        self.init_cls = self.init_dict["init_cls"](**self.init_dict)

        if isinstance(learn_params, dict):
            self.learn_params = {key: cp.asarray(value, dtype=cp.float32) for key, value in learn_params.items()}
        else:
            self.learn_params = {"lr": cp.asarray(0.001, dtype=cp.float32)}

        self.input = None
        self.next = None
        self.prev = prev
        self.input_dim = input_dim
        self.d_need_head = d_need_head

        if isinstance(self.input_dim, int):
            self.input_dim = (self.input_dim,)

        if input_need_shape is None:
            if len(self.input_dim) == 1:
                # (d_in,) → один элемент с d_in признаками
                self.input_need_shape = (1, self.input_dim[0])
            else:
                # например (L, d_in), (H, W, C)
                # считаем, что последняя ось — признаки
                self.input_need_shape = (np.prod(self.input_dim[:-1]), self.input_dim[-1])
        else:
            if isinstance(input_need_shape, int):
                self.input_need_shape = (1, input_need_shape)
            else:
                self.input_need_shape = tuple(input_need_shape)

        if d_need_head is None:
            self.d_need_head = self.input_need_shape[-1]

        self.trainable = trainable

        self.W = w
        self.bias = bias
        self.init_weights()

    def forward(self, x, train=False):
        x = x.reshape(x.shape[0], *self.input_need_shape)
        lin = x @ self.W
        if self.bias is not False:
            lin += self.bias

        lin = lin.reshape(x.shape[0], *self.input_dim[:-1], self.d_need_head)

        if train:
            self.input = x
        return lin

    def backward(self, loss_grad, optimizer):
        loss_grad = loss_grad.reshape(loss_grad.shape[0], -1, self.d_need_head)
        next_grad = (loss_grad @ self.W.T).reshape(loss_grad.shape[0], *self.input_dim)

        if self.trainable:
            dW = cp.mean(self.input.transpose(0, 2, 1) @ loss_grad, axis=0)
            optimizer.step(self.W, dW, self.learn_params)
            if self.bias is not False:
                db = loss_grad.sum(axis=0)
                optimizer.step(self.bias, db, self.learn_params, is_bias=True)
        return next_grad

    def init_weights(self):
        d_model = self.input_need_shape[-1]

        if self.W is None:
            self.W = self.init_cls(d_model, self.d_need_head)

        if self.bias is None:
            self.bias = cp.zeros((1, self.d_need_head))
        if self.bias is not False:
            self.bias = cp.array(self.bias, dtype=cp.float32)
        self.W = cp.asarray(self.W, dtype=cp.float32)

    def export(self):
        dicti = {"trainable": self.trainable, "w": self.W.copy(), "d_need_head": self.d_need_head,
                 "layer": MultiAttentionWO, "input_dim": self.input_dim, "init_dict": self.init_dict,
                 "learn_params": {key: value.get() for key, value in self.learn_params.items()}}

        if self.bias is not False:
            dicti["bias"] = self.bias.copy()
        else:
            dicti["bias"] = False

        return dicti


class SelfAttention(Layer):
    def __init__(self, input_dim, learn_params=None, d_need_head=None, trainable=True, Wq=None, Wk=None,
                 Wv=None, prev=None, input_need_shape=None, init_dict=None, **kwargs):

        if not isinstance(init_dict, dict):
            self.init_dict = {"init_cls": XavierUniform}
        else:
            self.init_dict = init_dict
        self.init_cls = self.init_dict["init_cls"](**self.init_dict)

        if isinstance(learn_params, dict):
            self.learn_params = {key: cp.asarray(value, dtype=cp.float32) for key, value in learn_params.items()}
        else:
            self.learn_params = {"lr": cp.asarray(0.001, dtype=cp.float32)}

        self.input = None
        self.next = None
        self.act = Softmax()
        self.prev = prev
        self.input_dim = input_dim
        self.d_need_head = d_need_head
        self.Q = None
        self.V = None
        self.K = None
        self.after_act = None

        if isinstance(self.input_dim, int):
            # Одинарное число — это просто размерность признаков
            self.input_dim = (self.input_dim,)

        # Если нужная форма не указана явно
        if input_need_shape is None:
            if len(self.input_dim) == 1:
                # (d_model,) → считаем, что L=1, d_model = self.input_dim[0]
                self.input_need_shape = (1, self.input_dim[0])
            else:
                # например (H, W, C) или (L, d_model)
                # сворачиваем все, кроме последней оси, в "длину последовательности"
                self.input_need_shape = (np.prod(self.input_dim[:-1]), self.input_dim[-1])
        else:
            if isinstance(input_need_shape, int):
                # одно число — это размерность признаков, L=1
                self.input_need_shape = (1, input_need_shape)
            else:
                self.input_need_shape = tuple(input_need_shape)

        # если d_need_head не указан, берём размерность признаков
        if d_need_head is None:
            self.d_need_head = self.input_need_shape[-1]

        self.trainable = trainable

        self.Wq = Wq
        self.Wk = Wk
        self.Wv = Wv

        self.init_weights()

    def forward(self, x, train=False):
        x = x.reshape(x.shape[0], *self.input_need_shape)
        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv

        # pre_act = cp.einsum('bij,bkj->bik', Q, K) / cp.sqrt(self.d_need_head)
        pre_act = Q @ K.transpose(0, 2, 1) / cp.sqrt(self.d_need_head)
        after_act = self.act.forward(pre_act, train=train)

        if train:
            self.input = x
            self.Q = Q
            self.K = K
            self.V = V
        return (after_act @ V).reshape(-1, *self.input_dim[:-1], self.d_need_head)

    def backward(self, loss_grad, optimizer):
        # 1. dV и dA
        loss_grad = loss_grad.reshape(loss_grad.shape[0], *self.input_need_shape[:-1], self.d_need_head)
        dV = self.act.after_act.transpose(0, 2, 1) @ loss_grad
        dA = loss_grad @ self.V.transpose(0, 2, 1)

        # 2. Градиент softmax
        dScores = self.act.backward(dA)  # softmax

        # 3. dQ и dK
        dQ = (dScores @ self.K) / cp.sqrt(self.d_need_head)
        dK = (dScores.transpose(0, 2, 1) @ self.Q) / cp.sqrt(self.d_need_head)

        # 4. Градиенты весов
        input = self.input.transpose(0, 2, 1)
        dWq = (input @ dQ).mean(axis=0)
        dWk = (input @ dK).mean(axis=0)
        dWv = (input @ dV).mean(axis=0)

        # 5. Градиент по входу
        next_grad = dQ @ self.Wq.T + dK @ self.Wk.T + dV @ self.Wv.T

        if self.trainable:
            optimizer.step(self.Wq, dWq, self.learn_params)
            optimizer.step(self.Wk, dWk, self.learn_params)
            optimizer.step(self.Wv, dWv, self.learn_params)

        return next_grad.reshape(-1, *self.input_dim)

    def init_weights(self):
        d_model = self.input_need_shape[-1]

        if self.Wq is None:
            self.Wq = self.init_cls(d_model, self.d_need_head)
        if self.Wk is None:
            self.Wk = self.init_cls(d_model, self.d_need_head)
        if self.Wv is None:
            self.Wv = self.init_cls(d_model, self.d_need_head)

        self.Wq = cp.asarray(self.Wq, dtype=cp.float32)
        self.Wk = cp.asarray(self.Wk, dtype=cp.float32)
        self.Wv = cp.asarray(self.Wv, dtype=cp.float32)

    def export(self):
        dicti = {"trainable": self.trainable, "input_need_shape": self.input_need_shape, "layer": SelfAttention,
                 "Wq": self.Wq.copy(), "Wv": self.Wv.copy(), "Wk": self.Wk.copy(), "input_dim": self.input_dim,
                 "learn_params": {key: value.get() for key, value in self.learn_params.items()},
                 "init_dict": self.init_dict}

        return dicti


class ConvAttention(MetaLayer):
    def __init__(self, input_dim, nn_class, optimizer, prev=None, input_need_shape=None, reduction=2, inner=None,
                 agg_mode="GAP+GMP", learn_params=None, forward_weight=False, mode="Channel", kernel_size=(7, 7),
                 trainable=True, **kwargs):

        self.learn_params = learn_params
        self.forward_weight = forward_weight
        self.trainable = trainable
        self.mode = mode
        if mode == "Channel":
            self.agg_axis = (1, 2)
            channels = input_dim[-1]
        else:  # Spatial
            self.agg_axis = (3,)
            channels = 1
            if agg_mode == "GAP+GMP":
                channels += 1

        self.act = Sigmoid()

        if inner is not None:
            if mode == "Channel":
                inner[0]["input_dim"] = channels
            else:  # Spatial
                inner[0]["input_dim"] = (*input_dim[:-1], channels)
        else:
            if mode == "Channel":
                hidden = max(1, channels // reduction)
                inner = [
                    {"layer": Dense, "input_dim": channels, "neurons": hidden, "learn_params": learn_params,
                     "trainable": trainable},
                    {"layer": Relu},
                    {"layer": Dense, "neurons": input_dim[-1], "learn_params": learn_params,
                     "init_dict": {"init_cls": XavierUniform}, "trainable": trainable}]
            else:  # Spatial
                inner = [{"input_dim": (*input_dim[:-1], channels), "kernel_size": kernel_size, "out_channels": 1,
                          "layer": Conv2D, "learn_params": learn_params, "init_dict": {"init_cls": XavierUniform},
                          "trainable": trainable}]

        self.inner = nn_class(inner, optimizer=optimizer, loss_func=None)
        self.agg_mode = agg_mode
        self.pre_act = None
        self.reduction = reduction
        self.after_act = None
        self.input = None
        self.next = None
        self.prev = prev
        self.input_dim = input_dim
        self.input_need_shape = input_need_shape
        self.dx = None

        self.gap = Aggregation(self.agg_axis, agg_func="mean")
        self.gmp = Aggregation(self.agg_axis, agg_func="max")
        if self.input_need_shape is None:
            self.input_need_shape = self.input_dim

    def forward(self, x, train=False):
        B, H, W, C = x.shape

        if self.agg_mode in ("GAP", "GMP"):
            if self.agg_mode == "GAP":
                pooled = self.gap.forward(x, train)  # keepdims
            else:
                pooled = self.gmp.forward(x, train)  # keepdims

            if self.mode == "Channel":
                pooled = cp.squeeze(pooled)
            pre_act = self.inner.forward(pooled, train=train)

        else:  # "GAP+GMP"
            pooled_1 = self.gap.forward(x, train)  # keepdims
            pooled_2 = self.gmp.forward(x, train)

            if self.mode == "Channel":
                pooled = cp.concatenate([pooled_1, pooled_2], axis=0)
                concat_pre_act = self.inner.forward(pooled, train=train)
                pre_act = concat_pre_act[:B] + concat_pre_act[B:]
            else:
                pooled = cp.stack([pooled_1, pooled_2], axis=-1)
                pre_act = self.inner.forward(pooled, train=train)

        after_act = self.act.forward(pre_act, train=True)
        if train or self.forward_weight:
            self.input = x

        if self.mode == "Channel":
            after_act = after_act.reshape(B, 1, 1, -1)

        return after_act if self.forward_weight else x * after_act

    def backward(self, loss_grad, optimizer=None):
        B, H, W, C = loss_grad.shape

        if not self.forward_weight:
            dc_w = loss_grad * self.input
            after_act = self.act.after_act
            if self.mode == "Channel":
                dc_w = dc_w.sum(axis=self.agg_axis)
                after_act = after_act.reshape(B, 1, 1, -1)
            else:
                dc_w = dc_w.sum(axis=self.agg_axis, keepdims=True)
            dx = loss_grad * after_act

        else:
            dx = self.dx
            dc_w = loss_grad
            if self.mode == "Channel":
                dc_w = loss_grad.reshape(-1, loss_grad.shape[-1])

        dact = self.act.backward(loss_grad=dc_w)

        if self.agg_mode == "GAP+GMP" and self.mode == "Channel":
            dact = cp.concatenate([dact, dact], axis=0)

        dpool = self.inner.backward(dact)
        if self.agg_mode == "GAP":
            dx_2 = self.gap.backward(dpool)

        elif self.agg_mode == "GMP":
            dx_2 = self.gmp.backward(dpool)

        else:  # GAP + GMP
            if self.mode == "Channel":
                dx_2 = self.gap.backward(dpool[:B]) + self.gmp.backward(dpool[B:])
            else:
                dx_2 = self.gap.backward(dpool[:, :, :, 0]) + self.gmp.backward(dpool[:, :, :, 1])

        return dx + dx_2

    def export(self):
        dicti = {"input_need_shape": self.input_need_shape, "layer": ConvAttention, "agg_mode": self.agg_mode,
                 "reduction": self.reduction, "forward_weight": self.forward_weight, "trainable": self.trainable,
                 "inner": self.inner.export(), "mode": self.mode, "input_dim": self.input_dim,
                 "learn_params": self.learn_params}

        return dicti


class MultiConvAttentionWO(MetaLayer):
    def __init__(self, input_dim, nn_class, optimizer, d_need_head, learn_params=None, trainable=True, prev=None,
                 input_need_shape=None, kernel_size=(1, 1), mode="Channel", inner=None, **kwargs):

        self.kernel_size = kernel_size
        self.learn_params = learn_params
        self.mode = mode
        self.prev = prev
        self.trainable = trainable
        self.d_need_head = d_need_head
        self.input_dim = input_dim

        if mode == "Channel":
            if inner is None:
                inner = [{"layer": MultiAttentionWO, "d_need_head": d_need_head, "learn_params": learn_params,
                          "trainable": trainable, "input_need_shape": input_need_shape,
                          "init_dict": {"init_cls": XavierUniform}}, {"layer": Sigmoid}]
            self.agg_axis = (1, 2)
        else:
            if "pooling_func" in kwargs:
                del kwargs["pooling_func"]
            if "out_channels" in kwargs:
                del kwargs["out_channels"]

            if inner is None:
                inner = [{"layer": Conv2D, "out_channels": d_need_head, "learn_params": learn_params,
                          "trainable": trainable, "input_need_shape": input_need_shape,
                          "init_dict": {"init_cls": XavierUniform},
                          "kernel_size": kernel_size}, {"layer": Sigmoid}]
            self.agg_axis = (3,)

        inner[0]["input_dim"] = input_dim
        self.inner = nn_class(inner, optimizer=optimizer, loss_func=None)
        self.channel_weights = None

    def forward(self, x, train=False):
        channel_weights = self.inner.forward(x, train=train)
        if train:
            self.channel_weights = channel_weights
        return self.prev.heads[0].layer_list[0].input * channel_weights

    def backward(self, loss_grad, optimizer=None):
        dx = loss_grad * self.channel_weights
        dc_w = loss_grad * self.prev.heads[0].layer_list[0].input

        if not self.channel_weights.shape == dc_w.shape:
            dc_w = dc_w.sum(axis=self.agg_axis, keepdims=True)

        loss_grad = self.inner.backward(dc_w)
        for head in self.prev.heads:
            head.layer_list[0].dx = dx

        return loss_grad.reshape(-1, *self.input_dim)

    def export(self):
        dicti = {"layer": MultiConvAttentionWO, "mode": self.mode, "d_need_head": self.d_need_head,
                 "inner": self.inner.export(), "input_dim": self.input_dim, "trainable": self.trainable,
                 "learn_params": self.learn_params, "kernel_size": self.kernel_size}
        return dicti
