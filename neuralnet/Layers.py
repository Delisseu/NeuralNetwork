import cupy as cp
import numpy as np

from neuralnet.Features import Sigmoid, Softmax, Relu
from neuralnet.Layers_Features import Padding, PatchExtractor, _gap_backward, \
    _gmp_backward, xavier_uniform, kaiming_uniform


class Layer:
    def backward(self, loss_grad, optimizer):
        pass

    def forward(self, x, train):
        pass

    def init_weights(self):
        pass

    def export(self):
        pass


class Conv2D(Layer):
    def __init__(self, out_channels, input_dim, kernel_size=(3, 3), init_func=kaiming_uniform,
                 regularization=False, alpha=0.0, trainable=True, lr=0.001, prev=None,
                 w=None, lamda=0.01, lamda_2=0.01, bias=None, input_need_shape=None, **kwargs):

        self.alpha = cp.asarray(alpha, dtype=cp.float32)
        self.init_func = kaiming_uniform
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
        self.lamda = cp.array(lamda, dtype=cp.float32)
        self.lamda_2 = cp.array(lamda_2, dtype=cp.float32)

        self.trainable = trainable
        self.learning_rate = cp.array(lr, dtype=cp.float32)
        self.W = w
        self.bias = bias

        self.regularization = regularization
        self.init_weights()

    def export(self):
        dicti = {"out_channels": self.out_channels, "lr": self.learning_rate, "regularization": self.regularization,
                 "trainable": self.trainable, "lamda": self.lamda.copy(), "lamda_2": self.lamda_2.copy(),
                 "w": self.W.copy(), "input_need_shape": self.input_need_shape, "kernel_size": self.kernel_size,
                 "layer": Conv2D, "input_dim": self.input_dim}

        if self.bias is not False:
            dicti["bias"] = self.bias.copy()
        else:
            dicti["bias"] = False

        dicti = self.padding.export() | dicti

        return dicti

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
            if self.regularization is not False:
                dW += self.regularization(self.W, self.lamda, self.lamda_2)  # (out_channels, input_dim * kH * kW)

            optimizer.step(self.W, dW, self.trainable, self.learning_rate)
            if self.bias is not False:
                db = grad_output.sum(axis=0)
                optimizer.step(self.bias, db, self.trainable, self.learning_rate)
        return grad_input

    def init_weights(self):

        if self.W is None:
            kH, kW = self.kernel_size
            fan_in = self.input_need_shape[-1] * kH * kW
            self.W = self.init_func(fan_in, self.out_channels, alpha=self.alpha)
            self.W = self.W.T

        if self.bias is None:
            self.bias = cp.zeros((self.out_channels,))

        if self.bias is not False:
            self.bias = cp.array(self.bias, dtype=cp.float32)

        self.W = cp.array(self.W, dtype=cp.float32)


class Dense(Layer):

    def __init__(self, neurons, input_dim, lr=0.001, regularization=False, init_func=kaiming_uniform,
                 trainable=True, lamda=0.01, lamda_2=0.01, alpha=0.0, w=None, bias=None, prev=None,
                 input_need_shape=None, **kwargs):

        self.alpha = cp.asarray(alpha, dtype=cp.float32)
        self.input = None
        self.init_func = init_func
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
        self.lamda = cp.array(lamda, dtype=cp.float32)
        self.lamda_2 = cp.array(lamda_2, dtype=cp.float32)
        self.learning_rate = cp.array(lr, dtype=cp.float32)
        self.W = w
        self.bias = bias
        self.regularization = regularization
        self.init_weights()

    def backward(self, loss_grad, optimizer):

        grad_input = (loss_grad @ self.W.T).reshape(loss_grad.shape[0], *self.input_dim)

        if self.trainable:
            dW = cp.dot(self.input.T, loss_grad)

            if self.regularization is not False:
                dW += self.regularization(self.W, self.lamda, self.lamda_2)

            optimizer.step(self.W, dW, self.trainable, self.learning_rate)
            if self.bias is not False:
                db = loss_grad.sum(axis=0)
                optimizer.step(self.bias, db, self.trainable, self.learning_rate)
        return grad_input

    def forward(self, x, train=False):
        x = x.reshape(x.shape[0], *self.input_need_shape)
        lin = x @ self.W

        if self.bias is not False:
            lin += self.bias

        if train:
            self.input = x

        return lin

    def init_weights(self):
        fan_in = self.input_need_shape[0]
        if self.W is None:
            self.W = self.init_func(fan_in, self.count_neurons, alpha=self.alpha)

        if self.bias is None:
            self.bias = cp.zeros((1, self.count_neurons))

        if self.bias is not False:
            self.bias = cp.array(self.bias, dtype=cp.float32)
        self.W = cp.array(self.W, dtype=cp.float32)

    def export(self):
        dicti = {"neurons": self.count_neurons, "lr": self.learning_rate.copy(),
                 "regularization": self.regularization,
                 "trainable": self.trainable, "lamda": self.lamda.copy(), "lamda_2": self.lamda_2.copy(),
                 "alpha": self.alpha.copy(), "w": self.W.copy(),
                 "input_need_shape": self.input_need_shape, "layer": Dense, "input_dim": self.input_dim}

        if self.bias is not False:
            dicti["bias"] = self.bias.copy()
        else:
            dicti["bias"] = False

        return dicti


class MultiHead(Layer):
    def __init__(self, input_dim, heads, nn_class, optimizer, prev, concat_axis=None, **kwargs):
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

    def export(self):
        dicti = {"heads": [nn.export() for nn in self.heads], "layer": MultiHead, "concat_axis": self.concat_axis,
                 "input_dim": self.input_dim}
        return dicti

    def forward(self, x, train=False):
        result = []
        for head in self.heads:
            result.append(head.forward(x, train=train))
            head.last_output_shape = result[-1].shape
        if self.concat_axis is not None:
            result = cp.concatenate(result, axis=self.concat_axis)

        return result

    def backward(self, loss_grad, optimizer):
        if self.concat_axis is not None:
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


class MultiAttentionWO(Layer):
    def __init__(self, input_dim, d_need_head, lr=0.001, trainable=True, w=None, prev=None,
                 input_need_shape=None, bias=False, **kwargs):

        self.input = None
        self.next = None
        self.prev = prev
        self.input_dim = input_dim
        self.d_need_head = d_need_head

        if input_need_shape is None:
            if len(input_dim) == 2:
                self.input_need_shape = (*input_dim, 1)
            else:
                self.input_need_shape = (np.prod(self.input_dim[:-1]), self.input_dim[-1])
        else:
            if isinstance(input_need_shape, int):
                self.input_need_shape = (input_need_shape, 1)
            else:
                self.input_need_shape = input_need_shape

        self.trainable = trainable
        self.learning_rate = cp.array(lr, dtype=cp.float32)

        self.W = w
        self.bias = bias
        self.init_weights()

    def backward(self, loss_grad, optimizer):
        loss_grad = loss_grad.reshape(loss_grad.shape[0], -1, self.d_need_head)
        next_grad = (loss_grad @ self.W.T).reshape(loss_grad.shape[0], *self.input_dim)

        if self.trainable:
            dW = cp.mean(self.input.transpose(0, 2, 1) @ loss_grad, axis=0)
            optimizer.step(self.W, dW, self.trainable, self.learning_rate)
            if self.bias is not False:
                db = loss_grad.sum(axis=0)
                optimizer.step(self.bias, db, self.trainable, self.learning_rate)
        return next_grad

    def forward(self, x, train=False):
        x = x.reshape(x.shape[0], *self.input_need_shape)
        lin = x @ self.W
        if self.bias is not False:
            lin += self.bias

        lin = lin.reshape(x.shape[0], *self.input_dim[:-1], self.d_need_head)

        if train:
            self.input = x
        return lin

    def init_weights(self):
        d_model = self.input_dim[-1]

        if self.W is None:
            self.W = xavier_uniform(d_model, self.d_need_head)

        if self.bias is None:
            self.bias = cp.zeros((1, self.d_need_head))
        if self.bias is not False:
            self.bias = cp.array(self.bias, dtype=cp.float32)
        self.W = cp.asarray(self.W, dtype=cp.float32)

    def export(self):
        dicti = {"lr": self.learning_rate.copy(), "trainable": self.trainable, "w": self.W.copy(),
                 "d_need_head": self.d_need_head, "layer": MultiAttentionWO, "input_dim": self.input_dim}

        if self.bias is not False:
            dicti["bias"] = self.bias.copy()
        else:
            dicti["bias"] = False

        return dicti


class SelfAttention(Layer):
    def __init__(self, input_dim, d_need_head=None, lr=0.001, trainable=True, Wq=None, Wk=None, Wv=None,
                 prev=None, input_need_shape=None, **kwargs):

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
            self.input_dim = (self.input_dim,)

        if input_need_shape is None:
            if len(input_dim) == 1:
                self.input_need_shape = (*input_dim, 1)
            else:
                self.input_need_shape = (np.prod(self.input_dim[:-1]), self.input_dim[-1])
        else:
            if isinstance(input_need_shape, int):
                self.input_need_shape = (input_need_shape, 1)
            else:
                self.input_need_shape = input_need_shape

        if d_need_head is None:
            self.d_need_head = self.input_need_shape[-1]

        self.trainable = trainable
        self.learning_rate = cp.array(lr, dtype=cp.float32)

        self.Wq = Wq
        self.Wk = Wk
        self.Wv = Wv

        self.init_weights()

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
            optimizer.step(self.Wq, dWq, True, self.learning_rate)
            optimizer.step(self.Wk, dWk, True, self.learning_rate)
            optimizer.step(self.Wv, dWv, True, self.learning_rate)

        return next_grad.reshape(-1, *self.input_dim)

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

    def init_weights(self):
        d_model = self.input_need_shape[-1]

        if self.Wq is None:
            self.Wq = xavier_uniform(d_model, self.d_need_head)
        if self.Wk is None:
            self.Wk = xavier_uniform(d_model, self.d_need_head)
        if self.Wv is None:
            self.Wv = xavier_uniform(d_model, self.d_need_head)

        self.Wq = cp.asarray(self.Wq, dtype=cp.float32)
        self.Wk = cp.asarray(self.Wk, dtype=cp.float32)
        self.Wv = cp.asarray(self.Wv, dtype=cp.float32)

    def export(self):
        dicti = {"lr": self.learning_rate.copy(), "trainable": self.trainable,
                 "input_need_shape": self.input_need_shape, "layer": SelfAttention,
                 "Wq": self.Wq.copy(), "Wv": self.Wv.copy(), "Wk": self.Wk.copy(), "input_dim": self.input_dim}

        return dicti


class ConvAttention(Layer):
    def __init__(self, input_dim, nn_class, optimizer, prev=None, input_need_shape=None, reduction=2, inner=None,
                 agg_mode="GAP+GMP", lr=0.001, forward_weight=False, mode="Channel", kernel_size=(7, 7), trainable=True,
                 **kwargs):

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
                    {"layer": Dense, "input_dim": channels, "neurons": hidden, "lr": lr, "trainable": trainable},
                    {"layer": Relu},
                    {"layer": Dense, "neurons": input_dim[-1], "lr": lr, "init_func": xavier_uniform,
                     "trainable": trainable}]
            else:  # Spatial
                inner = [{"input_dim": (*input_dim[:-1], channels), "kernel_size": kernel_size, "out_channels": 1,
                          "layer": Conv2D, "lr": lr, "init_func": xavier_uniform, "trainable": trainable}]

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

        if self.input_need_shape is None:
            self.input_need_shape = self.input_dim

    def forward(self, x, train=False):
        B, H, W, C = x.shape

        if self.agg_mode == "GAP":
            pooled = x.mean(axis=self.agg_axis)
            if self.mode != "Channel":
                pooled = pooled.reshape(*pooled.shape, 1)
            pre_act = self.inner.forward(pooled, train=train)

        elif self.agg_mode == "GMP":
            pooled = x.max(axis=self.agg_axis)
            if self.mode != "Channel":
                pooled = pooled.reshape(*pooled.shape, 1)
            pre_act = self.inner.forward(pooled, train=train)

        else:  # "GAP+GMP"
            pooled_1 = x.mean(axis=self.agg_axis)
            pooled_2 = x.max(axis=self.agg_axis)

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

    def backward(self, loss_grad, optimizer):
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
            dx_2 = _gap_backward(dpool, self.input, self.mode)

        elif self.agg_mode == "GMP":
            dx_2 = _gmp_backward(dpool, self.input, self.mode)

        else:  # GAP + GMP
            if self.mode == "Channel":
                dx_2 = _gap_backward(dpool[:B], self.input, self.mode) + _gmp_backward(dpool[B:], self.input,
                                                                                       self.mode)
            else:
                dx_2 = _gap_backward(dpool[:, :, :, :1], self.input, self.mode) + _gmp_backward(dpool[:, :, :, 1:2],
                                                                                                self.input,
                                                                                                self.mode)

        return dx + dx_2

    def export(self):
        dicti = {"input_need_shape": self.input_need_shape, "layer": ConvAttention, "agg_mode": self.agg_mode,
                 "reduction": self.reduction, "forward_weight": self.forward_weight,
                 "inner": self.inner.export(), "mode": self.mode, "input_dim": self.input_dim,
                 "trainable": self.trainable}

        return dicti


class MultiConvAttentionWO(Layer):
    def __init__(self, input_dim, nn_class, optimizer, d_need_head, lr=0.001, trainable=True, prev=None,
                 input_need_shape=None, kernel_size=(1, 1), mode="Channel", inner=None, **kwargs):
        self.mode = mode
        self.prev = prev
        self.trainable = trainable
        self.d_need_head = d_need_head
        self.input_dim = input_dim

        if mode == "Channel":
            if inner is None:
                inner = [{"layer": MultiAttentionWO, "d_need_head": d_need_head, "lr": lr, "trainable": trainable,
                          "input_need_shape": input_need_shape, "init_func": xavier_uniform}, {"layer": Sigmoid}]
            self.agg_axis = (1, 2)
        else:
            if "pooling_func" in kwargs:
                del kwargs["pooling_func"]
            if "out_channels" in kwargs:
                del kwargs["out_channels"]

            if inner is None:
                inner = [{"layer": Conv2D, "out_channels": d_need_head, "lr": lr, "trainable": trainable,
                          "input_need_shape": input_need_shape, "init_func": xavier_uniform,
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

    def backward(self, loss_grad, optimizer):
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
                 "inner": self.inner.export(), "input_dim": self.input_dim, "trainable": self.trainable}
        return dicti
