import cupy as cp
import numpy as np

from neuralnet.CONSTANTS import HALF, ONE, TWO, THREE


class Padding:
    def __init__(self, kernel_size, stride=1, padding="full", pad_shape=None, **kwargs):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pad_shape = pad_shape

    def forward(self, x, train=False):
        if self.pad_shape is None:
            self.pad_shape = self.compute_padding(x.shape)
        return self.apply_padding(x)

    def backward(self, grad, optimizer=None):
        h, w = self.pad_shape
        top, bottom = h
        left, right = w
        h_end = None if bottom == 0 else -bottom
        w_end = None if right == 0 else -right
        return grad[:, top: h_end, left: w_end]

    def apply_padding(self, x):
        return cp.pad(x, pad_width=((0, 0), *self.pad_shape, (0, 0)), mode='constant',
                      constant_values=0) if self.pad_shape != ((0, 0), (0, 0)) else x

    def compute_padding(self, input_shape):
        if self.padding == 'valid':
            return (0, 0), (0, 0)

        pad_total_h = self.kernel_size[0]
        pad_total_w = self.kernel_size[1]
        pad_top = pad_total_h // 2
        pad_left = pad_total_w // 2
        size_step_diff_bottom = input_shape[1] - (1 + ((input_shape[1] - 1) // self.stride) * self.stride)
        size_step_diff_right = input_shape[2] - (1 + ((input_shape[2] - 1) // self.stride) * self.stride)

        if pad_total_w % 2 == 0:
            size_step_diff_bottom += 1

        if pad_total_h % 2 == 0:
            size_step_diff_right += 1

        if self.padding == 'same_strict':
            pad_bottom = max(0, pad_top - size_step_diff_bottom)
            pad_right = max(0, pad_left - size_step_diff_right)

        elif self.padding == 'full':
            pad_bottom = (self.stride - 1 - size_step_diff_bottom) + pad_top
            pad_right = (self.stride - 1 - size_step_diff_right) + pad_left
        else:
            return (0, 0), (0, 0)

        return (pad_top, pad_bottom), (pad_left, pad_right)

    def export(self):
        return {"padding": self.padding, "pad_shape": self.pad_shape, "stride": self.stride, "layer": Padding,
                "kernel_size": self.kernel_size}


class PatchExtractor:
    def __init__(self, kernel_size, stride=1, flatten=True, input_dim=None, **kwargs):
        self.flatten = flatten
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.x_shape = None
        self.patch_shape = None

    def forward(self, x, train=False):
        self.x_shape = x.shape
        patches = extract_patches(x, *self.kernel_size, self.stride)

        self.patch_shape = patches.shape[1:]
        patches = patches.reshape(x.shape[0], *self.patch_shape[:2], -1)
        if self.flatten:
            patches = patches.reshape(-1, patches.shape[-1])

        return patches

    def backward(self, patches_grad, optimizer=None):
        return unpatchify(patches_grad.reshape(self.x_shape[0], *self.patch_shape), self.x_shape, *self.kernel_size,
                          self.stride)

    def export(self):
        return {"layer": PatchExtractor, "stride": self.stride, "flatten": self.flatten, "input_dim": self.input_dim,
                "kernel_size": self.kernel_size}


class Pooling:
    def __init__(self, pooling_shape=(2, 2), pooling_func="max", channelwise=False, pooling_stride=None, **kwargs):
        func_dict = {"max": cp.max, "mean": cp.mean, "min": cp.min, "GAP": cp.mean}
        self.channelwise = channelwise

        self.axis = (3, 4, 5) if self.channelwise else (4, 5)
        self.dict_key = pooling_func

        if pooling_func == "GAP":
            self.axis = (1, 2, 4, 5)

        self.pooling_func = func_dict.get(pooling_func, None)
        self.pooling_shape = pooling_shape
        self.pooling_size = cp.asarray(np.prod(pooling_shape), dtype=cp.float32)
        self.stride = pooling_stride
        if self.stride is None:
            self.stride = pooling_shape[0]

        self.patches_shape = None

        self.mask = None
        self.x_shape = None

    def export(self):
        return {"pooling_func": self.dict_key, "channelwise": self.channelwise, "pooling_shape": self.pooling_shape,
                "pooling_stride": self.stride, "layer": Pooling}

    def forward(self, x, train=False):
        patches = extract_patches(x, *self.pooling_shape, self.stride)
        pool_result = self.pooling_func(patches, axis=self.axis, keepdims=True)  # (B, H, W, C, kH, kW)

        self.patches_shape = patches.shape
        # Маска: где значение в патче совпадает
        if train:
            self.x_shape = x.shape
            if self.dict_key not in ("GAP", "mean"):
                # Маска для максимумов
                self.mask = (patches == pool_result).astype(cp.float32)
                # Считаем, сколько максимумов в каждом патче
                max_count = self.mask.sum(axis=(-2, -1), keepdims=True)
                # Делим на число максимумов, чтобы распределить градиент равномерно
                self.mask /= max_count

        return pool_result.reshape(pool_result.shape[:4])

    def backward(self, grad_output, optimizer=None):
        # Расширим до совпадения с маской

        grad_expanded = grad_output[..., cp.newaxis, cp.newaxis]  # (B, out_H, out_W, C, 1, 1)

        if self.dict_key in ("GAP", "mean"):
            grad_input_patches = cp.zeros_like(self.patches_shape, dtype=cp.float32)
            grad_input_patches += grad_expanded / self.pooling_size

        else:
            # Градиент приходит только в позиции максимумов или минимумов
            grad_input_patches = grad_expanded * self.mask

        grad = unpatchify(grad_input_patches, self.x_shape, *self.pooling_shape, self.stride)

        return grad


class BatchNorm:
    def __init__(self, input_dim, momentum=0.9, eps=1e-8, gamma=None, beta=None,
                 run_m=None, run_v=None, trainable=True, lr=0.001, prev=None, **kwargs):

        self.prev = prev
        self.next = None
        out_dim = input_dim[-1]
        self.learning_rate = cp.array(lr, dtype=cp.float32)
        self.trainable = trainable
        self.momentum = cp.array(momentum, dtype=cp.float32)
        self.eps = cp.array(eps, dtype=cp.float32)

        shape = (1,) * len(input_dim) + (out_dim,)

        self.axis = tuple(range(len(input_dim)))

        self.gamma = cp.ones(shape, dtype=cp.float32) if gamma is None else cp.asarray(gamma, dtype=cp.float32)
        self.beta = cp.zeros(shape, dtype=cp.float32) if beta is None else cp.asarray(beta, dtype=cp.float32)

        self.running_mean = cp.zeros(shape, dtype=cp.float32) if run_m is None else cp.asarray(run_m, dtype=cp.float32)
        self.running_var = cp.ones(shape, dtype=cp.float32) if run_v is None else cp.asarray(run_v, dtype=cp.float32)

        # Для backward
        self.std_inv = None
        self.x_mu = None
        self.z_norm = None

        self.batch_cache = {}
        self.m = cp.asarray(np.prod([(1, *input_dim)[ax] for ax in self.axis]), dtype=cp.float32)

    def _get_batch_const(self, batch_size):
        if batch_size not in self.batch_cache:
            self.batch_cache[batch_size] = cp.asarray(batch_size, dtype=cp.float32)
        return self.batch_cache[batch_size]

    def forward(self, x, train=False):
        if train:
            batch_mean = x.mean(axis=self.axis, keepdims=True)
            batch_var = x.var(axis=self.axis, keepdims=True)

            self.x_mu = x - batch_mean
            self.std_inv = ONE / cp.sqrt(batch_var + self.eps)
            self.z_norm = self.x_mu * self.std_inv

            # Обновляем EMA
            self.running_mean *= self.momentum
            self.running_mean += (ONE - self.momentum) * batch_mean
            self.running_var *= self.momentum
            self.running_var += (ONE - self.momentum) * batch_var

            out = self.gamma * self.z_norm + self.beta
        else:
            # Используем только EMA
            x_hat = (x - self.running_mean) / cp.sqrt(self.running_var + self.eps)
            out = self.gamma * x_hat + self.beta
        return out

    def backward(self, grad, optimizer):
        m = self.m * self._get_batch_const(grad.shape[0])

        # dL/dgamma и dL/dbeta
        dgamma = cp.sum(grad * self.z_norm, axis=self.axis, keepdims=True)
        dbeta = cp.sum(grad, axis=self.axis, keepdims=True)

        # dL/dx (основной градиент)
        dx_norm = grad * self.gamma
        dvar = cp.sum(dx_norm * self.x_mu * -HALF * self.std_inv ** THREE, axis=self.axis, keepdims=True)
        dmean = cp.sum(dx_norm * -self.std_inv, axis=self.axis, keepdims=True) + dvar * cp.mean(-TWO * self.x_mu,
                                                                                                axis=self.axis,
                                                                                                keepdims=True)

        out_grad = dx_norm * self.std_inv + dvar * TWO * self.x_mu / m + dmean / m
        optimizer.step(self.gamma, dgamma, self.trainable, self.learning_rate)
        optimizer.step(self.beta, dbeta, self.trainable, self.learning_rate)
        return out_grad

    def export(self):
        return {"momentum": self.momentum.copy(), "eps": self.eps.copy(), "trainable": self.trainable,
                "gamma": self.gamma.copy(), "lr": self.learning_rate.copy(),
                "beta": self.beta.copy(), "run_m": self.running_mean.copy(), "run_v": self.running_var.copy(),
                "layer": BatchNorm}


class LayerNorm:
    def __init__(self, input_dim=None, eps=1e-8, trainable=True, lr=0.001, gamma=None,
                 beta=None, prev=None, *args, **kwargs):
        """
        LayerNorm универсальный для Dense и Conv2D.
        input_dim: форма входа
        """

        self.prev = prev
        self.next = None
        self.eps = cp.asarray(eps, dtype=cp.float32)
        self.trainable = trainable
        self.learning_rate = cp.array(lr, dtype=cp.float32)

        shape = (1, *input_dim)
        self.axis = tuple(range(1, len(input_dim)))
        # Параметры масштабирования и сдвига
        # shape = (1, 1, ..., C) или (1, D) для Dense
        self.gamma = cp.ones(shape, dtype=cp.float32) if gamma is None else cp.asarray(gamma, dtype=cp.float32)
        self.beta = cp.zeros(shape, dtype=cp.float32) if beta is None else cp.asarray(beta, dtype=cp.float32)

        # Для backward
        self.m = cp.prod(cp.array(input_dim, dtype=cp.float32))
        self.x_mu = None
        self.std_inv = None
        self.z_norm = None

    def forward(self, x, train=False):
        # Нормализуем по всем осям кроме первой (батча)
        mean = cp.mean(x, axis=self.axis, keepdims=True)
        var = cp.var(x, axis=self.axis, keepdims=True)

        self.x_mu = x - mean
        self.std_inv = ONE / cp.sqrt(var + self.eps)
        self.z_norm = self.x_mu * self.std_inv

        out = self.gamma * self.z_norm + self.beta
        return out

    def backward(self, grad, optimizer):
        m = self.m

        dgamma = cp.sum(grad * self.z_norm, axis=0, keepdims=True)
        dbeta = cp.sum(grad, axis=0, keepdims=True)

        dx_norm = grad * self.gamma
        dvar = cp.sum(dx_norm * self.x_mu * -HALF * self.std_inv ** THREE, axis=self.axis, keepdims=True)
        dmean = cp.sum(dx_norm * -self.std_inv, axis=self.axis, keepdims=True) + dvar * cp.mean(-TWO * self.x_mu,
                                                                                                axis=self.axis,
                                                                                                keepdims=True)
        dx = dx_norm * self.std_inv + dvar * TWO * self.x_mu / m + dmean / m

        if self.trainable:
            optimizer.step(self.gamma, dgamma, self.trainable, self.learning_rate)
            optimizer.step(self.beta, dbeta, self.trainable, self.learning_rate)

        return dx

    def export(self):
        return {"eps": self.eps.copy(), "gamma": self.gamma.copy(), "beta": self.beta.copy(),
                "layer": LayerNorm, "lr": self.learning_rate.copy(), "trainable": self.trainable}


class Dropout:
    def __init__(self, drop_rate=0.2, **kwargs):
        self.rate = cp.array(drop_rate, dtype=cp.float32)
        self.mask = None

    def forward(self, after_act, train=False):
        if train:
            # Подправляем размер, чтобы подходил для Dense и для Conv2d
            mask_shape = [after_act.shape[0]] + [1] * (after_act.ndim - 2) + [after_act.shape[-1]]
            self.mask = cp.random.rand(*mask_shape, dtype=cp.float32) > self.rate
            return after_act * self.mask / (ONE - self.rate)
        else:
            return after_act * (ONE - self.rate)

    def backward(self, grad, optimizer=None):
        return grad * self.mask

    def export(self):
        return {"drop_rate": self.rate.copy()}


def extract_patches(x, kH, kW, stride):
    batch_size, H, W, C = x.shape
    out_h = (H - kH) // stride + 1
    out_w = (W - kW) // stride + 1

    shape = (batch_size, out_h, out_w, C, kH, kW)
    strides = (
        x.strides[0],
        x.strides[1] * stride,
        x.strides[2] * stride,
        x.strides[3],
        x.strides[1],
        x.strides[2]
    )
    return cp.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def unpatchify(patches, input_shape, kH, kW, stride):
    """
    Собирает патчи обратно в изображение.

    patches: (B, out_H, out_W, C, kH, kW)
    input_shape: (B, H, W, C)
    kH, kW: размер патча
    stride: шаг
    """
    B, H, W, C = input_shape

    if stride == kH == kW:
        tmp = patches.transpose(0, 1, 4, 2, 5, 3)
        # Объединяем соответствующие оси
        B, out_H, kH, out_W, kW, C = tmp.shape
        H, W = out_H * kH, out_W * kW
        grad_input = tmp.reshape(B, H, W, C)
    else:
        out_H = (H - kH) // stride + 1
        out_W = (W - kW) // stride + 1

        # Индексы для вставки патчей
        h_idx = cp.arange(kH)[None, :] + stride * cp.arange(out_H)[:, None]  # (out_H, kH)
        w_idx = cp.arange(kW)[None, :] + stride * cp.arange(out_W)[:, None]  # (out_W, kW)

        # Преобразуем патчи в (B, out_H, out_W, kH, kW, C)
        patches = patches.transpose(0, 1, 2, 4, 5, 3)

        # Создаём плоские индексы
        b_idx = cp.arange(B)[:, None, None, None, None, None]
        c_idx = cp.arange(C)[None, None, None, None, None, :]
        h_idx_broadcast = h_idx[None, :, None, :, None, None]
        w_idx_broadcast = w_idx[None, None, :, None, :, None]

        # Вычисляем linear indices для cp.add.at
        linear_idx = (
                b_idx * (H * W * C) +
                h_idx_broadcast * (W * C) +
                w_idx_broadcast * C +
                c_idx
        ).ravel()

        # Flatten значения патчей
        values = patches.ravel()

        # Собираем в изображение
        grad_input_flat = cp.zeros(B * H * W * C, dtype=patches.dtype)
        cp.add.at(grad_input_flat, linear_idx, values)

        grad_input = grad_input_flat.reshape(B, H, W, C)

    # Если получилось, что ядро пуллинга не покрыло все изображение
    pad_bottom = input_shape[1] - grad_input.shape[1]
    pad_right = input_shape[2] - grad_input.shape[2]

    if pad_bottom > 0 or pad_right > 0:
        grad_input = cp.pad(
            grad_input,
            ((0, 0), (0, pad_bottom), (0, pad_right), (0, 0)),
            mode='constant'
        )
    return grad_input


def _gap_backward(dpool, x, mode="Channel"):
    B, H, W, C = x.shape
    if mode == "Channel":
        reshape = (B, 1, 1, C)
        norm = (H * W)
    else:
        reshape = (B, H, W, 1)
        norm = C
    # равномерно распределяем градиент
    dx = cp.zeros_like(x, dtype=cp.float32)
    dx += dpool.reshape(*reshape) / norm
    return dx


def _gmp_backward(dpool, x, mode="Channel"):
    B, H, W, C = x.shape
    if mode == "Channel":
        # ищем индексы максимумов
        flat_idx = x.reshape(B, H * W, C).argmax(axis=1)  # (B, C), индексы вдоль H*W
        dx = cp.zeros_like(x.reshape(B, H * W, C), dtype=cp.float32)

        # раскладываем dpool в эти индексы
        b_idx = cp.arange(B)[:, None]
        c_idx = cp.arange(C)[None, :]
        dx[b_idx, flat_idx, c_idx] = dpool  # "разбрасываем" сразу по батчу и каналам
        return dx.reshape(B, H, W, C)
    else:
        flat_idx = x.argmax(axis=-1)  # (B,H,W), индекс канала
        dx = cp.zeros_like(x, dtype=cp.float32)

        b_idx, h_idx, w_idx = cp.indices((B, H, W))
        dx[b_idx, h_idx, w_idx, flat_idx] = dpool.reshape(B, H, W)
        return dx


def xavier_uniform(fan_in, fan_out, *args, **kwargs):
    limit = cp.sqrt(6 / (fan_in + fan_out))
    return cp.random.uniform(-limit, limit, size=(fan_in, fan_out), dtype=cp.float32)


def kaiming_uniform(fan_in, fan_out, alpha=0.0, *args, **kwargs):
    bound = cp.sqrt(6 / ((1 + alpha ** 2) * fan_in))
    return cp.random.uniform(-bound, bound, size=(fan_in, fan_out), dtype=cp.float32)
