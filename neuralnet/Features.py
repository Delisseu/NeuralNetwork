import cupy as cp
import numpy as np

from neuralnet.CONSTANTS import ZERO, ONE, TWO


class MSE:
    @staticmethod
    def __call__(y_hat, y):
        return cp.mean((y_hat - y) ** TWO)

    @staticmethod
    def grad(y_hat, y, *args, **kwargs):
        return TWO * (y_hat - y) / y.shape[0]


class MAE:
    @staticmethod
    def __call__(y_hat, y):
        return cp.mean(np.abs(y_hat - y))

    @staticmethod
    def grad(y_hat, y, *args, **kwargs):
        return cp.sign(y_hat - y) / y.shape[0]


class BCE:
    def __init__(self, pos_weight=1, neg_weight=1, eps=1e-8, *args, **kwargs):
        self.eps = cp.asarray(eps, dtype=cp.float32)
        self.pos_weight = cp.asarray(pos_weight, dtype=cp.float32)
        self.neg_weight = cp.asarray(neg_weight, dtype=cp.float32)

    def __call__(self, y_hat, y):
        y_hat = cp.clip(y_hat, self.eps, ONE - self.eps)
        return -cp.mean(
            self.pos_weight * y * cp.log(y_hat + self.eps) +
            self.neg_weight * (ONE - y) * cp.log(ONE - y_hat + self.eps)
        )

    def grad(self, y_hat, y, eps=cp.array(1e-8, dtype=cp.float32), *args, **kwargs):
        y_hat = cp.clip(y_hat, eps, 1 - eps)
        weights = self.pos_weight * y + self.neg_weight * (1 - y)
        return weights * (y_hat - y) / (y_hat * (1 - y_hat) * y.size)


class BCELogits:
    def __init__(self, pos_weight=1, neg_weight=1, eps=1e-8):
        self.eps = cp.asarray(eps, dtype=cp.float32)
        self.pos_weight = cp.asarray(pos_weight, dtype=cp.float32)
        self.neg_weight = cp.asarray(neg_weight, dtype=cp.float32)

    def __call__(self, logits, y):
        """
        Вычисляет BCE с логитами.
        logits: выход сети без сигмоиды
        y: целевые метки 0 или 1
        """
        # Используем численно стабильную формулу:
        max_logits = cp.maximum(logits, ZERO)
        log_exp = cp.log(ONE + cp.exp(-cp.abs(logits)))
        loss = max_logits - logits * y + log_exp

        # Применяем веса для положительных и отрицательных примеров
        weights = self.pos_weight * y + self.neg_weight * (ONE - y)
        return cp.mean(weights * loss)

    def grad(self, logits, y):
        """
        Градиент по логитам.
        grad = pos_weight*(sigmoid(logits)-y) для положительных
        grad = neg_weight*(sigmoid(logits)-y) для отрицательных
        """
        sigmoid = ONE / (ONE + cp.exp(-logits))
        weights = self.pos_weight * y + self.neg_weight * (ONE - y)
        return weights * (sigmoid - y) / y.size


class CCE:
    @staticmethod
    def __call__(y_hat, y, eps=cp.asarray(1e-8, dtype=cp.float32)):
        y_hat = cp.clip(y_hat, eps, ONE - eps)
        return -cp.mean(cp.sum(y * cp.log(y_hat), axis=1))

    @staticmethod
    def grad(y_hat, y, eps=cp.asarray(1e-8, dtype=cp.float32)):
        y_hat = cp.clip(y_hat, eps, ONE - eps)
        return -y / y_hat / y.size


class CCELogits:
    @staticmethod
    def __call__(y_hat, y):
        """
        Categorical Cross-Entropy (CCE) для логитов.
        y_hat: логиты
        y: one-hot метки
        """
        # численно стабильный log_softmax
        logsumexp = cp.log(cp.sum(cp.exp(y_hat - cp.max(y_hat, axis=1, keepdims=True)), axis=1, keepdims=True))
        log_probs = y_hat - cp.max(y_hat, axis=1, keepdims=True) - logsumexp
        return -cp.mean(cp.sum(y * log_probs, axis=1))

    @staticmethod
    def grad(y_hat, y):
        """
        Градиент CCE по логитам.
        y_hat: логиты
        y: one-hot метки
        """
        # превращаем логиты в вероятности
        exp_logits = cp.exp(y_hat - cp.max(y_hat, axis=1, keepdims=True))
        probs = exp_logits / cp.sum(exp_logits, axis=1, keepdims=True)

        return (probs - y) / y.size


def elastic_grad(W, lamda_1, lamda_2, *args, **kwargs):
    return l2_regularization_grad(W, lamda_1) + l1_regularization_grad(W, lamda_2)


def l2_regularization_grad(W, lamda, *args, **kwargs):
    return lamda * TWO * W


def l1_regularization_grad(W, lamda, *args, **kwargs):
    return lamda * cp.sign(W)


class Sigmoid:
    def __init__(self, input_dim=None, *args, **kwargs):
        self.input_dim = input_dim
        self.after_act = None

    def forward(self, pre_act, train=False):
        after_act = ONE / (ONE + cp.exp(-pre_act))
        if train:
            self.after_act = after_act
        return after_act

    def backward(self, loss_grad, optimizer=None):
        return loss_grad * self.after_act * (ONE - self.after_act)

    def export(self):
        return {"layer": Sigmoid, "input_dim": self.input_dim}


class Relu:
    def __init__(self, input_dim=None, *args, **kwargs):
        self.input_dim = input_dim
        self.pre_act = None

    def forward(self, pre_act, train=False):
        if train:
            self.pre_act = pre_act
        return cp.maximum(ZERO, pre_act)

    def backward(self, loss_grad, optimizer=None):
        return loss_grad * cp.where(self.pre_act > ZERO, ONE, ZERO)

    def export(self):
        return {"layer": Relu, "input_dim": self.input_dim}


class LeakyRelu:
    def __init__(self, alpha=0.01, input_dim=None, *args, **kwargs):
        self.input_dim = input_dim
        self.pre_act = None
        self.alpha = alpha

    def forward(self, pre_act, train=False):
        if train:
            self.pre_act = pre_act
        return cp.where(pre_act > ZERO, pre_act, self.alpha * pre_act)

    def backward(self, loss_grad, optimizer=None):
        return loss_grad * cp.where(self.pre_act > ZERO, ONE, self.alpha)

    def export(self):
        return {"layer": LeakyRelu, "alpha": self.alpha, "input_dim": self.input_dim}


class ELU:
    def __init__(self, alpha=0.01, input_dim=None, *args, **kwargs):
        self.input_dim = input_dim
        self.pre_act = None
        self.alpha = alpha

    def forward(self, pre_act, train=False):
        if train:
            self.pre_act = pre_act
        return cp.where(pre_act > ZERO, pre_act, self.alpha * (cp.exp(pre_act) - ONE))

    def backward(self, loss_grad, optimizer=None):
        return loss_grad * cp.where(self.pre_act > ZERO, ONE, self.alpha * cp.exp(self.pre_act))

    def export(self):
        return {"layer": ELU, "alpha": self.alpha, "input_dim": self.input_dim}


class Softmax:
    def __init__(self, input_dim=None, *args, **kwargs):
        self.input_dim = input_dim
        self.after_act = None

    def forward(self, pre_act, train=False):
        exp_shifted = cp.exp(pre_act - cp.max(pre_act, axis=-1, keepdims=True))
        after_act = exp_shifted / cp.sum(exp_shifted, axis=-1, keepdims=True)
        if train:
            self.after_act = after_act
        return after_act

    def backward(self, loss_grad, optimizer=None):
        dot = cp.sum(loss_grad * self.after_act, axis=-1, keepdims=True)
        grad_out = self.after_act * (loss_grad - dot)
        return grad_out

    def export(self):
        return {"layer": Softmax, "input_dim": self.input_dim}


class Tanh:
    def __init__(self, input_dim=None, *args, **kwargs):
        self.input_dim = input_dim
        self.after_act = None

    def forward(self, pre_act, train=False):
        after_act = cp.tanh(pre_act)
        if train:
            self.after_act = after_act
        return after_act

    def backward(self, loss_grad, optimizer=None):
        return loss_grad * (ONE - self.after_act ** TWO)

    def export(self):
        return {"layer": Tanh, "input_dim": self.input_dim}
