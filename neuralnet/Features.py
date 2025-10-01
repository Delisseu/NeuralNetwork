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


def l2_regularization_grad(W, lmda, *args, **kwargs):
    return lmda * TWO * W


def l1_regularization_grad(W, lmda, *args, **kwargs):
    return lmda * cp.sign(W)


class NoAct:
    @staticmethod
    def __call__(pre_act, *args, **kwargs):
        return pre_act

    @staticmethod
    def backward(loss_grad, *args, **kwargs):
        return loss_grad


class Sigmoid:
    def __init__(self):
        self.after_act = None

    def __call__(self, pre_act, train, *args, **kwargs):
        after_act = ONE / (ONE + cp.exp(-pre_act))
        if train:
            self.after_act = after_act
        return after_act

    def backward(self, loss_grad, *args, **kwargs):
        return loss_grad * self.after_act * (ONE - self.after_act)


class Relu:
    def __init__(self):
        self.pre_act = None

    def __call__(self, pre_act, train, *args, **kwargs):
        if train:
            self.pre_act = pre_act
        return cp.maximum(ZERO, pre_act)

    def backward(self, loss_grad, *args, **kwargs):
        return loss_grad * cp.where(self.pre_act > ZERO, ONE, ZERO)


class LeakyRelu:
    def __init__(self):
        self.pre_act = None

    def __call__(self, pre_act, alpha, train, *args, **kwargs):
        if train:
            self.pre_act = pre_act
        return cp.where(pre_act > ZERO, pre_act, alpha * pre_act)

    def backward(self, loss_grad, alpha, *args, **kwargs):
        return loss_grad * cp.where(self.pre_act > ZERO, ONE, alpha)


class ELU:
    def __init__(self):
        self.pre_act = None

    def __call__(self, pre_act, alpha, train, *args, **kwargs):
        if train:
            self.pre_act = pre_act
        return cp.where(pre_act > ZERO, pre_act, alpha * (cp.exp(pre_act) - ONE))

    def backward(self, loss_grad, alpha, *args, **kwargs):
        return loss_grad * cp.where(self.pre_act > ZERO, ONE, alpha * cp.exp(self.pre_act))


class Softmax:
    def __init__(self):
        self.after_act = None

    def __call__(self, pre_act, train, *args, **kwargs):
        exp_shifted = cp.exp(pre_act - cp.max(pre_act, axis=-1, keepdims=True))
        after_act = exp_shifted / cp.sum(exp_shifted, axis=-1, keepdims=True)
        if train:
            self.after_act = after_act
        return after_act

    def backward(self, loss_grad, *args, **kwargs):
        dot = cp.sum(loss_grad * self.after_act, axis=-1, keepdims=True)
        grad_out = self.after_act * (loss_grad - dot)
        return grad_out


class Tanh:
    def __init__(self):
        self.after_act = None

    def __call__(self, pre_act, train, *args, **kwargs):
        after_act = cp.tanh(pre_act)
        if train:
            self.after_act = after_act
        return after_act

    def backward(self, loss_grad, *args, **kwargs):
        return loss_grad * (ONE - self.after_act ** TWO)
