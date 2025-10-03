from neuralnet.Loaders import AsyncCupyDataLoader
from neuralnet.Features import *
from neuralnet.Layers import Dense, Conv2D, MultiHead, ConvAttention
from neuralnet.Optimizers import SGD


class NeuralNetwork:
    def __init__(self, layer_configs, loss_func=BCE(), optimizer=SGD()):
        """
        Инициализация нейросетевой модели.

        Args:
            layer_configs (list[dict]): список конфигураций слоёв.
                Каждый элемент — словарь с ключами:
                - "layer": класс слоя (например, Dense, Conv2D).
                - "input_dim": размерность входа (только для первого слоя).
                - остальные параметры зависят от типа слоя.
            loss_func (callable | list, optional): функция(и) потерь.
            optimizer (object, optional): оптимизатор (по умолчанию SGD).
        """
        self.layer_list = []
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.stream_load = cp.cuda.Stream(non_blocking=True)
        self.losses = []

        prev_layer = None
        first_config = layer_configs[0]
        input_dim = first_config["input_dim"]
        if isinstance(input_dim, int):
            input_dim = (input_dim,)

        dummy_input = cp.zeros(shape=(1, *input_dim), dtype=cp.float32)
        for config in layer_configs:

            layer_cls = config.get("layer", Dense)

            if isinstance(prev_layer, Dense):
                if layer_cls == Conv2D:
                    if not config.get("input_need_shape", None):
                        raise ValueError(
                            "Если после Dense следует Conv2d, то Conv2d нужно передать нужную форму тензора")

            if layer_cls in [MultiHead, ConvAttention]:
                config["nn_class"] = NeuralNetwork
                config["optimizer"] = self.optimizer

            if not prev_layer:
                layer = layer_cls(**config)

            else:
                config["prev"] = prev_layer
                if isinstance(dummy_input, list):
                    config["input_dim"] = dummy_input
                    layer = layer_cls(**config)
                else:
                    config["input_dim"] = dummy_input.shape[1:]
                    layer = layer_cls(**config)
                    prev_layer.next = layer

            dummy_input = layer.forward(dummy_input)

            self.layer_list.append(layer)
            prev_layer = layer

    def train(self, train_loader, epochs=30, early_stop=0, x_test=None, y_test=None, min_delta=1e-3,
              logs=True, early_target="loss_func"):
        """
        Обучение модели.

        Args:
            train_loader (iterable): генератор батчей (x, y).
            epochs (int, optional): число эпох обучения.
            early_stop (int, optional): количество эпох без улучшения,
                после которых обучение останавливается. Если 0 — отключено.
            x_test (np.ndarray, optional): валидационные данные X для ранней остановки.
            y_test (np.ndarray | list, optional): валидационные метки.
            min_delta (float, optional): минимальное улучшение метрики для сброса patience.
            logs (bool, optional): печатать ли прогресс по эпохам.
            early_target (str | callable | list, optional):
                метрика для отслеживания в ранней остановке.
                По умолчанию — loss_func.

        Returns:
            list | None: экспортированные веса лучшей модели (если early_stop > 0),
            иначе None.
        """

        if early_stop and (x_test is None or y_test is None):
            raise ValueError("Для ранней остановки нужны x_test и y_test")

        if early_target == "loss_func":
            early_target = self.loss_func

        if x_test is not None:

            x_test = AsyncCupyDataLoader(np.asarray(x_test, dtype=cp.float32), batch_size=train_loader.batch_size,
                                         shuffle=False, stream=train_loader.stream)
            if isinstance(early_target, list):
                y_test = [cp.asarray(arr, dtype=cp.float32) for arr in y_test]
            else:
                y_test = cp.asarray(y_test, dtype=cp.float32)

        best_loss = float('inf')
        min_delta = cp.asarray(min_delta, dtype=cp.float32)
        patience_counter = 0
        best_model = None
        best_epoch = 0
        val_loss = best_loss

        for epoch in range(1, epochs + 1):
            for bx, by in train_loader:
                y_hat = self.forward(bx, train=True)

                if isinstance(self.loss_func, list):
                    loss_grad = []
                    for i, loss_func in enumerate(self.loss_func):
                        loss = loss_func.grad(y_hat[i], by[i])
                        loss_grad.append(loss)
                else:
                    loss_grad = self.loss_func.grad(y_hat, by)
                self.backward(loss_grad)
            # Early stopping — после всей эпохи
            if early_stop:
                y_val_pred = self.predict(x_test, numpy=False)

                # поддержка single, list
                if isinstance(early_target, list):
                    val_losses = []
                    for i, loss_func in enumerate(early_target):
                        val_losses.append(loss_func(y_val_pred[i], y_test[i]))
                    val_loss = sum(val_losses)
                else:
                    val_loss = early_target(y_val_pred, y_test)

                del y_val_pred
                self.losses.append(val_loss)

                if cp.isnan(val_loss):
                    print(f"Model unstable at epoch {epoch}. Best val loss: {best_loss:.6f} at epoch {best_epoch}")
                    return best_model

                if val_loss < best_loss - min_delta:
                    best_epoch = epoch
                    best_loss = val_loss
                    patience_counter = 0
                    best_model = self.export()
                else:
                    patience_counter += 1

                if patience_counter >= early_stop:
                    print(f"Early stopping at epoch {epoch}. Best val loss: {best_loss:.6f} at epoch {best_epoch}")
                    return best_model

            if logs:
                print(f"Epoch: {epoch}, Metric: {val_loss:.6f}, Best_val_metric: {best_loss:.6f}")

        return best_model

    def predict(self, predictor_loader, numpy=True):
        """
        Предсказание на новых данных.

        Args:
            predictor_loader (AsyncCupyDataLoader): входные данные.
            numpy (bool, optional): вернуть результат как numpy-массив (True)
                или cupy-массив (False).

        Returns:
            np.ndarray | list[np.ndarray] | cp.ndarray:
                предсказания модели. Если используется многоголовая модель,
                возвращается список массивов.
        """
        if isinstance(self.loss_func, list):
            predictions = [[] for _ in range(len(self.loss_func))]
        else:
            predictions = []

        for xb, _ in predictor_loader:
            y_hat = self.forward(xb, train=False)
            if isinstance(self.loss_func, list):
                for i, z in enumerate(y_hat):
                    predictions[i].append(z)
            else:
                predictions.append(y_hat)

        if isinstance(self.loss_func, list):
            predictions = [
                cp.concatenate(p, axis=0).get() if numpy else cp.concatenate(p, axis=0)
                for p in predictions
            ]
        else:
            predictions = cp.concatenate(predictions, axis=0)
            predictions = predictions.get() if numpy else predictions
        return predictions

    def forward(self, x, train=False):
        """
        Прямой проход по слоям модели.

        Args:
            x (cp.ndarray): входной тензор.
            train (bool, optional): режим обучения (True) или инференса (False).

        Returns:
            cp.ndarray | list[cp.ndarray]: выход модели.
        """
        for layer in self.layer_list:
            x = layer.forward(x, train)
        return x

    def backward(self, loss_grad):
        """
        Обратное распространение ошибки.

        Args:
            loss_grad (cp.ndarray | list[cp.ndarray]): градиент функции потерь.

        Returns:
            cp.ndarray | list[cp.ndarray]: градиенты для входного слоя.
        """

        for layer in reversed(self.layer_list):
            loss_grad = layer.backward(loss_grad, self.optimizer)
        return loss_grad

    def export(self):
        """
        Экспорт параметров всех слоёв модели.

        Returns:
            list: список параметров (например, весов и смещений) по каждому слою.
        """
        return [layer.export() for layer in self.layer_list]
