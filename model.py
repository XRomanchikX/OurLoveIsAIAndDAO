import torch
import torch.nn as nn

class moving_avg(nn.Module):
    """Блок скользящего среднего для выделения тренда временного ряда.

    Аргументы:
        kernel_size: размер окна свёртки (ядра).
        stride: шаг скользящего среднего.

    """

    def __init__(self, kernel_size: int, stride: int):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Прямой проход для вычисления скользящего среднего.

        Аргументы:
            x: входной тензор.

        Возвращает:
            тензор после применения скользящего среднего.

        """
        # добавляем паддинг (повторяем крайние значения) с обеих сторон временного ряда
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)

        # применяем скользящее среднее по временной оси
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)

        return x


class series_decomp(nn.Module):
    """Блок декомпозиции временного ряда.

    Аргументы:
        kernel_size: размер окна для скользящего среднего.

    """

    def __init__(self, kernel_size: int):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        """Прямой проход для декомпозиции ряда на тренд и остаток.

        Аргументы:
            x: входной тензор.

        Возвращает:
            кортеж тензоров (остаток, тренд).

        """
        moving_mean = self.moving_avg(x)
        res = x - moving_mean

        return res, moving_mean


class DLinear_NN(nn.Module):
    def __init__(self, pred_len, seq_len, moving_avg=25, **kwargs):
        super(DLinear_NN, self).__init__()

        # Защита от типовых «обёрток»
        def _to_int(x):
            if isinstance(x, int):
                return x
            if isinstance(x, dict) and "value" in x:
                return int(x["value"])
            try:
                return int(x)
            except Exception:
                raise TypeError(f"Expected int-like, got {type(x)}: {x}")

        # Если вдруг пришли ещё и именованные — заберём их, чтобы не мешали
        seq_len = _to_int(kwargs.pop("seq_len", seq_len))
        pred_len = _to_int(kwargs.pop("pred_len", pred_len))
        moving_avg = int(kwargs.pop("moving_avg", moving_avg))

        self.seq_len = seq_len
        self.pred_len = pred_len

        # pred_len = 4 | seq_len = 160 | features_groups = {'series': 160, 'id': 0, 'fh': 0, 'datetime_features': 0, 'series_features': 0, 'cycle_features': 0, 'other_features': 0}
        self.decompsition = series_decomp(moving_avg)
        
        self.Linear_Seasonal = nn.Sequential(
            nn.Linear(seq_len, pred_len),  # первый слой
            nn.LayerNorm(pred_len),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(pred_len, 512),  # первый слой
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(512, pred_len),  # первый слой
            nn.LayerNorm(pred_len),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(pred_len, pred_len)  # если хочешь еще слои
        )

        # Задаём веса первого линейного слоя
        with torch.no_grad():
            self.Linear_Seasonal[0].weight = nn.Parameter(
                (1 / seq_len) * torch.ones([pred_len, seq_len])
            )

        # Аналогично для Linear_Trend
        self.Linear_Trend = nn.Sequential(
            nn.Linear(seq_len, pred_len),  # первый слой
            nn.LayerNorm(pred_len),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(pred_len, 512),  # первый слой
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(512, pred_len),  # первый слой
            nn.LayerNorm(pred_len),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(pred_len, pred_len)  # если хочешь еще слои
        )

        with torch.no_grad():
            self.Linear_Trend[0].weight = nn.Parameter(
                (1 / seq_len) * torch.ones([pred_len, seq_len])
            )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Прямой проход модели.

        Аргументы:
            x: входной тензор формы (batch_size, seq_len, num_features).

        Возвращает:
            выходной тензор формы (batch_size, pred_len, num_features).

        """
        # Декомпозиция временного ряда на тренд и остаток (сезонность)
        seasonal_init, trend_init = self.decompsition(x)

        # Транспонируем тензоры в формат (batch_size, num_features, seq_len)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(
            0, 2, 1
        )

        # Применяем линейные слои к тренду и остаткам
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        # Складываем результаты линейных слоёв
        x = seasonal_output + trend_output

        # Транспонируем обратно в формат (batch_size, seq_len, num_features)
        x = x.permute(0, 2, 1)

        return x[:, -self.pred_len :, :]