import numpy as np
from math import ceil
import pandas as pd

def create_submission(predictions):
    """Создание файла submission в требуемом формате"""
    
    # Берем первые 2000 предсказаний
    if len(predictions) > 2000:
        predictions = predictions[:2000]
    elif len(predictions) < 2000:
        # Дополняем если нужно
        predictions = np.pad(predictions, (0, 2000 - len(predictions)), mode='edge')

    # Параметры для выбора LDPC кода
    alpha = 0.33
    f_ec = 1.15
    R_range = [round(0.50 + 0.05 * x, 2) for x in range(9)]
    n = 32000
    d = 4800
    
    def calculate_ema(prev_ema, current_value, alpha=0.33):
        """Вычисляет EMA (Exponential Moving Average) для одного значения."""
        if prev_ema is None:
            return current_value
        return alpha * current_value + (1 - alpha) * prev_ema
    
    def ema_filter(values, alpha=0.33):
        """Применяет экспоненциальное сглаживание к массиву."""
        values = np.asarray(values, dtype=np.float64)
        smoothed = np.empty_like(values)
        smoothed[0] = values[0]
        for i in range(1, len(values)):
            smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]
        return smoothed

    def h(x):
        if x > 0:
            return -x * np.log2(x) - (1 - x) * np.log2(1 - x)
        elif x == 0:
            return 0.0
        else:
            raise ValueError('Invalid x for binary entropy')
    
    def select_code_rate(e_mu, f_ec, rates, frame_len, sp_count):
        r_candidate = 1 - h(e_mu) * f_ec
        R_res = 0.50
        s_n = sp_count
        p_n = 0
        for R in rates:
            p_n = int(ceil((1 - R) * frame_len - (1 - r_candidate) * (frame_len - sp_count)))
            s_n = int(sp_count - p_n)
            if p_n >= 0 and s_n >= 0:
                R_res = R
                return round(R_res, 2), s_n, p_n
        return round(R_res, 2), s_n, p_n
    
    # Создание submission
    prev_ema = None
    rows = []

    for E_mu_Z in predictions:
        ema_value = calculate_ema(prev_ema, float(E_mu_Z), alpha)
        prev_ema = ema_value
        R_n, s_n, p_n = select_code_rate(ema_value, f_ec, R_range, n, d)
        rows.append([f"{E_mu_Z:.16f}", R_n, s_n, p_n])
            
    submission_df = pd.DataFrame(rows)
    submission_df.to_csv('submission_pytorch.csv', header=False, index=False)

    print(f"Размер submission: {len(submission_df)} строк")