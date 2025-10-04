import torch
import torch.nn as nn

def calculate_ber(predictions, targets, threshold=0.5):
    """
    Вычисление Bit Error Rate (BER)
    """
    # Преобразуем в бинарные значения
    pred_binary = (predictions > threshold).float()
    target_binary = (targets > threshold).float()
    
    # Считаем количество ошибок
    errors = (pred_binary != target_binary).sum().item()
    total_bits = targets.numel()
    
    ber = errors / total_bits if total_bits > 0 else 0.0
    return ber

def calculate_snr(predictions, targets):
    """
    Вычисление Signal-to-Noise Ratio (SNR) в дБ
    """
    signal_power = torch.mean(targets ** 2)
    noise_power = torch.mean((targets - predictions) ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr_linear = signal_power / noise_power
    snr_db = 10 * torch.log10(snr_linear)
    
    return snr_db.item() if snr_db.numel() == 1 else snr_db

def calculate_spectral_efficiency(predictions, targets, bandwidth=1.0):
    """
    Вычисление Spectral Efficiency (SE)
    Простая оценка: SE = R/B, где R - скорость передачи данных, B - полоса пропускания
    """
    correct_predictions = (torch.abs(predictions - targets) < 0.1).sum().item()
    total_predictions = targets.numel()
    
    data_rate = correct_predictions / total_predictions
    spectral_efficiency = data_rate / bandwidth
    
    return spectral_efficiency

def calculate_decoding_complexity(model, sequence_length, batch_size=1):
    """
    Оценка сложности декодирования
    Используем количество параметров и операций как приближение
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if hasattr(model, 'lstm') or any(isinstance(layer, nn.LSTM) for layer in model.modules()):
        lstm_layers = [layer for layer in model.modules() if isinstance(layer, nn.LSTM)]
        if lstm_layers:
            lstm = lstm_layers[0]
            hidden_size = lstm.hidden_size
            num_layers = lstm.num_layers
            complexity = sequence_length * batch_size * num_layers * (hidden_size ** 2)
        else:
            complexity = total_params * sequence_length
    else:
        complexity = total_params * sequence_length
    
    return {
        'total_parameters': total_params,
        'computational_complexity': complexity,
        'complexity_score': total_params + complexity
    }
