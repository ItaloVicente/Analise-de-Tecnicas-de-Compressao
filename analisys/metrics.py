import numpy as np

def iou(y_true, y_pred):
    """Calcula o IoU para uma única imagem."""
    y_true_bin = (y_true > 0)
    y_pred_bin = (y_pred > 0)

    intersection = np.logical_and(y_true_bin, y_pred_bin).sum()
    union = np.logical_or(y_true_bin, y_pred_bin).sum()

    if union == 0:
        return 1.0

    return intersection / union

def mean_iou(y_trues, y_preds):
    """Calcula a média do IoU para um lote (batch) de imagens."""
    # Converte a lista de imagens em um array 3D e binariza tudo de uma vez
    y_trues_bin = np.array(y_trues) > 0
    y_preds_bin = np.array(y_preds) > 0

    # Calcula intersecção e união somando ao longo dos eixos X e Y (axis=1 e 2)
    intersections = np.logical_and(y_trues_bin, y_preds_bin).sum(axis=(1, 2))
    unions = np.logical_or(y_trues_bin, y_preds_bin).sum(axis=(1, 2))

    # np.where evita a divisão por zero: se a união for 0, o IoU é 1.0
    ious = np.divide(intersections, unions, out=np.ones_like(intersections, dtype=float), where=(unions != 0))
    
    return ious.mean()

def dice(y_true, y_pred):
    """Calcula o Dice coefficient para uma única imagem."""
    y_true_bin = (y_true > 0)
    y_pred_bin = (y_pred > 0)

    intersection = np.logical_and(y_true_bin, y_pred_bin).sum()
    denominator = y_true_bin.sum() + y_pred_bin.sum()

    if denominator == 0:
        return 1.0

    return (2.0 * intersection) / denominator

def mean_dice(y_trues, y_preds):
    """Calcula a média do Dice para um lote (batch) de imagens."""
    y_trues_bin = np.array(y_trues) > 0
    y_preds_bin = np.array(y_preds) > 0

    intersections = np.logical_and(y_trues_bin, y_preds_bin).sum(axis=(1, 2))
    denominators = y_trues_bin.sum(axis=(1, 2)) + y_preds_bin.sum(axis=(1, 2))

    dices = np.divide((2.0 * intersections), denominators, out=np.ones_like(denominators, dtype=float), where=(denominators != 0))
    
    return dices.mean()

def pixel_accuracy(y_trues, y_preds):
    """
    Calcula a acurácia global de pixels para todo o lote.
    Compara todas as matrizes inteiras em uma única operação.
    """
    y_trues_bin = np.array(y_trues) > 0
    y_preds_bin = np.array(y_preds) > 0

    # .mean() numa matriz booleana já retorna a proporção de acertos (True)
    return (y_trues_bin == y_preds_bin).mean()

def changed_pixels(y_trues, y_preds):
    """
    Calcula a taxa de pixels errados (diferentes) para todo o lote.
    """
    y_trues_bin = np.array(y_trues) > 0
    y_preds_bin = np.array(y_preds) > 0

    # Proporção de onde os pixels divergem (True)
    return (y_trues_bin != y_preds_bin).mean()