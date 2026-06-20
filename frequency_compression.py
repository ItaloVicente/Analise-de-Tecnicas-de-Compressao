import numpy as np

# Pré-computar a matriz de transformada DCT 8x8 para eficiência
def _build_dct_matrix():
    N = 8
    A = np.zeros((N, N), dtype=np.float32)
    for u in range(N):
        for x in range(N):
            if u == 0:
                A[u, x] = np.sqrt(1.0 / N)
            else:
                A[u, x] = np.sqrt(2.0 / N) * np.cos((2.0 * x + 1.0) * u * np.pi / (2.0 * N))
    return A

DCT_MATRIX = _build_dct_matrix()
DCT_MATRIX_T = DCT_MATRIX.T

def dct2_manual(block):
    """Aplica DCT 2D em um bloco 8x8 usando multiplicação de matrizes manuais"""
    return DCT_MATRIX @ block @ DCT_MATRIX_T

def idct2_manual(block):
    """Aplica IDCT 2D em um bloco 8x8 usando multiplicação de matrizes manuais"""
    return DCT_MATRIX_T @ block @ DCT_MATRIX

def get_quantization_matrix(quality_factor=50):
    """
    Gera uma matriz de quantização 8x8 baseada na matriz de luminância padrão JPEG
    e ajustada por um fator de qualidade (1-100).
    """
    base_matrix = np.array([
        [16,  11,  10,  16,  24,  40,  51,  61],
        [12,  12,  14,  19,  26,  58,  60,  55],
        [14,  13,  16,  24,  40,  57,  69,  56],
        [14,  17,  22,  29,  51,  87,  80,  62],
        [18,  22,  37,  56,  68, 109, 103,  77],
        [24,  35,  55,  64,  81, 104, 113,  92],
        [49,  64,  78,  87, 103, 121, 120, 101],
        [72,  92,  95,  98, 112, 100, 103,  99]
    ], dtype=np.float32)

    quality_factor = max(1, min(100, quality_factor))
    if quality_factor < 50:
        scale = 5000 / quality_factor
    else:
        scale = 200 - quality_factor * 2

    quant_matrix = np.floor((base_matrix * scale + 50) / 100)
    quant_matrix[quant_matrix < 1] = 1
    quant_matrix[quant_matrix > 255] = 255
    return quant_matrix

def frequency_compression_step(image, quant_matrix=None, quality_factor=50):
    """
    Módulo de Compressão Frequencial
    
    Etapa 1: Transformada Cosseno Discreta (DCT) - Implementação Manual
    Etapa 2: Quantização Frequencial (Compressão)
    Etapa 3: Descompressão (IDCT) - Implementação Manual
    
    Recebe:
        image: matriz bidimensional numpy (tons de cinza).
        quant_matrix: (opcional) matriz 8x8 customizada de quantização.
        quality_factor: (opcional) valor de 1 a 100 para escalar a matriz base.
        
    Retorna:
        output_image: imagem reconstruída no domínio espacial, contendo
                      artefatos da compressão, pronta para filtros subsequentes.
    """
    if quant_matrix is None:
        quant_matrix = get_quantization_matrix(quality_factor)
        
    h, w = image.shape
    
    # Padding para garantir que a imagem seja múltiplo de 8
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    
    if pad_h > 0 or pad_w > 0:
        padded_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='edge')
    else:
        padded_image = image.copy()
        
    ph, pw = padded_image.shape
    output_image = np.zeros_like(padded_image, dtype=np.float32)
    
    # Processamento bloco a bloco (8x8)
    for i in range(0, ph, 8):
        for j in range(0, pw, 8):
            block = padded_image[i:i+8, j:j+8].astype(np.float32)
            
            # Centralizando valores no entorno de zero para melhor desempenho da DCT
            block_centered = block - 128.0
            
            # Etapa 1: Transformada Cosseno Discreta (DCT) Manual
            dct_block = dct2_manual(block_centered)
            
            # Etapa 2: Quantização Frequencial (Compressão - etapa destrutiva)
            # Divisão escalar e arredondamento para o inteiro mais próximo
            quantized_block = np.round(dct_block / quant_matrix)
            
            # Etapa 3: Descompressão (IDCT) Manual
            # Multiplicação pelos pesos da matriz para reversão
            dequantized_block = quantized_block * quant_matrix
            
            # Transformada Inversa (IDCT) Manual
            idct_block = idct2_manual(dequantized_block)
            
            # Revertendo a centralização
            reconstructed_block = idct_block + 128.0
            
            output_image[i:i+8, j:j+8] = reconstructed_block
            
    # Removendo padding e recuperando dimensões originais da imagem de entrada
    output_image = output_image[:h, :w]
    
    # Garantindo valores de saída no intervalo dos pixels [0, 255]
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    
    return output_image
