import os
import math
import numpy as np
import cv2


def converter_para_cinza_na_mao(imagem_bgr): #Usada nas questões do trabalho
    canal_b = imagem_bgr[:, :, 0]
    canal_g = imagem_bgr[:, :, 1]
    canal_r = imagem_bgr[:, :, 2]

    imagem_cinza = 0.299 * canal_r + 0.587 * canal_g + 0.114 * canal_b

    return np.clip(imagem_cinza, 0, 255).astype(np.uint8)


def aplicar_compressao_btc(imagem_cinza, tamanho_bloco=4):
    altura, largura = imagem_cinza.shape

    altura_valida = altura - (altura % tamanho_bloco)
    largura_valida = largura - (largura % tamanho_bloco)
    imagem_recortada = imagem_cinza[:altura_valida, :largura_valida].astype(np.float64)

    imagem_reconstruida = np.zeros_like(imagem_recortada)

    m = tamanho_bloco * tamanho_bloco

    for i in range(0, altura_valida, tamanho_bloco):
        for j in range(0, largura_valida, tamanho_bloco):
            bloco = imagem_recortada[i:i + tamanho_bloco, j:j + tamanho_bloco]

            mu = np.mean(bloco)
            sigma = np.std(bloco)

            mapa_de_bits = bloco >= mu

            q = np.sum(mapa_de_bits)

            if q == 0 or q == m:
                cor_a = mu
                cor_b = mu
            else:
                cor_a = mu - sigma * math.sqrt(q / (m - q))
                cor_b = mu + sigma * math.sqrt((m - q) / q)

            bloco_reconstruido = np.zeros((tamanho_bloco, tamanho_bloco))

            bloco_reconstruido[mapa_de_bits == False] = cor_a
            bloco_reconstruido[mapa_de_bits == True] = cor_b

            imagem_reconstruida[i:i + tamanho_bloco, j:j + tamanho_bloco] = bloco_reconstruido

    return np.clip(imagem_reconstruida, 0, 255).astype(np.uint8)


def executar_pipeline_btc():
    pasta_entrada = '../crater_images/'
    pasta_saida = 'crater_images_btc/'

    if not os.path.exists(pasta_saida):
        os.makedirs(pasta_saida)

    for nome_arquivo in os.listdir(pasta_entrada):
        if nome_arquivo.lower().endswith(('.png', '.jpg', '.jpeg')):
            caminho_entrada = os.path.join(pasta_entrada, nome_arquivo)
            caminho_saida = os.path.join(pasta_saida, nome_arquivo)

            print(f"Processando imagem via BTC: {nome_arquivo}...")

            imagem_bgr = cv2.imread(caminho_entrada)
            imagem_cinza = converter_para_cinza_na_mao(imagem_bgr)
            imagem_degradada = aplicar_compressao_btc(imagem_cinza, tamanho_bloco=4)

            cv2.imwrite(caminho_saida, imagem_degradada)

    print("Processamento BTC concluído com sucesso!")

if __name__ == "__main__":
    executar_pipeline_btc()