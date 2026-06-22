from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import random

from pathlib import Path
from segmentation import Segmentation
from analisys.metrics import mean_iou, pixel_accuracy, mean_dice, changed_pixels
from compress .btc import aplicar_compressao_btc, converter_para_cinza_na_mao
from frequency_compression import frequency_compression_step


DATASET_DIR = Path("crater_images")

def sample_dataset(dataset_dir, n_samples, seed=42):
    paths = sorted(dataset_dir.glob("*.png"))
    
    random.seed(seed)
    sampled_paths = random.sample(paths, n_samples)
    
    return sampled_paths


def build_original_vector(image_paths):
    images = []

    for img_path in image_paths:
        image = mpimg.imread(img_path)
        
        # 1. Garante que removemos o canal Alpha (transparência) se for um PNG de 4 canais
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
            
        # 2. Converte de float [0.0, 1.0] para uint8 [0, 255] ANTES de qualquer processamento
        if image.dtype != np.uint8 and image.max() <= 1.0:
            image = np.clip(image * 255, 0, 255).astype(np.uint8)

        # 3. Agora sua função recebe uma matriz puramente de inteiros (0-255) com 3 canais
        image = converter_para_cinza_na_mao(image)

        images.append(image)

    return images

def build_compressed_vector(images, compress_fn):

    compressed_images = []

    for img in images:
        compressed_images.append(compress_fn(img))

    return compressed_images

def build_segmented_vector(images, segment_fn):

    segmented = []

    for img in images:
        segmented.append(segment_fn(img))

    return segmented

def evaluate(name, original, compressed):

    print(f"\n=== {name.upper()} ===")
    print(f"mIoU: {mean_iou(original, compressed):.4f}")
    print(f"PA: {pixel_accuracy(original, compressed):.4f}")
    print(f"mDice:  {mean_dice(original, compressed):.4f}")
    print(f"Changed Pixels: {changed_pixels(original, compressed)}")


def main():

    segmenter = Segmentation()

    # 📌 carregar dataset base
    sampled_paths = sample_dataset(DATASET_DIR, 357)
    originals = build_original_vector(sampled_paths)

    # =========================
    # 📌 BTC
    # =========================
    btc_compressed = build_compressed_vector(
        originals,
        aplicar_compressao_btc
    )

    btc_seg_orig = build_segmented_vector(
        originals,
        segmenter.closing_pipeline
    )

    btc_seg_comp = build_segmented_vector(
        btc_compressed,
        segmenter.closing_pipeline
    )

    evaluate(
        "CLOSING + BTC",
        btc_seg_orig,
        btc_seg_comp
    )

    btc_seg_orig_w = build_segmented_vector(
        originals,
        segmenter.sobel_watershed_pipeline
    )

    btc_seg_comp_w = build_segmented_vector(
        btc_compressed,
        segmenter.sobel_watershed_pipeline
    )

    evaluate(
        "WATERSHED + BTC",
        btc_seg_orig_w,
        btc_seg_comp_w
    )

    # =========================
    # 📌 DCT
    # =========================
    dct_compressed = build_compressed_vector(
        originals,
        frequency_compression_step
    )

    dct_seg_orig = build_segmented_vector(
        originals,
        segmenter.closing_pipeline
    )

    dct_seg_comp = build_segmented_vector(
        dct_compressed,
        segmenter.closing_pipeline
    )

    evaluate(
        "CLOSING + DCT",
        dct_seg_orig,
        dct_seg_comp
    )

    dct_seg_orig_w = build_segmented_vector(
        originals,
        segmenter.sobel_watershed_pipeline
    )

    dct_seg_comp_w = build_segmented_vector(
        dct_compressed,
        segmenter.sobel_watershed_pipeline
    )

    evaluate(
        "WATERSHED + DCT",
        dct_seg_orig_w,
        dct_seg_comp_w
    )


if __name__ == "__main__":
    main()