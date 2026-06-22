from pathlib import Path
import random

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from segmentation import Segmentation
from compress.btc import aplicar_compressao_btc, converter_para_cinza_na_mao
from frequency_compression import frequency_compression_step


DATASET_DIR = Path("crater_images")
OUTPUT_DIR = Path("results")


def sample_dataset(dataset_dir, n_samples, seed=42):

    paths = sorted(dataset_dir.glob("*.png"))

    random.seed(seed)

    return random.sample(paths, min(n_samples, len(paths)))


def build_original_vector(image_paths):

    images = []

    for img_path in image_paths:

        image = mpimg.imread(img_path)

        # Remove canal alpha
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]

        # Converte float [0,1] -> uint8 [0,255]
        if image.dtype != np.uint8 and image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        image = converter_para_cinza_na_mao(image)

        images.append(image)

    return images


def save_image(image, path, cmap="gray"):

    plt.figure(figsize=(6, 6))

    plt.imshow(image, cmap=cmap)

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        path,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()


def compare_segmentations(originals, segmenter, n_samples=5, seed=42):

    OUTPUT_DIR.mkdir(exist_ok=True)

    random.seed(seed)

    indices = random.sample(
        range(len(originals)),
        min(n_samples, len(originals))
    )

    for count, idx in enumerate(indices, start=1):

        print(f"[{count}/{len(indices)}] Processando imagem {idx}")

        sample_dir = OUTPUT_DIR / f"sample_{idx}"
        sample_dir.mkdir(exist_ok=True)

        original = originals[idx]

        # ======================
        # Compressões
        # ======================

        btc_img = aplicar_compressao_btc(original)

        dct_img = frequency_compression_step(original)

        # ======================
        # Closing
        # ======================

        closing_original = segmenter.closing_pipeline(original)

        closing_btc = segmenter.closing_pipeline(btc_img)

        closing_dct = segmenter.closing_pipeline(dct_img)

        # ======================
        # Watershed
        # ======================

        watershed_original = segmenter.sobel_watershed_pipeline(original)

        watershed_btc = segmenter.sobel_watershed_pipeline(btc_img)

        watershed_dct = segmenter.sobel_watershed_pipeline(dct_img)

        # ======================
        # Salvar imagens
        # ======================

        save_image(
            original,
            sample_dir / "original.png"
        )

        save_image(
            btc_img,
            sample_dir / "btc.png"
        )

        save_image(
            dct_img,
            sample_dir / "dct.png"
        )

        save_image(
            closing_original,
            sample_dir / "closing_original.png"
        )

        save_image(
            closing_btc,
            sample_dir / "closing_btc.png"
        )

        save_image(
            closing_dct,
            sample_dir / "closing_dct.png"
        )

        save_image(
            watershed_original,
            sample_dir / "watershed_original.png",
            cmap="viridis"
        )

        save_image(
            watershed_btc,
            sample_dir / "watershed_btc.png",
            cmap="viridis"
        )

        save_image(
            watershed_dct,
            sample_dir / "watershed_dct.png",
            cmap="viridis"
        )

    print("\nImagens salvas em:", OUTPUT_DIR)


def main():

    segmenter = Segmentation()

    sampled_paths = sample_dataset(
        DATASET_DIR,
        n_samples=357
    )

    originals = build_original_vector(sampled_paths)

    compare_segmentations(
        originals,
        segmenter,
        n_samples=2
    )


if __name__ == "__main__":
    main()