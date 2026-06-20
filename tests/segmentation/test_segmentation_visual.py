from pathlib import Path
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from segmentation import Segmentation


DATASET_DIR = PROJECT_ROOT / "crater_images"


def _load_first_n_images(n: int = 3):
    """Carrega as n primeiras imagens numeradas do dataset (1.png ... n.png)."""
    images = []

    for idx in range(1, n + 1):
        image_path = DATASET_DIR / f"{idx}.png"
        assert image_path.exists(), f"Imagem nao encontrada: {image_path}"

        image_rgb = mpimg.imread(image_path)
        assert image_rgb is not None, f"Falha ao ler imagem: {image_path}"

        if image_rgb.dtype != np.uint8:
            image_rgb = np.clip(image_rgb * 255, 0, 255).astype(np.uint8)

        images.append((image_path.name, image_rgb))

    return images


def test_segmentation_visual_pipelines_show():
    """Executa as pipelines e exibe grade 3x3 para validacao visual."""
    samples = _load_first_n_images(n=3)
    assert len(samples) == 3

    segmenter = Segmentation()

    fig, axes = plt.subplots(len(samples), 3, figsize=(12, 3 * len(samples)), squeeze=False)

    for row, (image_name, image_rgb) in enumerate(samples):
        # A partir da segunda imagem, muda parametros para cobrir set_params em execucao.
        if row == 1:
            segmenter.set_params(
                kernel_size=5,
                intensity_threshold=110,
                gradient_threshold=35,
                watershed_threshold=45,
            )

        closing_output = segmenter.closing_pipeline(image_rgb)
        watershed_output = segmenter.sobel_watershed_pipeline(image_rgb)

        assert closing_output is not None
        assert watershed_output is not None
        assert closing_output.shape == image_rgb.shape[:2]
        assert watershed_output.shape == image_rgb.shape[:2]
        assert np.issubdtype(closing_output.dtype, np.integer)
        assert np.issubdtype(watershed_output.dtype, np.integer)

        axes[row, 0].imshow(image_rgb)
        axes[row, 0].set_title(f"{image_name} | original")

        axes[row, 1].imshow(closing_output, cmap="gray", vmin=0, vmax=255)
        axes[row, 1].set_title("closing_pipeline")

        axes[row, 2].imshow(watershed_output, cmap="viridis")
        axes[row, 2].set_title("sobel_watershed_pipeline")

        for col in range(3):
            axes[row, col].axis("off")

    fig.tight_layout()
    plt.show()
    plt.close(fig)
