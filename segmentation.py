import numpy as np
from collections import deque

"""Módulo de segmentacao com suas pipelines.

Uso:
1. Instanciar `Segmentation` com os limiares e kernel desejados.
2. Ajustar parametros em tempo de execucao com `set_params`, se fizer sentido.
3. Chamar `closing_pipeline` e `sobel_watershed_pipeline`.

Os demais metodos sao internos (prefixo `_`) e existem para compor as etapas
dos pipelines.
"""

class Segmentation:
    """Classe de alto nivel para segmentacao por Fechamento e Sobel+Watershed."""

    def __init__(self, intensity_threshold=128, gradient_threshold=30, watershed_threshold=50, kernel_size=3):
        """Inicializa os limiares e parametros morfologicos usados nas pipelines."""
        self.intensity_threshold = intensity_threshold
        self.gradient_threshold = gradient_threshold
        self.watershed_threshold = watershed_threshold
        self.kernel_size = kernel_size

    def set_params(self, intensity_threshold=None, gradient_threshold=None, watershed_threshold=None, kernel_size=None):
        """Atualiza parametros da instancia sem necessidade de reinstanciar a classe."""
        if intensity_threshold is not None:
            self.intensity_threshold = intensity_threshold
        if gradient_threshold is not None:
            self.gradient_threshold = gradient_threshold
        if watershed_threshold is not None:
            self.watershed_threshold = watershed_threshold
        if kernel_size is not None:
            self.kernel_size = kernel_size
    
    def _to_grayscale(self, image):
        """Converte uma imagem colorida (RGB) para tons de cinza usando a fórmula de luminância padrão."""
        if image.ndim == 3:
            return (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]).astype(np.uint8)
        return image

    def _limiar(self, image, threshold):
        # Se o threshold for o padrão (128), usamos a média da própria imagem
        actual_threshold = threshold if threshold != 128 else np.mean(image)
        return (image >= actual_threshold).astype(np.uint8) * 255
    
    def _erosion(self, image):
        """Aplica erosao binaria usando um kernel quadrado de tamanho configuravel."""
        pad = self.kernel_size // 2
        padded = np.pad(image, pad, mode='edge')
        out = np.zeros_like(image)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                window = padded[y:y + self.kernel_size, x:x + self.kernel_size]
                if np.all(window == 255):
                    out[y, x] = 255
        return out
    
    def _dilation(self, image):
        """Aplica dilatacao binaria usando um kernel quadrado de tamanho configuravel."""
        pad = self.kernel_size // 2
        padded = np.pad(image, pad, mode='edge')
        out = np.zeros_like(image)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                window = padded[y:y + self.kernel_size, x:x + self.kernel_size]
                if np.any(window == 255):
                    out[y, x] = 255
        return out

    def closing_pipeline(self, image, already_binary=False):
        """Executa fechamento morfologico (dilatacao seguida de erosao)."""
        binary_image = image if already_binary else self._limiar(image, self.intensity_threshold)
        dilated = self._dilation(binary_image)
        return self._erosion(dilated)
    
    def _convolution(self, img_gray, kernel):
        """Calcula a convolucao 2D manual de uma imagem em tons de cinza com um kernel."""
        h, w = img_gray.shape
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2
        kernel_flipped = kernel[::-1, ::-1]
        padded = np.pad(img_gray, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        out = np.zeros_like(img_gray, dtype=np.float32)
        for y in range(h):
            for x in range(w):
                window = padded[y:y + kh, x:x + kw]
                out[y, x] = np.sum(window * kernel_flipped)
        return out

    def _gaussian_kernel(self, size, sigma):
        """Gera um kernel Gaussiano 2D normalizado."""
        ax = np.linspace(-(size // 2), size // 2, size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        return kernel / np.sum(kernel)

    def _sobel(self, img_blur):
        """Calcula magnitude e direcao do gradiente com os filtros de Sobel."""
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        gx = self._convolution(img_blur, sobel_x)
        gy = self._convolution(img_blur, sobel_y)
        magnitude = np.sqrt(gx**2 + gy**2).astype(np.float32)
        max_mag = float(np.max(magnitude))
        if max_mag > 0.0:
            magnitude = (magnitude / max_mag) * 255.0
        return magnitude
    
    def _distance_transform(self, binary_image):
        """Estima a distancia de cada pixel de primeiro plano ate o fundo."""
        h, w = binary_image.shape
        dist = np.where(binary_image == 0, 0.0, np.inf).astype(np.float32)

        for y in range(1, h):
            for x in range(1, w):
                if dist[y, x] != 0.0:
                    dist[y, x] = min(dist[y, x], dist[y - 1, x] + 1.0, dist[y, x - 1] + 1.0)

        for y in range(h - 2, -1, -1):
            for x in range(w - 2, -1, -1):
                if dist[y, x] != 0.0:
                    dist[y, x] = min(dist[y, x], dist[y + 1, x] + 1.0, dist[y, x + 1] + 1.0)

        return dist

    def _get_markers(self, binary_image):
        """Cria marcadores iniciais para o watershed a partir do mapa de distancia."""
        dist = self._distance_transform(binary_image)
        finite_values = dist[np.isfinite(dist)]
        if finite_values.size == 0:
            return np.zeros_like(binary_image, dtype=np.int32)

        max_dist = float(np.max(finite_values))
        if max_dist == 0.0:
            seed_mask = (binary_image > 0)
        else:
            seed_mask = (dist > (0.5 * max_dist))
            if not np.any(seed_mask):
                seed_mask = (dist == max_dist)

        return self._label_seed_components(seed_mask)

    def _label_seed_components(self, seed_mask):
        """Rotula componentes conexas para gerar marcadores distintos no watershed."""
        h, w = seed_mask.shape
        labels = np.zeros((h, w), dtype=np.int32)
        current_label = 0

        for y in range(h):
            for x in range(w):
                if not seed_mask[y, x] or labels[y, x] != 0:
                    continue

                current_label += 1
                self._dfs_label_component(seed_mask, labels, y, x, current_label)

        return labels

    def _dfs_label_component(self, seed_mask, labels, start_y, start_x, label_id):
        """Expande uma componente conexa usando DFS iterativo."""
        h, w = seed_mask.shape
        stack = [(start_y, start_x)]
        labels[start_y, start_x] = label_id

        while stack:
            cy, cx = stack.pop()
            for ny in range(max(0, cy - 1), min(h, cy + 2)):
                for nx in range(max(0, cx - 1), min(w, cx + 2)):
                    if seed_mask[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = label_id
                        stack.append((ny, nx))

    def _bfs_propagate_labels(self, magnitude, labels):
        """Propaga os marcadores para vizinhos validos com BFS."""
        h, w = labels.shape
        threshold = float(self.watershed_threshold)
        queue = deque()

        ys, xs = np.where(labels > 0)
        for y, x in zip(ys.tolist(), xs.tolist()):
            queue.append((y, x))

        while queue:
            cy, cx = queue.popleft()
            current_label = labels[cy, cx]

            for ny in range(max(0, cy - 1), min(h, cy + 2)):
                for nx in range(max(0, cx - 1), min(w, cx + 2)):
                    if labels[ny, nx] != 0 or magnitude[ny, nx] >= threshold:
                        continue
                    labels[ny, nx] = current_label
                    queue.append((ny, nx))

        return labels

    def _watershed_segmentation(self, magnitude, markers):
        """Propaga marcadores em regioes de baixo gradiente para segmentar a imagem."""
        labels = markers.copy()
        return self._bfs_propagate_labels(magnitude, labels)

    def sobel_watershed_pipeline(self, image):
        """Executa pipeline de segmentacao com suavizacao, Sobel e watershed."""
        img_blur = self._convolution(image, self._gaussian_kernel(3, 1.0))
        magnitude = self._sobel(img_blur)
        mask = (magnitude < float(self.gradient_threshold)).astype(np.uint8) * 255
        markers = self._get_markers(mask)
        return self._watershed_segmentation(magnitude, markers)