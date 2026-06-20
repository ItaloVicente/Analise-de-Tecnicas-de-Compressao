import numpy as np

class Segmentation:
    def __init__(self, intensity_threshold=128, gradient_threshold=30, watershed_threshold=50, kernel_size=3):
        """Inicializa os limiares e parametros morfologicos usados nas pipelines."""
        self.intensity_threshold = intensity_threshold
        self.gradient_threshold = gradient_threshold
        self.watershed_threshold = watershed_threshold
        self.kernel_size = kernel_size
    
    def to_grayscale(self, image):
        """Converte uma imagem colorida (RGB) para tons de cinza usando a fórmula de luminância padrão."""
        if image.ndim == 3:
            return (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]).astype(np.uint8)
        return image

    def limiar(self, image, threshold):
        """Binariza a imagem em 0/255 usando o limiar informado."""
        img_gray = self.to_grayscale(image)
        return (img_gray >= threshold).astype(np.uint8) * 255
    
    def erosion(self, image):
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
    
    def dilation(self, image):
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
        binary_image = image if already_binary else self.limiar(image, self.intensity_threshold)
        dilated = self.dilation(binary_image)
        return self.erosion(dilated)
    
    def convolution(self, img_gray, kernel):
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

    def gaussian_kernel(self, size, sigma):
        """Gera um kernel Gaussiano 2D normalizado."""
        ax = np.linspace(-(size // 2), size // 2, size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        return kernel / np.sum(kernel)

    def sobel(self, img_blur):
        """Calcula magnitude e direcao do gradiente com os filtros de Sobel."""
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        gx = self.convolution(img_blur, sobel_x)
        gy = self.convolution(img_blur, sobel_y)
        return np.sqrt(gx**2 + gy**2)
    
    def distance_transform(self, binary_image):
        """Estima a distancia de cada pixel de primeiro plano ate o fundo."""
        h, w = binary_image.shape
        dist = np.where(binary_image == 0, 0, np.inf).astype(np.float32)
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if dist[y, x] != 0:
                    dist[y, x] = np.min([dist[y-1, x], dist[y+1, x], 
                                         dist[y, x-1], dist[y, x+1]]) + 1
        return dist

    def get_markers(self, binary_image):
        """Cria marcadores iniciais para o watershed a partir do mapa de distancia."""
        dist = self.distance_transform(binary_image)
        return (dist > (0.5 * np.max(dist))).astype(np.int32)

    def watershed_segmentation(self, magnitude, markers):
        """Propaga marcadores em regioes de baixo gradiente para segmentar a imagem."""
        h, w = magnitude.shape
        labels = markers.copy()
        changed = True
        while changed:
            changed = False
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    if labels[y, x] == 0 and magnitude[y, x] < self.watershed_threshold:
                        neighbors = labels[y-1:y+2, x-1:x+2]
                        if np.any(neighbors > 0):
                            labels[y, x] = np.max(neighbors)
                            changed = True
        return labels

    def sobel_watershed_pipeline(self, image):
        """Executa pipeline de segmentacao com suavizacao, Sobel e watershed."""
        img_gray = self.to_grayscale(image).astype(np.float32)
        img_blur = self.convolution(img_gray, self.gaussian_kernel(3, 1.0))
        magnitude = self.sobel(img_blur)
        mask = 255 - self.limiar(magnitude, self.gradient_threshold)
        markers = self.get_markers(mask)
        return self.watershed_segmentation(magnitude, markers)