import numpy as np

class Segmentation:
    def __init__(self, limiar_threshold=128, kernel_size=3):
        self.threshold = limiar_threshold
        self.kernel_size = kernel_size
    
    def to_grayscale(self, image):
        """Converte RGB para Cinza usando a fórmula de luminância."""
        if image.ndim == 3:
            return (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]).astype(np.uint8)
        return image

    def limiar(self, image):
        """Aplica o limiar para segmentação."""
        img_gray = self.to_grayscale(image)
        return (img_gray >= self.threshold).astype(np.uint8) * 255
    
    def erosion(self, image):
        """Aplica a operação de erosão morfológica."""
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
        """Aplica a operação de dilatação morfológica."""
        pad = self.kernel_size // 2
        padded = np.pad(image, pad, mode='edge')
        out = np.zeros_like(image)
        
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                window = padded[y:y + self.kernel_size, x:x + self.kernel_size]
                if np.any(window == 255):
                    out[y, x] = 255
        return out

    def closing(self, image):
        """Aplica a operação de fechamento morfológico."""
        binary_image = self.limiar(image)
        dilated = self.dilation(binary_image)
        eroded = self.erosion(dilated)
        return eroded
    
    def sobel(self, image):
        """Aplica o filtro de Sobel para detecção de bordas."""
        #TODO
        pass
    
    #function to markers to watershed
    def watershed(self, image):
        """Aplica o algoritmo de Watershed para segmentação."""
        
        #TODO
        pass