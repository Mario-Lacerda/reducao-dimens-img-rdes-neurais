import numpy as np
from PIL import Image
def open_image(file_path):
    """
    Abre a imagem e retorna uma matriz numpy representando os pixels.
    """
    image = Image.open(file_path)
    return np.array(image)

def grayscale(image):
    """
    Converte uma imagem colorida para tons de cinza.
    """
    # Calcula a luminosidade de cada pixel.
    luminosidade = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]

    # Cria uma imagem em tons de cinza com a luminosidade calculada.
    grayscale_image = luminosidade.astype(np.uint8)

    return grayscale_image

def binarize(image, threshold=128):
    """
    Binariza uma imagem em tons de cinza.
    Pixels abaixo do limiar tornam-se pretos (0), e pixels acima tornam-se brancos (255).
    """
    binarized_image = (image > threshold) * 255
    return binarized_image.astype(np.uint8)