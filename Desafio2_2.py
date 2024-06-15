import numpy as np
from PIL import Image
from func_desafio_dio import open_image, grayscale, binarize 

"""
Criando função main do script, onde a mesma le o arquivo passado via codigo e as funções anteriores mudam a imagem de Lance Armstrong, sim sou ciclista e fã, para escalas de cinza deixando 
a mesma também binarizada (preto e branco)
"""
def main():
    # Carregando a imagem.
    input_image_path = r"D:\A1-Leandro\Pessoal\DIO\Desafio2\Desafio_dio\lance.jpg"
    original_image = open_image(input_image_path)

    # Convertendo para tons de cinza.
    grayscale_image = grayscale(original_image)

    # Salvando a imagem em tons de cinza.
    grayscale_output_path = r"D:\A1-Leandro\Pessoal\DIO\Desafio2\Desafio_dio\imagem_tons_de_cinza.jpg"
    Image.fromarray(grayscale_image).save(grayscale_output_path)

    # Binarizando a imagem.
    binarized_image = binarize(grayscale_image)

    # Salvando a imagem binarizada.
    binarized_output_path = r"D:\A1-Leandro\Pessoal\DIO\Desafio2\Desafio_dio\imagem_binarizada.jpg"
    Image.fromarray(binarized_image).save(binarized_output_path)

if __name__ == "__main__":
    main()