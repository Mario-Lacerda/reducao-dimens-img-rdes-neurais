# Desafio Dio - Projeto para redução de dimensionalidade em imagens.



A redução de dimensionalidade é uma técnica usada para reduzir o número de recursos em um conjunto de dados, mantendo as informações mais importantes. Isso pode ser útil para redes neurais, pois pode ajudar a melhorar o desempenho e reduzir o tempo de treinamento.



#### Existem vários algoritmos diferentes que podem ser usados para redução de dimensionalidade, incluindo:

- **Análise de Componentes Principais (PCA)**: PCA é um algoritmo linear que projeta os dados em um novo espaço com um número menor de dimensões.

  

- **Análise de Componentes Independentes (ICA)**: ICA é um algoritmo não linear que projeta os dados em um novo espaço onde os componentes são estatisticamente independentes.

  

- **Autoencoders**: Autoencoders são redes neurais que são treinadas para reconstruir seus dados de entrada. Eles podem ser usados para redução de dimensionalidade aprendendo uma representação compacta dos dados.

A escolha do algoritmo de redução de dimensionalidade depende dos dados específicos e da tarefa em questão.



#### **Exemplo de redução de dimensionalidade usando PCA**

Aqui está um exemplo de como usar PCA para redução de dimensionalidade em imagens:

python



```python
import numpy as np
from sklearn.decomposition import PCA

# Carregar as imagens
images = np.load('images.npy')

# Reduzir a dimensionalidade das imagens usando PCA
pca = PCA(n_components=100)
reduced_images = pca.fit_transform(images)

# Salvar as imagens reduzidas
np.save('reduced_images.npy', reduced_images)
```



Este código carrega as imagens, reduz sua dimensionalidade usando PCA e salva as imagens reduzidas.



#### **Exemplo de redução de dimensionalidade usando um autoencoder**

Aqui está um exemplo de como usar um autoencoder para redução de dimensionalidade em imagens:

python



```python
import tensorflow as tf

# Criar o autoencoder
autoencoder = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
  tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
  tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(100),
  tf.keras.layers.Dense(784),
  tf.keras.layers.Reshape((28, 28, 1)),
  tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same'),
  tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same'),
  tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
])

# Compilar o autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

# Treinar o autoencoder
autoencoder.fit(images, images, epochs=10)

# Salvar o autoencoder
autoencoder.save('autoencoder.h5')
```



Este código cria um autoencoder, o compila, o treina e o salva. O autoencoder pode então ser usado para reduzir a dimensionalidade das imagens.



#### **Recursos adicionais**

- Redução de Dimensionalidade em Imagens
- Redução de Dimensionalidade em Machine Learning
- Redução de Dimensionalidade em Imagens para Redes Neurais





<h4
    > Seguindo o exemplo do algoritmo de binarização apresentado em nossa última aula, realize a implementação em Python para transformar uma imagem colorida para níveis de cinza (0 a 255) e para binarizada (0 e 255), preto e branco. <h4/>    


* **Para a implementação, será necessário criar a solução sem utilizar bibliotecas como opencv, ou seja, elaborar o código.**
* **Por meio da imagem é possível visualizar os dois casos esperados:**

<img src="img\lena.png" align='center' alt="drawing" width="603" heigth="221"/>



## Solução do Desafio


```python
# libs para carregar e tratar a imagem
from PIL import Image
import numpy as np

# Para exibir a imagem dentro do jupyter notebook
from IPython.display import display
```


```python
# Caminho para a imagem
image_path = '../img/img.png'

# Carregar a imagem usando a PIL
image = Image.open(image_path)

# Exibir informações sobre a imagem
print("Formato da imagem:", image.format)
print("Modo da imagem:", image.mode)
print("Dimensões da imagem:", image.size)

# Exibir a imagem
display(image)
```

    Formato da imagem: PNG
    Modo da imagem: RGB
    Dimensões da imagem: (512, 512)




![png](img/output_6_1.png)
    




* **Imagem carregada com sucesso, agora vamos criar o código para converter a imagem em escala de cinza.**


```python
# Converter a imagem para um array NumPy
image_array = np.array(image)

# Obter as dimensões da imagem
height, width, _ = image_array.shape

# Criar uma nova matriz para a imagem em escala de cinza
gray_array = np.zeros((height, width), dtype=np.uint8)

# Converter a imagem para escala de cinza
for i in range(height):
    for j in range(width):
        # Obter os valores RGB do pixel
        r, g, b = image_array[i, j]

        # Calcular o valor de escala de cinza usando a média ponderada dos canais RGB
        # link de pesquisa dos valoers: https://capivararex.wordpress.com/2016/04/17/dip02-conversao-rgb-para-escala-de-cinza/
        gray_value = int(0.3 * r + 0.59 * g + 0.11 * b)

        # Atribuir o valor de escala de cinza ao pixel correspondente na nova matriz
        gray_array[i, j] = gray_value

# Criar uma nova imagem a partir da matriz de escala de cinza
gray_image = Image.fromarray(gray_array, mode='L')

# Exibir a imagem em escala de cinza
display(gray_image)
```


​    
![png](img/output_8_0.png)
​    



#### **Observação:** 

* Veja como é simples realizar a conversão utilizando a biblioteca **PIL**.


```python
# Criar uma nova imagem a partir da matriz de escala de cinza
gray_image = image.convert('L')

# Exibir a imagem em escala de cinza
display(gray_image)

```


​    
![png](img/output_10_0.png)
​    



#### **Agora vamos converter a imagem em escala cinza para preto e branca**.


```python
# Converter a imagem em escala de cinza para um array NumPy
gray_array = np.array(gray_image)

# Definir o valor de limiar
threshold = 110

# Criar uma nova matriz para a imagem em preto e branco
bw_array = np.zeros_like(gray_array, dtype=np.uint8)

# Converter a imagem em escala de cinza para preto e branco
bw_array[gray_array < threshold] = 0  # Valor do pixel preto
bw_array[gray_array >= threshold] = 255  # Valor do pixel branco

# Criar uma nova imagem a partir da matriz em preto e branco
bw_image = Image.fromarray(bw_array, mode='L')

# Exibir a imagem em preto e branco
display(bw_image)


```


​    
![png](img/output_12_0.png)
​    



**O valor de limiar(threshold) é definido como 110. Pixels com valores menores que o limiar serão considerados como preto, enquanto pixels com valores maiores ou iguais ao limiar serão considerados como branco. É possível alterar esse valor afim de encontrar a melhor proporção em preto e branco.**



### Salvando as imagens.


```python
# Salvar a imagem em escala de cinza
gray_image.save('../img/imagem_cinza.png')

# Salvar a imagem em preto e branco
bw_image.save('../img/imagem_pb.png')

```



### Comparando o tamanho das imagens


```python
import os

# Obter o tamanho do arquivo da imagem colorida
size_color = os.path.getsize('../img/img.png')
size_mb_color = size_color / (1024 * 1024)  # Converter para megabytes

# Obter o tamanho do arquivo da imagem em escala de cinza
size_gray = os.path.getsize('../img/imagem_cinza.png')
size_mb_gray = size_gray / (1024 * 1024)  # Converter para megabytes

# Obter o tamanho do arquivo da imagem em preto e branco
size_bw = os.path.getsize('../img/imagem_pb.png')
size_mb_bw = size_bw / (1024 * 1024)  # Converter para megabytes

# Exibir o tamanho das imagens em megabytes
print("Tamanho da imagem colorida: {:.2f} MB".format(size_mb_color))
print("Tamanho da imagem em escala de cinza: {:.2f} MB".format(size_mb_gray))
print("Tamanho da imagem em preto e branco: {:.2f} MB".format(size_mb_bw))

```

    Tamanho da imagem colorida: 0.36 MB
    Tamanho da imagem em escala de cinza: 0.12 MB
    Tamanho da imagem em preto e branco: 0.01 MB



### A redução é um processo fundamental na visão computacional por várias razões:



* **Eficiência computacional**: A redução permite diminuir a quantidade de dados de entrada, tornando os algoritmos de visão computacional mais eficientes. Ao reduzir o tamanho das imagens ou a dimensionalidade dos dados, é possível economizar recursos computacionais, como tempo de processamento e armazenamento.

  

* **Eliminação de ruído e detalhes irrelevantes:** Em muitos casos, a redução pode ajudar a eliminar ruídos indesejados nas imagens ou detalhes irrelevantes para a análise ou tarefa em questão. Ao reduzir a informação desnecessária, é possível melhorar a qualidade dos dados e a eficácia dos algoritmos.

  

* **Extração de características:** A redução pode ser usada para extrair características importantes e representativas dos dados. Por exemplo, técnicas de redução de dimensionalidade, como a Análise de Componentes Principais (PCA) ou a Análise de Discriminante Linear (LDA), podem ajudar a identificar as características mais discriminantes e relevantes nos dados.

  

* **Simplificação da representação:** Reduzir a complexidade dos dados pode facilitar sua representação e interpretação. Imagens de alta resolução ou dados de alta dimensionalidade podem ser difíceis de visualizar e compreender. A redução ajuda a simplificar a representação dos dados, tornando-os mais acessíveis e compreensíveis.

  

* **Robustez e generalização:** A redução pode ajudar a melhorar a robustez e a generalização dos modelos de visão computacional. Ao reduzir a dimensionalidade ou a complexidade dos dados, é possível evitar overfitting e aumentar a capacidade dos modelos de generalizar padrões e características para novos dados não vistos anteriormente.







<H1>Descrição do Desafio</H1>

Seguindo o exemplo do algoritmo de binarização apresentado em nossa última aula, realize a implementação em Python para transformar uma imagem colorida para níveis de cinza (0 a 255) e para binarizada (0 e 255), preto e branco.

Antes de rodar o sistema pip install requirements.txt para instalar as bibliotecas necessária.



#### Resultado:



<H2>Imagem original</H2>



<H2>Imagem em tons de cinza</H2>



<img src ="C:\S  O   S O M E N T E        B  O   T    C   A    M   P    S        D    I    O        OK\REDUCAO-DIMENSIN-IMG-REDES-NEURAIS-MARIO2\Redu-o-Dimensionalidade-em-imagens-para-Redes-Neurais\imagem_tons_de_cinza.jpg">



<H2>Imagem Binarizada</H2>



<img src ="imagem_binarizada.jpg">





#### Agradecimento Especial:

###### (autoria e parceria)   -   https://github.com/lerocha1
