import numpy as np
from PIL import Image, ImageFilter # pillow
import matplotlib.pyplot as plt
import math
import cv2
from scipy import ndimage

def plot(img, imgNew, msg):
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("Imagem original")
    ax[1].imshow(imgNew, cmap='gray')
    ax[1].set_title(msg)
    plt.show()

def plotBits(bit_planes):

    for i in range(8):
      plt.subplot(3, 3, i + 2)
      plt.title(f'Plano de Bit {i}')
      plt.imshow(bit_planes[i], cmap='gray', vmin=0, vmax=255)
      plt.axis('off')

    plt.show()

def codifica_cinza(imagem, message):

    message += '\0'
    img = Image.open(imagem).convert('L')

    newImg = np.array(img)

    binary_message = ''.join(format(ord(char), '08b') for char in message)

    if len(binary_message) > newImg.size:
        raise ValueError("A mensagem é muito longa para ser escondida nesta imagem.")

    index = 0
    for i in range(newImg.shape[0]):
        for j in range(newImg.shape[1]):
            if index < len(binary_message):
                newImg[i, j] = (newImg[i, j] & ~1) | int(binary_message[index])
                index += 1
            else:
                break
    img_new = Image.fromarray(newImg)
    img_new.save("house_encoded.tif")

    plot(img, newImg, "Imagem com Mensagem Oculta")


def decodifica_cinza(imagem):
    img = Image.open(imagem).convert('L')
    img_matrix = np.array(img)

    binary_message = ''

    for i in range(img_matrix.shape[0]):
        for j in range(img_matrix.shape[1]):
            binary_message += str(img_matrix[i, j] & 1)

    message = ''
    for i in range(0, len(binary_message), 8):
        char = chr(int(binary_message[i:i+8], 2))
        if char == '\0':
            break
        message += char

    return message

def codifica_rgb(imagem, message):
    img = Image.open(imagem).convert('RGB')

    newImg = np.array(img)

    message += '\0'

    binary_message = ''.join(format(ord(char), '08b') for char in message)

    if len(binary_message) > newImg.size * 3:
        raise ValueError("A mensagem é muito longa para ser escondida nesta imagem.")

    index = 0
    for i in range(newImg.shape[0]):
        for j in range(newImg.shape[1]):
            for k in range(3):
                if index < len(binary_message):
                    newImg[i, j, k] = (newImg[i, j, k] & ~1) | int(binary_message[index])
                    index += 1
                else:
                    break

    img_new = Image.fromarray(newImg)

    plot(img, newImg, "Imagem Colorida com Mensagem Oculta")

    img_new.save('lena_color_encoded.tif')


def decodifica_rgb(imagem):
    img = Image.open(imagem).convert('RGB')

    img_matrix = np.array(img)

    binary_message = ''

    for i in range(img_matrix.shape[0]):
        for j in range(img_matrix.shape[1]):
            for k in range(3):
                binary_message += str(img_matrix[i, j, k] & 1)

    message = ''
    for i in range(0, len(binary_message), 8):
        char = chr(int(binary_message[i:i+8], 2))
        if char == '\0':
            break
        message += char

    return message

def transLog(imagem, c):
    img = Image.open(imagem).convert('L')

    newImg = np.array(img)

    imgLog = c * np.log(1 + newImg)
    imgLog = np.clip(imgLog, 0, 255).astype(np.uint8)

    msg = "Transformacao Logaritmica com c = " + str(c)

    plot(img, imgLog, msg)


def transGama(imagem, c, y):

  img = Image.open(imagem)

  newImg = np.array(img)

  imgGama = c * np.power(newImg, y)
  imgGama = np.clip(imgGama, 0, 255).astype(np.uint8)

  msg = "Transformacao Gama com c = " + str(c) + " e y = " + str(y)

  plot(img, imgGama, msg)


def repBits(imagem):
  img = Image.open(imagem).convert('L')


  newImg = np.array(img)

  bit_planes = []

  for i in range(8):
      bit_plane = (newImg >> i) & 1

      bit_plane = bit_plane * 255
      bit_planes.append(bit_plane)

  plotBits(bit_planes)


def equalizarHistograma(imagem, escolha):
  img = Image.open(imagem).convert('L')

  newImg = np.array(img)

  if(escolha == 0):
    histograma = [0] * 256

    for row in newImg:
        for pixel in row:
            histograma[pixel] += 1


    cdf = [0] * len(histograma)
    soma_acumulada = 0

    for i in range(len(histograma)):
        soma_acumulada += histograma[i]
        cdf[i] = soma_acumulada


    total_pixels = len(newImg) * len(newImg[0])

    cdf_min = min([value for value in cdf if value > 0])

    imagem_equalizada = []

    for row in newImg:
        nova_linha = []
        for pixel in row:
            s_k = round((cdf[pixel] - cdf_min) / (total_pixels - cdf_min) * 255)
            nova_linha.append(s_k)
        imagem_equalizada.append(nova_linha)

    modo = 'na mão'
  elif (escolha == 1):
    hist, bins = np.histogram(newImg.flatten(), bins=256, range=[0, 256])

    cdf = hist.cumsum()

    cdf_normalizada = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())

    imagem_equalizada = np.interp(newImg.flatten(), bins[:-1], cdf_normalizada).reshape(newImg.shape)
    modo = 'numpy'

  elif (escolha == 2):

    img = cv2.imread(imagem, cv2.IMREAD_GRAYSCALE)

    imagem_equalizada = cv2.equalizeHist(img)
    modo = 'opencv'

  elif (escolha == 3):

    hist, bin_edges = np.histogram(img, bins=256, range=(0, 255))
    cdf = np.cumsum(hist) / np.sum(hist)

    imagem_equalizada = np.interp(img, bin_edges[:-1], cdf * 255).astype('uint8')
    modo = 'scipy'

  msg = "Imagem Equalizada - " + modo

  plot(img, imagem_equalizada, msg)

def main():

  #ESTEGANOGRAFIA TONS DE CINZA
  codifica_cinza('img/house.tif', 'processamento')
  msgEscondida = decodifica_cinza('house_encoded.tif')
  print("\nMensagem oculta extraída:", msgEscondida)


  #ESTEGANOGRAFIA RGB
  codifica_rgb('img/lena_color.tif', 'processamento')
  msgEscondida = decodifica_rgb('lena_color_encoded.tif')
  print("\nMensagem oculta extraída:", msgEscondida)

  #Transformacao logaritmica
  transLog('img/fractured_spine.tif', 1)
  transLog('img/fractured_spine.tif', 10)
  transLog('img/fractured_spine.tif', 25)
  transLog('img/fractured_spine.tif', 100)

  transLog('img/enhance-me.tif', 1)
  transLog('img/enhance-me.tif', 10)
  transLog('img/enhance-me.tif', 25)
  transLog('img/enhance-me.tif', 100)

  #Transformacao de Potencia (Gama)
  transGama('img/fractured_spine.tif', 1, 1)
  transGama('img/fractured_spine.tif', 10, 1)
  transGama('img/fractured_spine.tif', 100, 1)

  transGama('img/fractured_spine.tif', 1, 15)
  transGama('img/fractured_spine.tif', 10, 15)
  transGama('img/fractured_spine.tif', 100, 15)

  transGama('img/fractured_spine.tif', 10, 1)
  transGama('img/fractured_spine.tif', 10, 10)
  transGama('img/fractured_spine.tif', 10, 30)

  transGama('img/enhance-me.tif', 1, 1)
  transGama('img/enhance-me.tif', 10, 1)
  transGama('img/enhance-me.tif', 100, 1)

  transGama('img/enhance-me.tif', 1, 15)
  transGama('img/enhance-me.tif', 10, 15)
  transGama('img/enhance-me.tif', 100, 15)

  transGama('img/enhance-me.tif', 10, 1)
  transGama('img/enhance-me.tif', 10, 10)
  transGama('img/enhance-me.tif', 10, 30)

  #Representacao de Bits
  repBits('img/fractured_spine.tif')
  repBits('img/enhance-me.tif')

  #Equalizar Histograma (0 = na mão / 1 = numpy / 2 = opencv / 3 = scipy)
  equalizarHistograma('img/fractured_spine.tif', 0)
  equalizarHistograma('img/enhance-me.tif', 0)

  equalizarHistograma('img/fractured_spine.tif', 1)
  equalizarHistograma('img/enhance-me.tif', 1)

  equalizarHistograma('img/fractured_spine.tif', 2)
  equalizarHistograma('img/enhance-me.tif', 2)

  equalizarHistograma('img/fractured_spine.tif', 3)
  equalizarHistograma('img/enhance-me.tif', 3)


if __name__ == "__main__":
    main()