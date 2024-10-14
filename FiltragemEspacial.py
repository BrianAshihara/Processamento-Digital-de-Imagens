import numpy as np
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt
import math
import cv2
import scipy
from scipy import ndimage

def plot(img, imgNew, msg):
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("Imagem original")
    ax[1].imshow(imgNew, cmap='gray')
    ax[1].set_title(msg)
    plt.show()


def convolucao(imagem, kernel, escolha, tipo):
    img = Image.open(imagem)
    npImg = np.array(img)
    altura, largura = npImg.shape
    k_altura, k_largura = kernel.shape

    if escolha == 0:
        escolhaMsg = "Manual"
        altura, largura = npImg.shape
        k_altura, k_largura = kernel.shape

        pad_h = k_altura // 2
        pad_w = k_largura // 2

        imagem_padded = np.pad(npImg, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

        imgNew = np.zeros((altura, largura))

        # Realiza a convolução
        for i in range(altura):
            for j in range(largura):
                regiao = imagem_padded[i:i + k_altura, j:j + k_largura]
                resultado = np.sum(regiao * kernel)
                imgNew[i, j] = resultado

    elif escolha == 1:
      escolhaMsg = "OpenCV"

      imgNew = cv2.filter2D(npImg, -1, kernel)
    elif escolha == 2:
        imgNew = scipy.signal.convolve2d(npImg, kernel, mode='same')
        escolhaMsg = "Scipy"

    msg = escolhaMsg + " - " + tipo
    plot(img, imgNew, msg)


def laplacianoSomado(imagem, laplaciano, escolha):

  img = Image.open(imagem)
  npImg = np.array(img)
  altura, largura = npImg.shape
  k_altura, k_largura = laplaciano.shape
  if escolha == 0:
    escolhaMsg = "Manual"
    altura, largura = npImg.shape
    k_altura, k_largura = laplaciano.shape
    pad_h = k_altura // 2
    pad_w = k_largura // 2

    imagem_padded = np.pad(npImg, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    imgNew = np.zeros((altura, largura))

    # Realiza a convolução
    for i in range(altura):
      for j in range(largura):
          regiao = imagem_padded[i:i + k_altura, j:j + k_largura]
          resultado = np.sum(regiao * laplaciano)
          imgNew[i, j] = resultado
  elif escolha == 1:
    escolhaMsg = "OpenCV"

    imgNew = cv2.filter2D(npImg, -1, laplaciano)
  elif escolha == 2:
      imgNew = scipy.signal.convolve2d(npImg, laplaciano, mode='same')
      escolhaMsg = "Scipy"

  msg = escolhaMsg + " - " + "Laplaciano somado a imagem original"

  imgNew = imgNew + npImg

  plot(img, imgNew, msg)


def main():


  mean = np.array((
    [0.1111, 0.1111, 0.1111],
    [0.1111, 0.1111, 0.1111],
    [0.1111, 0.1111, 0.1111]), dtype="float")

  gauss = np.array((
    [0.0625, 0.125, 0.0625],
    [0.1250, 0.250, 0.1250],
    [0.0625, 0.125, 0.0625]), dtype="float")

  laplacian = np.array((
    [0,  1, 0],
    [1, -4, 1],
    [0,  1, 0]), dtype="int")

  sobelX = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="int")

  sobelY = np.array((
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]), dtype="int")

  gradiente = sobelX + sobelY


  #CONVOLUCAO UTILIZANDO MASCARAS (0 - MANUAL / 1 - OPENCV / 2 - SCIPY)
  convolucao('img/lena.tif', mean, 0, "Media")
  convolucao('img/lena.tif', gauss, 0, "Gaussiano")
  convolucao('img/lena.tif', laplacian, 0, "Laplaciano")
  convolucao('img/lena.tif', sobelX, 0, "SobelX")
  convolucao('img/lena.tif', sobelY, 0, "SobelY")
  convolucao('img/lena.tif', gradiente, 0, "Gradiente")
  laplacianoSomado('img/lena.tif', laplacian, 0)

  convolucao('img/lena.tif', mean, 1, "Media")
  convolucao('img/lena.tif', gauss, 1, "Gaussiano")
  convolucao('img/lena.tif', laplacian, 1, "Laplaciano")
  convolucao('img/lena.tif', sobelX, 1, "SobelX")
  convolucao('img/lena.tif', sobelY, 1, "SobelY")
  convolucao('img/lena.tif', gradiente, 1, "Gradiente")
  laplacianoSomado('img/lena.tif', laplacian, 1)

  convolucao('img/lena.tif', mean, 2, "Media")
  convolucao('img/lena.tif', gauss, 2, "Gaussiano")
  convolucao('img/lena.tif', laplacian, 2, "Laplaciano")
  convolucao('img/lena.tif', sobelX, 2, "SobelX")
  convolucao('img/lena.tif', sobelY, 2, "SobelY")
  convolucao('img/lena.tif', gradiente, 2, "Gradiente")
  laplacianoSomado('img/lena.tif', laplacian, 2)

  convolucao('img/cameraman.tif', mean, 0, "Media")
  convolucao('img/cameraman.tif', gauss, 0, "Gaussiano")
  convolucao('img/cameraman.tif', laplacian, 0, "Laplaciano")
  convolucao('img/cameraman.tif', sobelX, 0, "SobelX")
  convolucao('img/cameraman.tif', sobelY, 0, "SobelY")
  convolucao('img/cameraman.tif', gradiente, 0, "Gradiente")
  laplacianoSomado('img/cameraman.tif', laplacian, 0)

  convolucao('img/cameraman.tif', mean, 1, "Media")
  convolucao('img/cameraman.tif', gauss, 1, "Gaussiano")
  convolucao('img/cameraman.tif', laplacian, 1, "Laplaciano")
  convolucao('img/cameraman.tif', sobelX, 1, "SobelX")
  convolucao('img/cameraman.tif', sobelY, 1, "SobelY")
  convolucao('img/cameraman.tif', gradiente, 1, "Gradiente")
  laplacianoSomado('img/cameraman.tif', laplacian, 1)

  convolucao('img/cameraman.tif', mean, 2, "Media")
  convolucao('img/cameraman.tif', gauss, 2, "Gaussiano")
  convolucao('img/cameraman.tif', laplacian, 2, "Laplaciano")
  convolucao('img/cameraman.tif', sobelX, 2, "SobelX")
  convolucao('img/cameraman.tif', sobelY, 2, "SobelY")
  convolucao('img/cameraman.tif', gradiente, 2, "Gradiente")
  laplacianoSomado('img/cameraman.tif', laplacian, 2)

  convolucao('img/biel.png', mean, 0, "Media")
  convolucao('img/biel.png', gauss, 0, "Gaussiano")
  convolucao('img/biel.png', laplacian, 0, "Laplaciano")
  convolucao('img/biel.png', sobelX, 0, "SobelX")
  convolucao('img/biel.png', sobelY, 0, "SobelY")
  convolucao('img/biel.png', gradiente, 0, "Gradiente")
  laplacianoSomado('img/biel.png', laplacian,0)

  convolucao('img/biel.png', mean, 1, "Media")
  convolucao('img/biel.png', gauss, 1, "Gaussiano")
  convolucao('img/biel.png', laplacian, 1, "Laplaciano")
  convolucao('img/biel.png', sobelX, 1, "SobelX")
  convolucao('img/biel.png', sobelY, 1, "SobelY")
  convolucao('img/biel.png', gradiente, 1, "Gradiente")
  laplacianoSomado('img/biel.png', laplacian,1)

  convolucao('img/biel.png', mean, 2, "Media")
  convolucao('img/biel.png', gauss, 2, "Gaussiano")
  convolucao('img/biel.png', laplacian, 2, "Laplaciano")
  convolucao('img/biel.png', sobelX, 2, "SobelX")
  convolucao('img/biel.png', sobelY, 2, "SobelY")
  convolucao('img/biel.png', gradiente, 2, "Gradiente")
  laplacianoSomado('img/biel.png', laplacian,2)


if __name__ == "__main__":
    main()