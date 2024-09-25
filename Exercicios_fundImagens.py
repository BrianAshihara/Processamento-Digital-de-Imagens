# Exemplo de leitura e plot de imagens
# Bibliotecas Numpy, Pillow, Matplotlib

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

def Negativo(escolha):
    img = Image.open(escolha)
    print(img.format)
    print(img.size)
    print(img.mode)
    npImg = np.array(img)
    npImg = 255 - npImg
    imgNew = Image.fromarray(npImg)
    plot(img, imgNew, "Imagem Negativa")

def centroPreto(escolha):
    img = Image.open(escolha)
    print(img.format)
    print(img.size)
    print(img.mode)
    npImg = np.array(img)
    altura, largura = img.size
    centro_altura = altura // 2
    centro_largura = largura // 2

    npImg[centro_largura-50:centro_altura+50, centro_largura-50:centro_altura+50] = 0
    imgNew = Image.fromarray(npImg)
    plot(img, imgNew, "Imagem com o Centro Preto")

def quadradosBrancos(escolha):
    img = Image.open(escolha)
    print(img.format)
    print(img.size)
    print(img.mode)
    npImg = np.array(img)

    altura, largura = img.size

    npImg[0:100,0:100] = 255
    npImg[largura-100:largura,0:100] = 255
    npImg[0:100,largura-100:largura] = 255
    npImg[largura-100:largura,altura-100:altura] = 255
    imgNew = Image.fromarray(npImg)
    plot(img, imgNew, "Imagem com Quadrados Brancos")

def diminuirBrilho(escolha):
    img = Image.open(escolha)
    print(img.format)
    print(img.size)
    print(img.mode)
    npImg = np.array(img)

    npImg = npImg * 0.5
    imgNew = Image.fromarray(npImg)
    plot(img, imgNew, "Imagem com o Brilho Reduzido")

def filtroMedia(escolha, modo):
    img = Image.open(escolha)
    print(img.format)
    print(img.size)
    print(img.mode)
    npImg = np.array(img)
    if(modo == 1):
      newImg = cv2.imread(escolha)
      newImg = cv2.blur(newImg, (3, 3))
      msgModo = " - opencv"
    elif(modo == 2):
        tamanho = 3
        altura, largura = npImg.shape[:2]
        newImg = np.zeros_like(npImg)


        for i in range(altura):
            for j in range(largura):
                janela = npImg[i:i+tamanho, j:j+tamanho]
                newImg[i, j] = np.mean(janela, axis=(0, 1))
        msgModo = " - numpy"
    elif(modo == 3):
      newImg = img.filter(ImageFilter.BLUR)
      msgModo = " - pillow"
    elif(modo == 4):
      newImg = ndimage.uniform_filter(npImg, size=3)
      msgModo = " - scipy"

    msg = "Imagem com Filtro Media"
    msgFinal = msg + msgModo
    plot(img, newImg, msgFinal)


def filtroMediana(escolha, modo):
    img = Image.open(escolha)
    print(img.format)
    print(img.size)
    print(img.mode)
    npImg = np.array(img)
    if(modo == 1):
      newImg = cv2.imread(escolha)
      newImg = cv2.medianBlur(newImg, 3)
      msgModo = " - opencv"
    elif(modo == 2):
      tamanho = 3
      altura, largura = npImg.shape[:2]
      newImg = np.zeros_like(npImg)


      for i in range(altura):
        for j in range(largura):
          janela = npImg[i:i+tamanho, j:j+tamanho]
          newImg[i, j] = np.median(janela, axis=(0, 1))

      msgModo = " - numpy"
    elif(modo == 3):
      newImg = img.filter(ImageFilter.MedianFilter(size=3))
      msgModo = " - pillow"
    elif(modo == 4):
      newImg = ndimage.median_filter(npImg, size=3)
      msgModo = " - scipy"

    msg  = "Imagem com Filtro Mediana"
    msgFinal = msg + msgModo
    plot(img, newImg, msgFinal)


def escala(escolha, zoomOrShrink, modo):
    img = Image.open(escolha)
    print(img.format)
    print(img.size)
    print(img.mode)
    npImg = np.array(img)

    if(modo == 1):
      newImg = cv2.imread(escolha)
      if(zoomOrShrink == 0):
        newImg = cv2.resize(newImg, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_LINEAR)
        msg = "Imagem com Zoom - opencv"
      elif(zoomOrShrink == 1):
        newImg = cv2.resize(newImg, None, fx=0.666, fy=0.666, interpolation=cv2.INTER_LINEAR)
        msg = "Imagem com Reducao - opencv"
    elif(modo == 2):
      if(zoomOrShrink == 0):
          fx, fy = 2.5, 2.5
          msg = "Imagem com Zoom - numpy"
      elif(zoomOrShrink == 1):
            fx, fy = 0.666, 0.666
            msg = "Imagem com Reducao - numpy"

      altura, largura = npImg.shape[:2]
      nova_altura = int(altura * fx)
      nova_largura = int(largura * fy)

      newImg = np.zeros((nova_altura, nova_largura), dtype=npImg.dtype)

      for i in range(nova_altura):
        for j in range(nova_largura):

          orig_i = int(i / fy)
          orig_j = int(j / fx)

          orig_i = min(orig_i, altura - 1)
          orig_j = min(orig_j, largura - 1)
          newImg[i, j] = npImg[orig_i, orig_j]

    elif(modo == 3):
      if(zoomOrShrink == 0):
        newImg = img.resize((int(img.width * 2.5), int(img.height * 2.5)))
        msg = "Imagem com Zoom - pillow"
      elif(zoomOrShrink == 1):
        newImg = img.resize((int(img.width * 0.67), int(img.height * 0.67)))
        msg = "Imagem com Reducao - pillow"
    elif(modo == 4):
      if(zoomOrShrink == 0):
          newImg = ndimage.zoom(npImg, (2.5, 2.5), order=2)
          msg = "Imagem com Zoom - scipy"
      elif(zoomOrShrink == 1):
          newImg = ndimage.zoom(npImg, (0.666, 0.666), order=2)
          msg = "Imagem com Reducao - scipy"

    plot(img, newImg, msg)



def rotacao(escolha, grau, modo):
    img = Image.open(escolha)
    print(img.format)
    print(img.size)
    print(img.mode)
    npImg = np.array(img)

    if(modo == 1):
      newImg = cv2.imread(escolha)
      image_center = tuple(np.array(npImg.shape[1::-1]) / 2)
      rot_mat = cv2.getRotationMatrix2D(image_center, grau, 1.0)
      newImg = cv2.warpAffine(npImg, rot_mat, npImg.shape[1::-1], flags=cv2.INTER_LINEAR)
      msgModo = " - opencv"
    elif(modo == 2):

      angulo_rad = math.radians(grau)
      altura, largura = npImg.shape[:2]

      nova_altura = int(abs(altura * math.cos(angulo_rad)) + abs(largura * math.sin(angulo_rad)))
      nova_largura = int(abs(largura * math.cos(angulo_rad)) + abs(altura * math.sin(angulo_rad)))


      newImg = np.zeros((nova_altura, nova_largura, 3), dtype=npImg.dtype)

      centro_nova = (nova_largura // 2, nova_altura // 2)
      centro_original = (largura // 2, altura // 2)
      msgModo = " - numpy"

      for i in range(altura):
          for j in range(largura):

              x = j - centro_original[0]
              y = i - centro_original[1]
              x_novo = int(x * math.cos(angulo_rad) - y * math.sin(angulo_rad)) + centro_nova[0]
              y_novo = int(x * math.sin(angulo_rad) + y * math.cos(angulo_rad)) + centro_nova[1]

              if 0 <= x_novo < nova_largura and 0 <= y_novo < nova_altura:
                  newImg[y_novo, x_novo] = npImg[i, j]
    elif(modo == 3):
      newImg = img.rotate(grau)
      msgModo = " - pillow"
    elif(modo == 4):
      newImg = ndimage.rotate(npImg, grau, reshape=True)
      msgModo = " - scipy"


    msg = "Imagem com Rotacao de " + str(grau) + " graus"
    msgFinal = msg + msgModo

    plot(img, newImg, msgFinal)


def translacao(escolha, x, y, modo):
    img = Image.open(escolha)
    print(img.format)
    print(img.size)
    print(img.mode)
    npImg = np.array(img)


    msg = "Imagem com Translacao nas coordenadas X = " + str(x) + " e Y = " + str(y)

    if modo == 1:
      newImg = cv2.imread(escolha)
      newImg = cv2.warpAffine(newImg, np.float32([[1, 0, x], [0, 1, y]]), (newImg.shape[1], newImg.shape[0]))
      msgModo = " - opencv"

    elif modo == 2:
      altura, largura = npImg.shape[:2]
      newImg = np.zeros((altura, largura, 3), dtype=npImg.dtype)
      for i in range(altura):
        for j in range(largura):
          if 0 <= i + y < altura and 0 <= j + x < largura:
            newImg[i + y, j + x] = npImg[i, j]

      msgModo = " - numpy"
    elif modo == 3:
      newImg = img.transform(img.size, Image.AFFINE, (1, 0, x, 0, 1, y))
      msgModo = " - pillow"

    elif modo == 4:
      newImg = ndimage.shift(npImg, (x, y))
      msgModo = " - scipy"


    msgFinal = msg + msgModo
    plot(img, newImg, msgFinal)



def main():

    #1 - OPERACOES PONTO A PONTO
    #Negativo
    Negativo('img/lena_gray_512.tif')
    Negativo('img/cameraman.tif')
    Negativo('img/house.tif')

    #Centro Preto
    centroPreto('img/lena_gray_512.tif')
    centroPreto('img/cameraman.tif')
    centroPreto('img/house.tif')

    # 4 Quadrados Brancos nos Cantos
    quadradosBrancos('img/lena_gray_512.tif')
    quadradosBrancos('img/cameraman.tif')
    quadradosBrancos('img/house.tif')

    #Diminuir Brilho
    diminuirBrilho('img/lena_gray_512.tif')
    diminuirBrilho('img/cameraman.tif')
    diminuirBrilho('img/house.tif')

    #2 - OPERACOES VIZINHANCA
    #Filtro Media (1 = opencv / 2 = numpy / 3 = pillow / 4 = scipy)
    filtroMedia('img/lena_gray_512.tif', 1)
    filtroMedia('img/cameraman.tif', 1)
    filtroMedia('img/house.tif', 1)

    filtroMedia('img/lena_gray_512.tif', 2)
    filtroMedia('img/cameraman.tif', 2)
    filtroMedia('img/house.tif', 2)

    filtroMedia('img/lena_gray_512.tif', 3)
    filtroMedia('img/cameraman.tif', 3)
    filtroMedia('img/house.tif', 3)

    filtroMedia('img/lena_gray_512.tif', 4)
    filtroMedia('img/cameraman.tif', 4)
    filtroMedia('img/house.tif', 4)

    #Filtro Mediana (1 = opencv / 2 = numpy / 3 = pillow / 4 = scipy)
    filtroMediana('img/lena_gray_512.tif', 1)
    filtroMediana('img/cameraman.tif', 1)
    filtroMediana('img/house.tif', 1)

    filtroMediana('img/lena_gray_512.tif', 2)
    filtroMediana('img/cameraman.tif', 2)
    filtroMediana('img/house.tif', 2)

    filtroMediana('img/lena_gray_512.tif', 3)
    filtroMediana('img/cameraman.tif', 3)
    filtroMediana('img/house.tif', 3)

    filtroMediana('img/lena_gray_512.tif', 4)
    filtroMediana('img/cameraman.tif', 4)
    filtroMediana('img/house.tif', 4)


    #Transformacao Geometrica
    #Escala (escolha a imagem e depois forneça se quer aumentar ou diminuir a imagem: 0 = aumentar / 1 = diminuir) | (1 = opencv / 2 = numpy / 3 = pillow / 4 = scipy)
    escala('img/lena_gray_512.tif', 1, 1)
    escala('img/cameraman.tif', 1, 1)
    escala('img/house.tif', 1, 1)
    
    escala('img/lena_gray_512.tif', 1, 2)
    escala('img/cameraman.tif', 1, 2)
    escala('img/house.tif', 1, 2)


    escala('img/lena_gray_512.tif', 1, 3)
    escala('img/cameraman.tif', 1, 3)
    escala('img/house.tif', 1, 3)


    escala('img/lena_gray_512.tif', 1, 4)
    escala('img/cameraman.tif', 1, 4)
    escala('img/house.tif', 1, 4)


    escala('img/lena_gray_512.tif', 0, 1)
    escala('img/cameraman.tif', 0, 1)
    escala('img/house.tif', 0, 1)


    escala('img/lena_gray_512.tif', 0, 2)
    escala('img/cameraman.tif', 0, 2)
    escala('img/house.tif', 0, 2)


    escala('img/lena_gray_512.tif', 0, 3)
    escala('img/cameraman.tif', 0, 3)
    escala('img/house.tif', 0, 3)


    escala('img/lena_gray_512.tif', 0, 4)
    escala('img/cameraman.tif', 0, 4)
    escala('img/house.tif', 0, 4)



    #Rotacao (graus) (1 = opencv / 2 = numpy / 3 = pillow / 4 = scipy)
    #Rotacao 45 graus
    rotacao('img/lena_gray_512.tif', 45, 1)
    rotacao('img/cameraman.tif', 45, 1)
    rotacao('img/house.tif', 45, 1)

    rotacao('img/lena_gray_512.tif', 45, 2)
    rotacao('img/cameraman.tif', 45, 2)
    rotacao('img/house.tif', 45, 2)

    rotacao('img/lena_gray_512.tif', 45, 3)
    rotacao('img/cameraman.tif', 45, 3)
    rotacao('img/house.tif', 45, 3)

    rotacao('img/lena_gray_512.tif', 45, 4)
    rotacao('img/cameraman.tif', 45, 4)
    rotacao('img/house.tif', 45, 4)
    
    rotacao('img/lena_gray_512.tif', 90, 1)
    rotacao('img/cameraman.tif', 90, 1)
    rotacao('img/house.tif', 90, 1)

    rotacao('img/lena_gray_512.tif', 90, 2)
    rotacao('img/cameraman.tif', 90, 2)
    rotacao('img/house.tif', 90, 2)

    rotacao('img/lena_gray_512.tif', 90, 3)
    rotacao('img/cameraman.tif', 90, 3)
    rotacao('img/house.tif', 90, 3)

    rotacao('img/lena_gray_512.tif', 90, 4)
    rotacao('img/cameraman.tif', 90, 4)
    rotacao('img/house.tif', 90, 4)

    rotacao('img/lena_gray_512.tif', 100, 1)
    rotacao('img/cameraman.tif', 100, 1)
    rotacao('img/house.tif', 100, 1)

    rotacao('img/lena_gray_512.tif', 100, 2)
    rotacao('img/cameraman.tif', 100, 2)
    rotacao('img/house.tif', 100, 2)

    rotacao('img/lena_gray_512.tif', 100, 3)
    rotacao('img/cameraman.tif', 100, 3)
    rotacao('img/house.tif', 100, 3)

    rotacao('img/lena_gray_512.tif', 100, 4)
    rotacao('img/cameraman.tif', 100, 4)
    rotacao('img/house.tif', 100, 4)

    #Translacao
    #Translacao com X e Y que quiser  (1 = opencv / 2 = numpy / 3 = pillow / 4 = scipy)
    translacao('img/lena_gray_512.tif', 100, 100, 1)
    translacao('img/cameraman.tif', 100, 100, 1)
    translacao('img/house.tif', 100, 100, 1)

    translacao('img/lena_gray_512.tif', 100, 100, 2)
    translacao('img/cameraman.tif', 100, 100, 2)
    translacao('img/house.tif', 100, 100, 2)

    translacao('img/lena_gray_512.tif', 100, 100, 3)
    translacao('img/cameraman.tif', 100, 100, 3)
    translacao('img/house.tif', 100, 100, 3)

    translacao('img/lena_gray_512.tif', 100, 100, 4)
    translacao('img/cameraman.tif', 100, 100, 4)
    translacao('img/house.tif', 100, 100, 4)

    #Translacao em 35 pixels no X e 45 pixels no Y  (1 = opencv / 2 = numpy / 3 = pillow / 4 = scipy)
    translacao('img/lena_gray_512.tif', 35, 45,1)
    translacao('img/cameraman.tif', 35, 45,1)
    translacao('img/house.tif', 35, 45,1)

    translacao('img/lena_gray_512.tif', 35, 45,2)
    translacao('img/cameraman.tif', 35, 45,2)
    translacao('img/house.tif', 35, 45,2)

    translacao('img/lena_gray_512.tif', 35, 45,3)
    translacao('img/cameraman.tif', 35, 45,3)
    translacao('img/house.tif', 35, 45,3)

    translacao('img/lena_gray_512.tif', 35, 45,4)
    translacao('img/cameraman.tif', 35, 45,4)
    translacao('img/house.tif', 35, 45,4)

if __name__ == "__main__":
    main()