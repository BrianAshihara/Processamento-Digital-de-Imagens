# Exemplo de leitura e plot de imagens
# Bibliotecas Numpy, Pillow, Matplotlib

import numpy as np
from PIL import Image # pillow
import matplotlib.pyplot as plt
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

def filtroMedia(escolha):
    img = Image.open(escolha)
    print(img.format)
    print(img.size)
    print(img.mode)
    npImg = np.array(img)
    newImg = cv2.imread(escolha)


    newImg = cv2.blur(newImg, (3, 3))

    plot(img, newImg, "Imagem com Filtro Media")


def filtroMediana(escolha):
    img = Image.open(escolha)
    print(img.format)
    print(img.size)
    print(img.mode)
    npImg = np.array(img)

    newImg = cv2.imread(escolha)
    newImg = cv2.medianBlur(newImg, 3)

    plot(img, newImg, "Imagem com Filtro Mediana")

def escala(escolha, zoomOrShrink):
    img = Image.open(escolha)
    print(img.format)
    print(img.size)
    print(img.mode)
    npImg = np.array(img)

    if(zoomOrShrink == 0):
        newImg = ndimage.zoom(npImg, (2.5, 2.5), order=2)
        msg = "Imagem com Zoom"
    elif(zoomOrShrink == 1):
        newImg = ndimage.zoom(npImg, (0.666, 0.666), order=2)
        msg = "Imagem com Reducao"

    plot(img, newImg, msg)


def rotacao(escolha, grau):
    img = Image.open(escolha)
    print(img.format)
    print(img.size)
    print(img.mode)
    npImg = np.array(img)

    newImg = ndimage.rotate(npImg, grau, reshape=True)
    msg = "Imagem com Rotacao de " + str(grau) + " graus"

    plot(img, newImg, msg)


def translacao(escolha, x, y):
    img = Image.open(escolha)
    print(img.format)
    print(img.size)
    print(img.mode)
    npImg = np.array(img)

    newImg = ndimage.shift(npImg, (x, y))
    msg = "Imagem com Translacao nas coordenadas X = " + str(x) + " e Y = " + str(y)

    plot(img, newImg, msg)



def main():

    #OPERACOES PONTO A PONTO
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


    #OPERACOES VIZINHANCA
    #Filtro Media
    filtroMedia('img/lena_gray_512.tif')
    filtroMedia('img/cameraman.tif')
    filtroMedia('img/house.tif')

    #Filtro Mediana
    filtroMediana('img/lena_gray_512.tif')
    filtroMediana('img/cameraman.tif')
    filtroMediana('img/house.tif')


    #Transformacao Geometrica
    #Escala (escolha a imagem e depois forneça se quer aumentar ou diminuir a imagem: 0 = aumentar / 1 = diminuir)
    #Zoom
    escala('img/lena_gray_512.tif', 0)
    escala('img/cameraman.tif', 0)
    escala('img/house.tif', 0)

    #Shrink
    escala('img/lena_gray_512.tif', 1)
    escala('img/cameraman.tif', 1)
    escala('img/house.tif', 1)


    #Rotacao
    #Rotacao 45 graus
    rotacao('img/lena_gray_512.tif', 45)
    rotacao('img/cameraman.tif', 45)
    rotacao('img/house.tif', 45)
    #Rotacao 90 graus
    rotacao('img/lena_gray_512.tif', 90)
    rotacao('img/cameraman.tif', 90)
    rotacao('img/house.tif', 90)
    #Rotacao 100 graus
    rotacao('img/lena_gray_512.tif', 100)
    rotacao('img/cameraman.tif', 100)
    rotacao('img/house.tif', 100)


    #Translacao
    #Translacao com X e Y que quiser
    translacao('img/lena_gray_512.tif', 100, 100)
    translacao('img/cameraman.tif', 100, 100)
    translacao('img/house.tif', 100, 100)

    #Translacao em 35 pixels no X e 45 pixels no Y

    translacao('img/lena_gray_512.tif', 35, 45)
    translacao('img/cameraman.tif', 35, 45)
    translacao('img/house.tif', 35, 45)

if __name__ == "__main__":
    main()