import numpy as np
from PIL import Image, ImageFilter # pillow
import matplotlib.pyplot as plt
import math
import cv2
from scipy import ndimage
from skimage.io import imread, imshow
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage import color, exposure, transform
from skimage.exposure import equalize_hist

def plot(img, imgNew, msg):
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("Imagem original")
    ax[1].imshow( np.log(1 + np.abs(imgNew)), cmap='gray')
    ax[1].set_title(msg)
    plt.show()
    # salvando - np.log(1 + np.abs(imgNew))

def transFourier(image):
  img = Image.open(image).convert('L')

  npImg = np.array(img)
  imgNew = np.fft.fft2(npImg)
  imgNew = np.fft.fftshift(imgNew)

  plt.imsave('img/transformada_fourier.png', np.log(1 + np.abs(imgNew)), cmap='gray')

  plot(img, imgNew, "Transformada de Fourier")

def inversaFourier(image):
  img = Image.open(image).convert('L')
  img = np.array(img)

  imgshift = np.fft.ifftshift(img)
  imgNew = np.fft.fft2(img)

  imgNew = np.real(imgNew)
  plot(img, imgNew, "Transformada Inversa de Fourier")


def main():

  transFourier('img/car.tif')
  inversaFourier('img/transformada_fourier.png')

if __name__ == "__main__":
    main()