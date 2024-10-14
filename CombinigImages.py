import numpy as np
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt
import math
import cv2
import scipy
from scipy import ndimage
# from google.colab.patches import cv2_imshow

def plot(img, imgNew, msg1, msg2):
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title(msg1)
    ax[1].imshow(imgNew, cmap='gray')
    ax[1].set_title(msg2)
    plt.show()


def encontraDefeitos(original, defeituosa):

  imgOriginal = Image.open(original)
  imgDefeituosa = Image.open(defeituosa)


  npOriginal = np.array(imgOriginal)
  npDefeituosa = np.array(imgDefeituosa)

  imgNew = npOriginal - npDefeituosa

  imgNewAbs = np.abs(imgNew)

  plot(imgNew ,imgNewAbs, "Imagem da Diferença","Imagem Absoluta da Diferença")

  return imgNewAbs


def transformaPeB(img):

  imgNew = img.copy()
  ahah = 10

  for x in range(0, imgNew.shape[0]):
    for y in range(0, imgNew.shape[1]):
      if imgNew[x,y] > ahah:
        imgNew[x,y] = 255
      else:
        imgNew[x,y] = 0

  return imgNew


def subtraiVideo(vid, background):

  vid = cv2.VideoCapture(vid)
  frames = []

  frame_largura = int(vid.get(3))
  frame_altura = int(vid.get(4))
  frame_rate = 30
  size = (frame_largura, frame_altura)

  fourcc = cv2.VideoWriter_fourcc(*'MJPG')
  result = cv2.VideoWriter('resultadoVideo.avi', fourcc, frame_rate, size, 0)

  npBack = np.array(Image.open(background))

  while True:
    ret, frame = vid.read()
    if ret:
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

      gray = cv2.absdiff(frame, npBack)

      _, gray = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

      result.write(gray)

      cv2.imshow('Frame', gray)
      cv2.waitKey(1)

    else:
      break
  vid.release()
  cv2.destroyAllWindows()
  vid.release()
  result.release()


def main():

  res = encontraDefeitos('img/pcbCroppedTranslated.png','img/pcbCroppedTranslatedDef.png' )
  transformaPeB(res)

  print("\n\n")

  subtraiVideo('img/video.avi', 'img/background.png')

if __name__ == "__main__":
    main()