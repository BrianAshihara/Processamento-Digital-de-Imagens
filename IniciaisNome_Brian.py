import numpy as np
import matplotlib.pyplot as plt

colunas = 30
linhas = 15
image_matrix = np.zeros([linhas, colunas])
print(image_matrix.shape)

#B

for i in range(1,14):
    image_matrix[i,3] = 255

for i in range(4,8):
  image_matrix[1,i] = 255

image_matrix[2,8] = 255
image_matrix[3,8] = 255
image_matrix[4,8] = 255
image_matrix[5,8] = 255
image_matrix[6,8] = 255

for i in range(4,8):
  image_matrix[7,i] = 255

image_matrix[8,8] = 255
image_matrix[9,8] = 255
image_matrix[10,8] = 255
image_matrix[11,8] = 255
image_matrix[12,8] = 255

for i in range(4,8):
  image_matrix[13,i] = 255


#C

for i in range(2,13):
  image_matrix[i,11] = 255


for i in range(12,17):
  image_matrix[1,i] = 255


for i in range(12,17):
  image_matrix[13,i] = 255


image_matrix[2,17] = 255

image_matrix[12,17] = 255

#A

for i in range(2,14):
  image_matrix[i,20] = 255

for i in range(2,14):
  image_matrix[i,26] = 255


for i in range(21,26):
  image_matrix[1,i] = 255

for i in range(21,26):
  image_matrix[8,i] = 255

plt.imshow(image_matrix, cmap='gray')
plt.show()