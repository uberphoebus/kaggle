import matplotlib.image as img
import matplotlib.pyplot as plt

filename = './logo.png'

ndarray = img.imread(filename)
plt.imshow(ndarray)
plt.show()