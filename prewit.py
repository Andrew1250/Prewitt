import cv2
import numpy as np
import matplotlib.pyplot as plt

# membaca citra dalam mode grayscale
img = cv2.imread('gambar 50.JPG')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# mengonversi citra menjadi array NumPy
array = np.asarray(img)

# menampilkan array sebelum dikonvolusi
print("Array sebelum dikonvolusi:")
print(array)

# definisi kernel Prewitt
prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

# konvolusi array dengan kernel Prewitt
out_x = cv2.filter2D(array, -1, prewitt_x)
out_y = cv2.filter2D(array, -1, prewitt_y)

# menampilkan array setelah dikonvolusi
print("Array setelah dikonvolusi dengan kernel Prewitt X:")
print(out_x)
print("Array setelah dikonvolusi dengan kernel Prewitt Y:")
print(out_y)

# menampilkan output konversi dengan array matriks
print("Output konversi dengan array matriks:")
out_matrix = np.sqrt(np.square(out_x) + np.square(out_y)).astype('uint8')

out_matrix_2 = np.hypot(out_x, out_y)
out_matrix_2 = out_matrix_2 / out_matrix_2.max() / 255

out_image = np.array(cv2.addWeighted(out_x, 0.5, out_y, 0.5, 0))
print(out_matrix)

plt.imshow(out_matrix_2)
plt.show()
# menampilkan gambar hasil konversi
# cv2.imshow('Output Konversi', np.hstack([out_matrix, out_image]))
# cv2.waitKey(0)
# cv2.destroyAllWindows()