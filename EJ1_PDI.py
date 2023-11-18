import cv2
import numpy as np


imagen = cv2.imread('monedas.jpg')
imgzeros = np.zeros_like(imagen)

gray_image = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
mascara = np.zeros_like(gray_image)
gris = cv2.GaussianBlur(gray_image, (11, 11), 0)

bordes = cv2.Canny(gris, 0, 100)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
bordes = cv2.dilate(bordes, kernel, iterations=1)

kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
bordes = cv2.erode(bordes, kernel2, iterations=1)

contours, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    if cv2.contourArea(contour) > 5000:
        hull = cv2.convexHull(contour)
        cv2.drawContours(mascara, [hull], -1, (255), thickness=cv2.FILLED)
gray_image = cv2.resize(gray_image, dsize=None, fx=0.3, fy=0.3)
mascara = cv2.resize(mascara, dsize=None, fx=0.3, fy=0.3)
#cv2.imshow("Image with Convex Hulls", mascara)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

img_zeros_blue = np.zeros_like(mascara)
img_zeros_green = np.zeros_like(mascara)
gray_image = cv2.GaussianBlur(gray_image, (19, 19), 0)

retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mascara, 8, cv2.CV_32S)
for i in range(1, retval):
    obj = (labels == i).astype(np.uint8) * 130
    contours, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = cv2.contourArea(contours[0])
    perimetro = cv2.arcLength(contours[0], True)
    FP = area / perimetro**2

    if 0.07 < FP < 0.1:
        if area > 6000:
            img_zeros_blue[labels == i] = 255
        else:
            img_zeros_blue[labels == i] = 128
    else:
        contador_numero = 0
        # Aplica la misma lógica de indexación en la imagen original
        retval, l, stats, c = cv2.connectedComponentsWithStats(np.uint8(labels == i), 8, cv2.CV_32S)
        #print(stats)
        dado = gray_image[stats[1][1]:stats[1][3]+stats[1][1],stats[1][0]:stats[1][3]+stats[1][0]]
        contourst, _ = cv2.findContours(cv2.Canny(dado,0,100), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contourd in contourst:
            hull = cv2.convexHull(contourd)
            area = cv2.contourArea(hull)
            if area < 188 and area > 156:
                contador_numero +=1
                print(area)
        #cv2.imshow('img',cv2.Canny(dado,0,100))
        print('-------------------------------')
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(contador_numero)
        img_zeros_green[labels == i] = 42*contador_numero

# Combina las imágenes de color rojo y verde
img_combined = cv2.merge([img_zeros_blue, img_zeros_green, np.zeros_like(mascara)])

cv2.imshow("Image", img_combined)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(contador_numero)