import cv2
import numpy as np

#Función para limpiar ruido segun cercanía a componentes conectadas
def limpiar_ruido(imagen_binaria, distancia_umbral):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imagen_binaria, connectivity=4)

    for i in range(1, num_labels):
        #Calculo distancia de un centroide a todos los centroides y busco el mínimo
        centroide_actual = centroids[i]
        distancias = np.linalg.norm(centroide_actual - centroids, axis=1)
        distancias[i] = np.inf
        mindis = min(distancias)
        
        #Si hay demasiada distancia a todas las componentes se borra la componente conectada
        if mindis > distancia_umbral:
            imagen_binaria[labels == i] = 0

    return imagen_binaria

for i in range(1,13):

  if i < 10:
    ruta = 'img0' + str(i) + '.png'
  else:
    ruta = 'img' + str(i) + '.png'
    
  imagen_original = cv2.imread('Patentes/' + ruta)

  #Convertir a escala de grises
  imagen_gris = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2GRAY)

  #Umbralado
  imagen_umbralizada = cv2.adaptiveThreshold(imagen_gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, -10) #9 y -10 anda en 10/12

  #Componentes conectados
  num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imagen_umbralizada, connectivity=4)

  #Filtrar por área y relacion de aspecto
  area_minima = 25
  area_maxima = 90
  aspect_ratio_min = 0.4
  aspect_ratio_max = 0.7

  area_filtrada = np.zeros_like(imagen_umbralizada)

  for i in range(1, num_labels):

      aspect_ratio = stats[i, cv2.CC_STAT_WIDTH] / stats[i, cv2.CC_STAT_HEIGHT]

      if (
          area_minima < stats[i][4] and stats[i][4] < area_maxima
          and aspect_ratio_min < aspect_ratio and aspect_ratio < aspect_ratio_max
      ):
          area_filtrada[labels == i] = imagen_umbralizada[labels == i]


  # 7. Eliminar ruido usando la distancia entre componentes conectados
  distancia_umbral = 15
  area_limpiada = limpiar_ruido(area_filtrada.copy(), distancia_umbral)

  se = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
  area_limpiada_open = cv2.morphologyEx(area_limpiada, cv2.MORPH_OPEN, se)
  imagen_final = cv2.morphologyEx(area_limpiada_open, cv2.MORPH_CLOSE, se)
  w, h = imagen_final.shape

  # Mostrar resultados recortando cerca del centro de la imagen
  cv2.imshow("Imagen original", imagen_original)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  cv2.imshow("Patente filtrada", imagen_final[80:w-50,170:h-50])
  cv2.waitKey(0)
  cv2.destroyAllWindows()


  # Componentes conectadas de la imagen con solo la patente
  num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imagen_final[80:w-50,170:h-50], connectivity=4)

  # Mostrar cada caracter de cada patente
  for label in range(1, num_labels):  # Comienza desde 1 para excluir el fondo
    component_mask = (labels == label).astype(np.uint8) * 255
    cv2.imshow("Caracter filtrado", component_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
