# -*- coding: utf-8 -*-
"""
Visión por Computador (CSI): Proyecto Final
20 de diciembre de 2019

@author: Alberto Estepa Fernández
@author: Carlos Santiago Sánchez Muñoz
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import floor, exp
from random import sample

###########################################
###   LECTURA E IMPRESIÓN DE IMÁGENES   ###
###########################################

""" Lee una imagen ya sea en grises o en color. Devuelve la imagen.
- file_name: archivo de la imagen.
- flag_color (op): modo en el que se va a leer la imagen -> grises o color. Por defecto será en color.
"""
def leer_imagen(file_name, flag_color = 1):
    if flag_color == 0:
        print("Leyendo '" + file_name + "' en gris")
    elif flag_color==1:
        print("Leyendo '" + file_name + "' en color")
    else:
        print("flag_color debe ser 0 o 1")

    img = cv2.imread(file_name, flag_color)

    if flag_color==1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)
    return img

""" Normaliza una matriz.
- image: matriz a normalizar.
- image_title (op): título de la imagen. Por defecto ' '.
"""
def normaliza(image, image_title = " "):
    # En caso de que los máximos sean 255 o las mínimos 0 no iteramos en los bucles
    if len(image.shape) == 2:
        max = np.amax(image)
        min = np.amin(image)
        if max>255 or min<0:
            print("Normalizando imagen '" + image_title + "'")
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    image[i][j] = (image[i][j]-min)/(max-min) * 255
    elif len(image.shape) == 3:
        max = [np.amax(image[:,:,0]), np.amax(image[:,:,1]), np.amax(image[:,:,2])]
        min = [np.amin(image[:,:,0]), np.amin(image[:,:,1]), np.amin(image[:,:,2])]

        if max[0]>255 or max[1]>255 or max[2]>255 or min[0]<0 or min[1]<0 or min[2]<0:
            print("Normalizando imagen '" + image_title + "'")
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    for k in range(image.shape[2]):
                        image[i][j][k] = (image[i][j][k]-min[k])/(max[k]-min[k]) * 255

    return image

""" Imprime una imagen a través de una matriz.
- image: imagen a imprimir.
- flag_color (op): bandera para indicar si la imagen es en B/N o color. Por defecto color.
- image_title(op): título de la imagen. Por defecto 'Imagen'
- window_title (op): título de la ventana. Por defecto 'Ejercicio'
"""
def pintaI(image, flag_color=1, image_title = "Imagen", window_title = "Ejercicio"):
    image = normaliza(image, image_title)               # Normalizamos la matriz
    image = image.astype(np.uint8)
    plt.figure(0).canvas.set_window_title(window_title) # Ponemos nombre a la ventana
    if flag_color == 0:
        plt.imshow(image, cmap = "gray")
    else:
        plt.imshow(image)
    plt.title(image_title)              # Ponemos nombre a la imagen
    plt.xticks([])                      # Se le pasa una lista de posiciones en las que se deben colocar los
    plt.yticks([])                      # ticks, si pasamos una lista vacía deshabilitamos los xticks e yticks
    plt.show()
    image = image.astype(np.float64)    # Devolvemos su formato

###############################
###   FILTROS Y PIRÁMIDES   ###
###############################

""" Sobremuestrea la imagen dada al tamaño de la matriz dado y aplica una máscara.
- img: imagen a tratar.
- tam: tamaño.
"""
def convolution(img, tam):
    im_expanded = np.zeros(tam.shape, img.dtype)    # tamaño objetivo
    im_expanded[::2, ::2, ...] = img                # rellenamos en las impares
    mask = 1.0 / 20 * np.array([1, 5, 8, 5, 1])     # máscara
    res = im_expanded                               # hago copia

    for c in range(2):                  # Itero filas, luego columnas
        cp = res.copy()                 # copia auxiliar
        for i in range(res.shape[0]):   # Recorro las filas
            # Aplico el filtro en las filas de la imagen, almacenandolo en la imagen auxiliar
            cp[i,:] = cv2.filter2D(src=res[i,:], dst=cp[i,:], ddepth=cv2.CV_32F, kernel=mask, borderType=cv2.BORDER_DEFAULT)
        res = cv2.transpose(cp)         # Transpongo y ahora las columnas son filas

    return res

""" Calcula la pirámide gaussiana de una imagen.
- img: imagen de la que calcular la pirámide.
- levels (op): niveles de la pirámide. Por defecto 8.
"""
def gaussianPyramid(img, levels=8):
    pyramid = [img] # imagen original
    actual = img
    for i in range(levels):
        actual = np.uint8(actual)
        # Convolucion + subsampling
        actual = cv2.pyrDown(actual)
        actual = np.uint32(actual)
        pyramid.append(actual)
    return pyramid

""" Calcula la pirámide laplaciana de una imagen.
- img: imagen de la que calcular la pirámide.
- levels (op): niveles de la pirámide. Por defecto 8.
"""
def laplacianPyramid(img, levels=8):
    gaussiana = gaussianPyramid(img, levels)
    pyramid = []
    for i in range(len(gaussiana) - 1):
        actual = gaussiana[i]
        siguiente = gaussiana[i+1]
        # Upsampling
        siguiente = np.float32(siguiente)
        aumento = convolution(siguiente, actual)
        siguiente = np.uint32(siguiente)

        # Hacemos la diferencia
        pyramid.append(actual - aumento)

    # Ultimo nivel de la gaussiana que es el primer nivel de la laplaciana
    pyramid.append(gaussiana[-1])

    return pyramid

####################################
###   CONSTRUCCIÓN DE MOSAICOS   ###
####################################

""" Dadas dos imágenes calcula los keypoints y descriptores para obtener los matches
usando "BruteForce+crossCheck". Devuelve la imagen compuesta.
- img1: Primera imagen para el match.
- img2: Segunda imagen para el match.
- n (op): número de matches a mostrar. Por defecto 50.
- flag (op): indica si se muestran los keypoints y los matches (0) o solo los matches (2).
            Por defecto 2.
- flagReturn (op): indica si debemos devolver los keypoints y matches o la imagen.
            Por defecto devolvemos la imagen.
"""
def getMatches_BFCC(img1, img2, n = 50, flag = 2, flagReturn = 1):
    # Inicializamos el descriptor AKAZE
    detector = cv2.AKAZE_create()
    # Se obtienen los keypoints y los descriptores de las dos imágenes
    keypoints1, descriptor1 = detector.detectAndCompute(img1, None)
    keypoints2, descriptor2 = detector.detectAndCompute(img2, None)

    # Se crea el objeto BFMatcher activando la validación cruzada
    bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck = True)
    # Se consiguen los puntos con los que hace match
    matches1to2 = bf.match(descriptor1, descriptor2)

    if len(matches1to2)<=n:
        n = len(matches1to2)
    # Se ordenan los matches dependiendo de la distancia entre ambos
    #matches1to2 = sorted(matches1to2, key = lambda x:x.distance)[0:n]
    # Se guardan n puntos aleatorios
    matches1to2 = sample(matches1to2, n)

    # Imagen con los matches
    img_match = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches1to2, outImg=None)

    # El usuario nos indica si quiere los keypoints y matches o la imagen
    if flagReturn:
        return img_match
    else:
        return keypoints1, keypoints2, matches1to2

""" Dadas dos imágenes calcula los keypoints y descriptores para obtener los matches
usando "Lowe-Average-2NN". Devuelve la imagen compuesta.
Si se indica el flag "improve" como True, elegirá los mejores matches.
- img1: Primera imagen para el match.
- img2: Segunda imagen para el match.
- n (op): número de matches a mostrar. Por defecto 50.
- ratio (op): Radio para la distancia entre puntos. Por defecto 0.8.
- flag (op): indica si se muestran los keypoints y los matches (0) o solo los matches (2).
            Por defecto 2.
- flagReturn (op): indica si debemos devolver los keypoints y matches o la imagen.
            Por defecto devolvemos la imagen.
"""
def getMatches_LA2NN(img1, img2, n = 50, ratio = 0.8, flag = 2, flagReturn = 1):
    # Inicializamos el descriptor AKAZE
    detector = cv2.AKAZE_create()
    # Se obtienen los keypoints y los descriptores de las dos imágenes
    keypoints1, descriptor1 = detector.detectAndCompute(img1, None)
    keypoints2, descriptor2 = detector.detectAndCompute(img2, None)

    # Se crea el objeto BFMatcher
    bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)
    # Escogemos los puntos con los que hace match indicando los vecinos más cercanos para la comprobación (2)
    matches1to2 = bf.knnMatch(descriptor1, descriptor2, k=2)

    # Mejora de los matches -> los puntos que cumplan con un radio en concreto
    best1to2 = []
    # Se recorren todos los matches
    for p1, p2 in matches1to2:
        if p1.distance < ratio * p2.distance:
            best1to2.append([p1])

    if len(best1to2)<=n:
        n=len(best1to2)
    # Se ordenan los matches dependiendo de la distancia entre ambos
    #matches1to2 = sorted(best1to2, key = lambda x:x[0].distance)[0:n]
    # Se guardan n puntos aleatorios
    matches1to2 = sample(best1to2, n)

    # Imagen con los matches
    img_match = cv2.drawMatchesKnn(img1, keypoints1, img2, keypoints2, best1to2, outImg=None)

    # El usuario nos indica si quiere los keypoints y matches o la imagen
    if flagReturn:
        return img_match
    else:
        return keypoints1, keypoints2, best1to2

""" Calcula la homografía entre dos imágenes.
- img1: primera imagen.
- img2: segunda imagen.
- flag (op): si vale 1 se calculará con Lowe-Average-2NN y si vale 0
    con BruteForce+crossCheck. Por defecto vale 1.
"""
def getHomography(img1, img2, flag=1):
    # Obtenemos los keyPoints y matches entre las dos imagenes.
    if(flag):
        kpts1, kpts2, matches = getMatches_LA2NN(img1, img2, flagReturn=0)
        # Ordeno los puntos para usar findHomography
        puntos_origen = np.float32([kpts1[punto[0].queryIdx].pt for punto in matches]).reshape(-1, 1, 2)
        puntos_destino = np.float32([kpts2[punto[0].trainIdx].pt for punto in matches]).reshape(-1, 1, 2)
    else:
        kpts1, kpts2, matches = getMatches_BFCC(img1, img2, flagReturn=0)
        # Ordeno los puntos para usar findHomography
        puntos_origen = np.float32([kpts1[punto.queryIdx].pt for punto in matches]).reshape(-1, 1, 2)
        puntos_destino = np.float32([kpts2[punto.trainIdx].pt for punto in matches]).reshape(-1, 1, 2)
    # Llamamos a findHomography
    homografia , _ = cv2.findHomography(puntos_origen, puntos_destino, cv2.RANSAC, 1)
    return homografia

""" Calcula el mosaico resultante de 2 imágenes.
- list: Lista de imágenes.
"""
def getMosaic(img1, img2):
    width = img1.shape[1] + img2.shape[1]  # Ancho del mosaico
    height = img1.shape[0]                 # Alto del mosaico
    #print("El mosaico resultante tiene tamaño ({}, {})".format(width, height))

    # Homografía 1
    hom1 = np.matrix([[1,0,0],[0,1,0],[0,0,1]], dtype=float)
    res1 = cv2.warpPerspective(img1, hom1, (width, height))
    # Homografía 2
    hom2 = getHomography(img2, img1, 1)
    hom2 = hom1 * hom2
    res2 = cv2.warpPerspective(img2, hom1*hom2, (width, height))

    return res1, res2

########################
###   BURT ADELSON   ###
########################

"""  Proyeccion cilindrica de una imagen
- img: imagen a proyectar.
- f: distancia focal.
- s: factor de escalado.
"""
def cylindricalProjection(img, f, s):
    if len(img.shape) == 3:
        # Si está en color separamos en los distintos canales
        canals = cv2.split(img)
        canals_proy = []
        for n in range(len(canals)):
            # Proyección cilindrica
            canals_proy.append(cylindricalProjection(canals[n],f,s))
        # Mezclamos canales
        proyected=cv2.merge(canals_proy)

    else:
        proyected = np.zeros(img.shape) # Imagen proyectada
        y_center = img.shape[0]/2       # centro y
        x_center = img.shape[1]/2       # centro x
        # Proyectamos la imagen
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                y_proy = floor(s * ((i-y_center) / np.sqrt((j-x_center)*(j-x_center) + f*f)) + y_center)
                x_proy = floor(s * np.arctan((j-x_center) / f) + x_center)
                proyected[y_proy][x_proy] = img[i][j]
        # Normalizamos al tipo uint8
        proyected = cv2.normalize(proyected, proyected, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return proyected

"""  Inversa de la proyeccion cilindrica de una imagen
- img: imagen a proyectar.
- f: distancia focal.
- s: factor de escalado.
"""
def cylindricalProjectionInverse(img, f, s):
    if len(img.shape) == 3:
        # Si está en color separamos en los distintos canales
        canals = cv2.split(img)
        canals_proy = []
        for n in range(len(canals)):
            # Proyección cilindrica
            canals_proy.append(cylindricalProjection(canals[n],f,s))
        # Mezclamos canales
        proyected=cv2.merge(canals_proy)

    else:
        proyected = np.zeros(img.shape) # Imagen proyectada
        y_center = img.shape[0]/2       # centro y
        x_center = img.shape[1]/2       # centro x
        # Proyectamos la imagen
        for y_proy in range(img.shape[0]):
            for x_proy in range(img.shape[1]):
                j = floor(np.tan((x_proy - x_center) / s) * f + x_center)
                i = floor((y_proy - y_center) / s * np.sqrt((j-x_center)*(j-x_center) + f*f) + y_center)
                proyected[i][j] = img[y_proy][x_proy]
        # Normalizamos al tipo uint8
        proyected = cv2.normalize(proyected, proyected, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return proyected

""" Dada una lista de imágenes devuelve otra con las proyecciones cilíndricas.
- list: lista de imágenes a proyectar.
- f: distancia focal.
- s: factor de escalado.
- title (op): título. Por defecto "Imagen".
"""
def cylindricalProjectionList(list, f, s, title="Imagen"):
    proy = []

    print("Calculando las proyecciones cilíndricas de '" + title + "'")
    for i in range(len(list)):
        proy.append(cylindricalProjection(list[i], f, s))

    return proy

"""  Proyeccion esférica de una imagen
- img: imagen a proyectar.
- f: distancia focal.
- s: factor de escalado.
"""
def sphericalProjection(img, f, s):
    if len(img.shape) == 3:
        # Si está en color separamos en los distintos canales
        canals = cv2.split(img)
        canals_proy = []
        for n in range(len(canals)):
            # Proyección esférica
            canals_proy.append(sphericalProjection(canals[n],f,s))
        # Mezclamos canales
        proyected=cv2.merge(canals_proy)

    else:
        proyected = np.zeros(img.shape) # Imagen proyectada
        y_center = img.shape[0]/2       # centro y
        x_center = img.shape[1]/2       # centro x
        # Proyectamos la imagen
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                y_proy = floor(s * np.arctan((i-y_center) / np.sqrt((j-x_center)*(j-x_center)+f*f)) + y_center)
                x_proy = floor(s * np.arctan((j-x_center) / f) + x_center)
                proyected[y_proy][x_proy] = img[i][j]
        # Normalizamos al tipo uint8
        proyected = cv2.normalize(proyected, proyected, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return proyected

""" Dada una lista de imágenes devuelve otra con las proyecciones esféricas.
- list: lista de imágenes a proyectar.
- f: distancia focal.
- s: factor de escalado.
- title (op): título. Por defecto "Imagen".
"""
def sphericalProjectionList(list, f, s, title="Imagen"):
    proy = []

    print("Calculando las proyecciones esféricas de '" + title + "'")
    for i in range(len(list)):
        proy.append(sphericalProjection(list[i], f, s))

    return proy

""" Limpia el área que no está en ninguna imagen.
- img1: primera imagen a tratar.
- img2: segunda imagen a tratar.
"""
def cleanImage(img1, img2):
    mask = np.zeros((img1.shape[0], img1.shape[1]))
    mask[np.nonzero(img1)[0:2]] = 1
    mask[np.nonzero(img2)[0:2]] = 2

    # Si la imagen está a color creamos una copia en B/N
    if len(mask.shape) == 3:
        copia_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        copia_mask = mask

    col = np.any(copia_mask.T != 0,  axis = 1)  # columnas de 0s
    raw = np.any(copia_mask.T != 0, axis = 0)   # filas de 0s

    # Borramos filas y columnas sobrantes
    mask = mask[:,col][raw,:]
    img1 = img1[:,col][raw,:]
    img2 = img2[:,col][raw,:]
    mask[mask==1]=0
    mask[mask==2]=1

    return img1, img2

""" Ajusta las imagenes al mismo tamaño (devuelve formato uint32).
- img1: primera imagen a ajustar.
- img2: segunda imagen a ajustar.
"""
def adjustImages(img1, img2):
    # Obtenemos las dimensiones de las imágenes
    y1, x1 = img1.shape[0:2]
    y2, x2 = img2.shape[0:2]

    # Obtenemos las dimensiones mas pequeñas tanto para la x como para la y
    x_min = min(x1, x2)
    y_min = min(y1, y2)

    # Calculamos la diferencia respecto a las nuevas dimensiones
    y1_dif = max(y1 - y_min, 0); x1_dif = max(x1 - x_min, 0)
    y2_dif = max(y2 - y_min, 0); x2_dif = max(x2 - x_min, 0)

    # Obtenemos las nuevas dimensiones (recorte a la imagen de los bordes)
    y10 = y1_dif // 2 + y1_dif % 2; y11 = y1_dif // 2
    x10 = x1_dif // 2 + x1_dif % 2; x11 = x1_dif // 2
    y20 = y2_dif // 2 + y2_dif % 2; y21 = y2_dif // 2
    x20 = x2_dif // 2 + x2_dif % 2; x21 = x2_dif // 2

    # Ajustamos ambas imágenes a las mismas dimensiones y formateamos
    new_img1 = img1[y10:y1 - y11, x10:x1 - x11]
    new_img2 = img2[y20:y2 - y21, x20:x2 - x21]
    new_img1 = np.uint32(new_img1)
    new_img2 = np.uint32(new_img2)

    return new_img1, new_img2

""" Para cada nivel de la Laplaciana mezcla la primera mitad de la imagen
de la primera pirámide con la segunda de la segunda imagen.
- laplaciana1: primera pirámide a mezclar.
- laplaciana1: segunda pirámide a mezclar.
"""
def mixLaplacians(laplaciana1, laplaciana1):
    finalLaplacian = []
    for i in range(len(laplaciana1)):
        aux = np.zeros(laplaciana1[i].shape, laplaciana1[i].dtype)
        mitad = laplaciana1[i].shape[1] // 2
        aux[:, :mitad, ...] = laplaciana1[i][:, :mitad, ...]
        aux[:, -mitad:, ...] = laplaciana1[i][:, -mitad:, ...]
        if laplaciana1[i].shape[1] % 2 == 1:
            # Numero de columnas impar -> media aritmética
            aux[:, mitad, ...] = (laplaciana1[i][:, mitad, ...] + laplaciana1[i][:, mitad, ...])/2

        finalLaplacian.append(aux)
    return finalLaplacian

""" Restaura una pirámide laplaciana.
- pyramid: Pirámide a restaurar.
"""
def laplacianRestoring(pyramid):
    # Cogemos el ultimo nivel de la laplaciana
    recuperacion = pyramid[-1]
    # Recorremos todos los niveles
    for i in range(len(pyramid) - 1):
        # Cogemos el siguiente
        siguiente = pyramid[-2 - i]
        # Transformamos al formato adecuado para el upsample
        recuperacion = np.float32(recuperacion)
        # Realizamos la convolucion y aumento
        aumento = convolution(recuperacion, siguiente)
        # Recuperamos el formato
        recuperacion = np.uint32(recuperacion)
        # Realizamos la suma y recuperamos la imagen
        recuperacion = aumento + siguiente
    # Guardamos la imagen en el formato uint32
    recuperacion = np.uint32(recuperacion)
    return recuperacion

""" Aplica el algoritmo Burt Adelson a dos imágenes.
- img1: primera imagen a tratar.
- img2: segunda imagen a tratar.
- levels (op): niveles de la pirámide laplaciana que se usa en el algoritmo. Por defecto 6.
"""
def BurtAdelson(img1, img2, levels=6):
    res1, res2 = getMosaic(img1, img2)             # Calulamos el mosaico
    res1, res2 = cleanImage(res1, res2)            # Limpiamos las imágenes
    res1, res2 = adjustImages(res1, res2)          # Ajustamos imágenes a uint32 y mismo tamaño
    lap1 = laplacianPyramid(res1, levels)          # Pirámide laplacia 1
    lap2 = laplacianPyramid(res2, levels)          # Pirámide laplacia 1
    lap_splined = mixLaplacians(lap1, lap2)        # Pirámide laplaciana combinada
    img_splined = laplacianRestoring(lap_splined)  # Restauramos la laplaciana combinada
    np.clip(img_splined, 0, 255, out=img_splined)  # Normalizamos al rango [0,255]
    img_splined = np.uint8(img_splined)            # Formato uint8 para visualización
    return img_splined

""" Aplica el algoritmo Burt Adelson a N imágenes.
- img_list: lista de imágenes a tratar.
- levels (op): niveles de la pirámide laplaciana que se usa en el algoritmo. Por defecto 6.
- title (op): título del conjunto de imágenes.
"""
def BurtAdelson_N(img_list, levels=6, title="Imágenes"):
    print("BurtAdelson al conjunto de imágenes '" + title + "'")
    centro = len(img_list)//2
    right = BurtAdelson(img_list[centro], img_list[centro+1], levels)
    left = BurtAdelson(img_list[centro-1], img_list[centro], levels)

    for n in range(centro, -1, -1):
        left = BurtAdelson(left, img_list[n], levels)
    for n in range(centro, len(img_list)):
        right = BurtAdelson(right, img_list[n], levels)

    mosaic = BurtAdelson(left, right, levels)

    return mosaic


#######################
###       MAIN      ###
#######################

""" Programa principal. """
if __name__ == "__main__":
    print("\n----------   PREPROCESANDO IMÁGENES   ----------")
    # Leemos las imágenes que necesitamos
    carlosV =   leer_imagen("imagenes/carlosV.jpg", 1)
    yos =   [leer_imagen("imagenes/yosemite1.jpg", 1),
             leer_imagen("imagenes/yosemite2.jpg", 1),
             leer_imagen("imagenes/yosemite3.jpg", 1)]
    al =    [leer_imagen("imagenes/al1.png", 1),
             leer_imagen("imagenes/al2.png", 1),
             leer_imagen("imagenes/al3.png", 1)]
    #alham = [leer_imagen("imagenes/alham1.png", 1),
    #         leer_imagen("imagenes/alham2.png", 1),
    #         leer_imagen("imagenes/alham3.png", 1),
    #         leer_imagen("imagenes/alham4.png", 1)]

    levels = 6      # Niveles para las pirámides en BurtAdelson

    yosProyCil = cylindricalProjectionList(yos, 800, 800, "Yosemite")
    alProyCil = cylindricalProjectionList(al, 800, 800, "Alhambra 1")
    #alhamProyCil = cylindricalProjectionList(alham, 900, 900, "Alhambra 2")
    yosProyEsf = sphericalProjectionList(yos, 800, 800, "Yosemite")
    alProyEsf = sphericalProjectionList(al, 600, 600, "Alhambra 1")
    #alhamProyEsf = sphericalProjectionList(alham, 900, 900, "Alhambra 2")

    print("\n----------   PROBANDO PROYECCIONES   ----------")
    # Ejemplo para probar proyecciones cilíndricas
    print("Proyecciones cilíndricas")
    # Alhambra
    proy_cilindrica = cylindricalProjection(al[1], 600, 600)
    pintaI(proy_cilindrica, 1, "Proyeccion cilindrica. f=600. s=600.", "VC Proyecto - BurtAdelson")
    proy_cilindrica = cylindricalProjection(al[1], 800, 800)
    pintaI(proy_cilindrica, 1, "Proyeccion cilindrica. f=800. s=800.", "VC Proyecto - BurtAdelson")
    # Carlos V
    proy_cilindrica = cylindricalProjection(carlosV, 600, 600)
    pintaI(proy_cilindrica, 1, "Proyeccion cilindrica. f=600. s=600.", "VC Proyecto - BurtAdelson")
    proy_cilindrica = cylindricalProjection(carlosV, 900, 900)
    pintaI(proy_cilindrica, 1, "Proyeccion cilindrica. f=900. s=900.", "VC Proyecto - BurtAdelson")

    # Ejemplo para probar proyecciones esféricas
    print("Proyecciones esféricas")
    # Alhambra
    proy_esferica = sphericalProjection(al[1], 600, 600)
    pintaI(proy_esferica, 1, "Proyeccion esférica. f=600. s=600.", "VC Proyecto - BurtAdelson")
    proy_esferica = sphericalProjection(al[1], 800, 800)
    pintaI(proy_esferica, 1, "Proyeccion esférica. f=800. s=800.", "VC Proyecto - BurtAdelson")
    # Carlos V
    proy_esferica = sphericalProjection(carlosV, 600, 600)
    pintaI(proy_esferica, 1, "Proyeccion esférica. f=600. s=600.", "VC Proyecto - BurtAdelson")
    proy_esferica = sphericalProjection(carlosV, 900, 900)
    pintaI(proy_esferica, 1, "Proyeccion esférica. f=900. s=900.", "VC Proyecto - BurtAdelson")

    input("Pulsa 'Enter' para continuar\n")

    print("\n----------   PROBANDO BURTADELSON   ----------")
    print("El número de NIVELES de la pirámide para Burt-Adelson es: {}".format(levels))
    # Ejemplo para probar un mosaico de yosemite
    yosPanCil = BurtAdelson_N(yosProyCil, levels, "Yosemite (cilíndrica - {} niveles)".format(levels))
    yosPanEsf = BurtAdelson_N(yosProyEsf, levels, "Yosemite (esférica - {} niveles)".format(levels))
    pintaI(yosPanCil, 1, "Mosaico de Yosemite (cilíndrica - {} niveles).".format(levels), "VC Proyecto - BurtAdelson")
    pintaI(yosPanEsf, 1, "Mosaico de Yosemite (esférica - {} niveles).".format(levels), "VC Proyecto - BurtAdelson")

    # Ejemplo para probar un mosaico de la alhambra
    alPanCil = BurtAdelson_N(alProyCil, levels, "Alhambra (cilíndrica - {} niveles)".format(levels))
    alPanEsf = BurtAdelson_N(alProyEsf, levels, "Alhambra (esférica - {} niveles)".format(levels))
    pintaI(alPanCil, 1, "Mosaico de la Alhambra (cilíndrica - {} niveles).".format(levels), "VC Proyecto - BurtAdelson")
    pintaI(alPanEsf, 1, "Mosaico de la Alhambra (esférica - {} niveles).".format(levels), "VC Proyecto - BurtAdelson")

    # Ejemplo para probar un mosaico de la alhambra 2
    #alhamPanCil = BurtAdelson_N(alhamProyCil, levels, "Alhambra (cilíndrica - {} niveles)".format(levels))
    #alhamPanEsf = BurtAdelson_N(alhamProyEsf, levels, "Alhambra (esférica - {} niveles)".format(levels))
    #pintaI(alhamPanCil, 1, "Mosaico de la Alhambra (cilíndrica - {} niveles).".format(levels), "VC Proyecto - BurtAdelson")
    #pintaI(alhamPanEsf, 1, "Mosaico de la Alhambra (esférica - {} niveles).".format(levels), "VC Proyecto - BurtAdelson")

    input("Pulsa 'Enter' para continuar\n")
