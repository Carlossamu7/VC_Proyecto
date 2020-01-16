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
import warnings

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
    plt.show()
    image = image.astype(np.float64)    # Devolvemos su formato

###############################
###   FILTROS Y PIRÁMIDES   ###
###############################

""" Aplica una máscara Gaussiana 2D. Devuelve la imagen con las máscara aplicada.
- image: la imagen a tratar.
- kernel_x: kernel en las dirección X.
- kernel_y: kernel en las dirección Y.
- border_type (op): tipo de bordes. BORDER_DEFAULT.
"""
def convolution(image, kernel_x, kernel_y, border_type = cv2.BORDER_DEFAULT):
    kernel_x = np.transpose(kernel_x)
    kernel_x = cv2.flip(kernel_x, 0)
    kernel_y = cv2.flip(kernel_y, 1)
    im_conv = cv2.filter2D(image, -1, kernel_x, borderType = border_type)
    im_conv = cv2.filter2D(im_conv, -1, kernel_y, borderType = border_type)
    return im_conv

""" Aplica una máscara Gaussiana 2D. Devuelve la imagen con las máscara aplicada.
- image: la imagen a tratar.
- sigma_x: sigma en la dirección X.
- sigma_y (op): sigma en la dirección Y. Por defecto sigma_y = sigma_x
- k_size_x (op): tamaño del kernel en dirección X (positivo e impar). Por defecto es 0, se obtiene a través de sigma.
- k_size_y (op): tamaño del kernel en dirección Y (positivo e impar). Por defecto es 0, se obtiene a través de sigma.
- border_type (op): tipo de bordes. BORDER_DEFAULT.
"""
def gaussian_blur(image, sigma_x, sigma_y = 0, k_size_x = 0, k_size_y = 0, border_type = cv2.BORDER_DEFAULT):
    if sigma_y == 0:
        sigma_y = sigma_x
    if k_size_x == 0:
        k_size_x = int(6*sigma_x + 1)
    if k_size_y == 0:
        k_size_y = int(6*sigma_y + 1)

    kernel_x = cv2.getGaussianKernel(k_size_x, sigma_x)
    kernel_y = cv2.getGaussianKernel(k_size_y, sigma_y)
    return convolution(image, kernel_x, kernel_y, border_type)

""" Aplica máscara laplaciana a imagen. Devuelve la imagen con la máscara aplicada.
- im: Imagen a la que aplicar la máscara.
- k_size: Tamaño del kernel para Laplacian.
- border_type (op): Tipo de borde. Por defecto BORDER_DEFAULT.
"""
def laplacian_gaussian(image, k_size, border_type = cv2.BORDER_DEFAULT):
    k_x1, k_y1 = cv2.getDerivKernels(2, 0, k_size, normalize = True)
    k_x2, k_y2 = cv2.getDerivKernels(0, 2, k_size, normalize = True)
    im_convolution_x = convolution(image, k_x1, k_y1, border_type)
    im_convolution_y = convolution(image, k_x2, k_y2, border_type)
    return im_convolution_x + im_convolution_y

""" Hace un subsampling de la imagen pasada como argumento. Devuelve la imagen recortada.
- image: imagen a recortar.
"""
def subsampling(image):
    n_fil = int(image.shape[0]/2)
    n_col = int(image.shape[1]/2)
    cp = np.copy(image)

    for a in range(0, n_fil):
        cp = np.delete(cp, a, axis = 0)
    for a in range(0, n_col):
        cp = np.delete(cp, a, axis = 1)

    return cp

""" Hace un upsampling de la imagen pasada como argumento. Devuelve la imagen agrandada.
- image: imagen a agrandar.
- n_fil: número de filas de la matriz resultante.
- n_col: número de columnas de la matriz resultante.
"""
def upsampling(image, n_fil, n_col):
    fil = False
    col = False

    if n_fil % 2 == 1:
        n_fil = n_fil-1
        fil = True

    if n_col % 2 == 1:
        n_col = n_col-1
        col = True

    if len(image.shape)==2:
        if fil and col:
            salida = np.zeros((n_fil+1, n_col+1))
        elif fil:
            salida = np.zeros((n_fil+1, n_col))
        elif col:
            salida = np.zeros((n_fil, n_col+1))
        else:
            salida = np.zeros((n_fil, n_col))

        # Relleno la matriz, en cada iteración escribo 4 elementos de la matriz de salida
        for i in range(0, n_fil, 2):
            for j in range(0, n_col, 2):
                salida[i][j] = image[int(i/2)][int(j/2)]
                salida[i+1][j] = image[int(i/2)][int(j/2)]
                salida[i][j+1] = image[int(i/2)][int(j/2)]
                salida[i+1][j+1] = image[int(i/2)][int(j/2)]

        # Si el número de filas era impar escribo la última fila la cual borré con n_fil = n_fil-1
        if fil:
            for j in range(0, n_col, 2):
                salida[n_fil][j] = image[image.shape[0]-1][int(j/2)]
                salida[n_fil][j+1] = image[image.shape[0]-1][int(j/2)]

        # Si el número de columnas era impar escribo la última columna la cual borré con n_col = n_col-1
        if col:
            for i in range(0, n_fil, 2):
                salida[i][n_col] = image[int(i/2)][image.shape[1]-1]
                salida[i+1][n_col] = image[int(i/2)][image.shape[1]-1]

            # Si se da el caso de que n_fil y n_col eran impares falta el último elemento por escribir en cada banda
            if fil and col:
                salida[n_fil][n_col] = image[image.shape[0]-1][image.shape[1]-1]

    if len(image.shape)==3:
        if fil and col:
            salida = np.zeros((n_fil+1, n_col+1, image.shape[2]))
        elif fil:
            salida = np.zeros((n_fil+1, n_col, image.shape[2]))
        elif col:
            salida = np.zeros((n_fil, n_col+1, image.shape[2]))
        else:
            salida = np.zeros((n_fil, n_col, image.shape[2]))

        # Escribo en todos los canales
        for k in range(0, image.shape[2]):
            # Relleno la matriz, en cada iteración escribo 4 elementos de la matriz de salida
            for i in range(0, n_fil, 2):
                for j in range(0, n_col, 2):
                    salida[i][j][k] = image[int(i/2)][int(j/2)][k]
                    salida[i+1][j][k] = image[int(i/2)][int(j/2)][k]
                    salida[i][j+1][k] = image[int(i/2)][int(j/2)][k]
                    salida[i+1][j+1][k] = image[int(i/2)][int(j/2)][k]

            # Si el número de filas era impar escribo la última fila la cual borré con n_fil = n_fil-1
            if fil:
                for k in range(0, image.shape[2]):
                    #salida[n_fil,:,k][::2] = image[image.shape[0]-1,:,k]
                    #salida[n_fil,:,k][1::2] = image[image.shape[0]-1,:,k]
                    for j in range(0, n_col, 2):
                        salida[n_fil][j][k] = image[image.shape[0]-1][int(j/2)][k]
                        salida[n_fil][j+1][k] = image[image.shape[0]-1][int(j/2)][k]

            # Si el número de columnas era impar escribo la última columna la cual borré con n_col = n_col-1
            if col:
                for k in range(0, image.shape[2]):
                    #salida[:,n_col,k][::2] = image[:,image.shape[1]-1,k]
                    #salida[:,n_col,k][1::2] = image[:,image.shape[1]-1,k]
                    for i in range(0, n_fil, 2):
                        salida[i][n_col,k] = image[int(i/2)][image.shape[1]-1][k]
                        salida[i+1][n_col][k] = image[int(i/2)][image.shape[1]-1][k]

                    # Si se da el caso de que n_fil y n_col eran impares falta el último elemento por escribir en cada banda
                    if fil and col:
                        salida[n_fil][n_col][k] = image[image.shape[0]-1][image.shape[1]-1][k]

    return salida

""" Genera representación de pirámide gaussiana. Devuelve la lista de imágenes que forman la pirámide gaussiana.
- image: La imagen a la que generar la pirámide gaussiana.
- levels (op): Número de niveles de la pirámide gaussiana. Por defecto 4.
- border_type (op): Tipo de borde a utilizar. Por defecto BORDER DEFAULT.
"""
def gaussian_pyramid(image, levels = 4, border_type = cv2.BORDER_DEFAULT):
    pyramid = [image]
    blur = np.copy(image)
    for n in range(levels):
        blur = gaussian_blur(blur, 1, 1, 7, 7, border_type = border_type)
        blur = subsampling(blur)
        pyramid.append(blur)
    return pyramid

""" Genera representación de pirámide laplaciana. Devuelve la lista de imágenes que forman la pirámide laplaciana.
- image: La imagen a la que generar la pirámide laplaciana.
- levels (op): Número de niveles de la pirámide laplaciana. Por defecto 4.
- border_type (op): Tipo de borde a utilizar. BORDER DEFAULT.
"""
def laplacian_pyramid(image, levels = 4, border_type = cv2.BORDER_DEFAULT):
    gau_pyr = gaussian_pyramid(image, levels+1, border_type)
    lap_pyr = []
    for n in range(levels):
        gau_n_1 = upsampling(gau_pyr[n+1], gau_pyr[n].shape[0], gau_pyr[n].shape[1])
        #gau_n_1 = 4*gaussian_blur(gau_n_1, 1, 1, 7, 7)   # Otra opción para la laplaciana: poniendo 0s.
        gau_n_1 = gaussian_blur(gau_n_1, 1, 1, 7, 7, border_type = border_type)
        lap_pyr.append(normaliza(gau_pyr[n] - gau_n_1, "Etapa {} de la pirámide gaussiana.".format(n)))
    lap_pyr.append(gau_pyr[levels])
    return lap_pyr

# Funciones que dado un sigma genera una máscara Gaussiana
def Mascara_Gaussiana(x,sigma):
    # Mascara gaussiana
    return exp(-((x*x)/(2*(sigma*sigma))))

def Gaussiana(sigma):

    # Definimos un array de valores desde -3sigma hasta 3sigma+1
    mascara = np.arange(-floor(3*sigma), floor(3*sigma+1))

    # Calculamos la funcion para cada valor del vector
    vector = []
    for i in mascara:
        vector.append([Mascara_Gaussiana(i,sigma)])

    # Lo transformamos en un array
    mascara = np.array(vector)

    # Dividimos cada elemento de la mascara por la media de todos ellos
    mascara = np.divide(mascara, np.sum(mascara))
    return mascara

# Función que dada una imagen y una matriz con un tamaño, sobremuestrea la imagen
# al tamaño de la matriz y aplica una convolución con kernel gaussiano sobre
# la misma
def convolucionSeparable(img, tam, kernel_gaussiano):
    # Aumentamos el tamaño de la imagen al deseado
    im_expanded = np.zeros(tam.shape, img.dtype)
    im_expanded[::2, ::2, ...] = img

    # Generamos el kernel a utilizar
    if kernel_gaussiano:
        mascara = Gaussiana(1)
    else:
        mascara = 1.0 / 10 * np.array([1, 5, 8, 5, 1])

    copia = im_expanded # Realizo una copia de la imagen sobre la que trabajare

    for c in range(2): # Una iteracion para filas, otra para columnas
        aux = copia.copy() # Realizo una copia auxiliar
        for i in range(copia.shape[0]): # Recorro las filas
            # Aplico el filtro en las filas de la imagen, almacenandolo en
            # la imagen auxiliar, con la mascara calculada anteriormente
            # y el borde indicado
            aux[i,:] = cv2.filter2D(src=copia[i,:],dst=aux[i,:],ddepth=cv2.CV_32F,kernel=mascara,borderType=cv2.BORDER_DEFAULT)
        # Realizo la traspuesta para poder actuar en filas y columnas como si
        # ambas fueran filas
        copia = cv2.transpose(aux)

    return copia

# Funcion que realiza el proceso de pirámide Gaussiana
def PiramideGaussiana(img, levels=8):
    pyramid = [img] # imagen original
    actual = img
    for i in range(levels):
        actual = np.uint8(actual)
        # Convolucion + subsampling
        actual = cv2.pyrDown(actual)
        actual = np.uint32(actual)
        # Guardamos la imagen
        pyramid.append(actual)
    return pyramid


def PiramideLaplaciana(img, levels=8):
    gaussiana = PiramideGaussiana(img, levels)
    pyramid = []
    for i in range(len(gaussiana) - 1):
        actual = gaussiana[i]
        siguiente = gaussiana[i+1]
        # Upsampling
        siguiente = np.float32(siguiente)
        aumento = convolucionSeparable(siguiente, actual, False)
        siguiente = np.uint32(siguiente)

        # Hacemos la diferencia
        pyramid.append(actual - aumento)

    # Ultimo nivel de la gaussiana que es el primer nivel de la laplaciana
    pyramid.append(gaussiana[-1])
    #for i in range(len(pyramid)):
        #pintaI(pyramid[i])
    return pyramid

def PiramideLaplaciana_nuevo(img, levels=8):
    gaussiana = PiramideGaussiana(img, levels)
    pyramid = []
    for i in range(len(gaussiana) - 1):
        actual = gaussiana[i]
        siguiente = gaussiana[i+1]

        siguiente = np.uint8(siguiente)
        siguiente = cv2.pyrUp(siguiente, dstsize=(actual.shape[1],actual.shape[0]))
        siguiente = np.uint32(siguiente)

        # Hacemos la diferencia
        pyramid.append(actual - siguiente)

    # Ultimo nivel de la gaussiana que es el primer nivel de la laplaciana
    #pyramid.append(gaussiana[-1])
    #for i in range(len(pyramid)):
        #pintaI(pyramid[i])
    return pyramid

def RestaurarLaplaciana(piramide):
    # Cogemos el ultimo nivel de la laplaciana
    recuperacion = piramide[-1]
    # Recorremos todos los niveles
    for i in range(len(piramide) - 1):
        # Cogemos el siguiente
        siguiente = piramide[-2 - i]
        # Transformamos al formato adecuado para el upsample
        recuperacion = np.float32(recuperacion)
        # Realizamos la convolucion y aumento
        aumento = convolucionSeparable(recuperacion, siguiente, False)
        # Recuperamos el formato
        recuperacion = np.uint32(recuperacion)
        # Realizamos la suma y recuperamos la imagen
        recuperacion = aumento + siguiente
    # Guardamos la imagen en el formato uint32
    recuperacion = np.uint32(recuperacion)
    return recuperacion

def RestaurarLaplaciana_nuevo(piramide):
    # Cogemos el ultimo nivel de la laplaciana
    recuperacion = piramide[-1]
    for i in range(len(piramide) - 1):
        siguiente = piramide[-2 - i]
        # Realizamos la convolucion y aumento
        aumento = cv2.pyrUp(recuperacion, dstsize=(siguiente.shape[1],siguiente.shape[0]))
        recuperacion = aumento + siguiente
    return recuperacion

####################################
###   CONSTRUCCIÓN DE MOSAICOS   ###
####################################

""" Dadas dos imágenes calcula los keypoints y descriptores para obtener los matches
usando "BruteForce+crossCheck". Devuelve la imagen compuesta.
- img1: Primera imagen para el match.
- img2: Segunda imagen para el match.
- n (op): número de matches a mostrar. Por defecto 100.
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

    # Escogemos n elementos aleatorios para mostrarlos
    #elem_correspondencias = sample(range(len(matches1to2)), n)
    #best1to2 = []
    #for i in range(len(elem_correspondencias)):
    #    best1to2.append(matches1to2[i])

    if len(matches1to2)<=n:
        n = len(matches1to2)
    # Se ordenan los matches dependiendo de la distancia entre ambos
    matches1to2 = sorted(matches1to2, key = lambda x:x.distance)[0:n]
    # Se guardan n puntos aleatorios
    #matches1to2 = sample(matches1to2, n)

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
- n (op): número de matches a mostrar. Por defecto 100.
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

""" Calcula el mosaico resultante de N imágenes.
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

""" Calcula la homografia que lleva la imagen al centro del mosaico.
- img: Imagen
- mosaicWidth: ancho del mosaico.
- mosaicHeight: alto del mosaico.
"""
def identityHomography(img, mosaicWidth, mosaicHeight):
    tx = mosaicWidth/2 - img.shape[0]/2     # Calculamos traslación en x
    ty = mosaicHeight/2 - img.shape[1]/2    # Calculamos traslación en y
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)

""" Calcula el mosaico resultante de N imágenes.
- list: Lista de imágenes.
"""
def getMosaicN(list):
    homographies = [None] * len(list)                       # Lista de homografías
    ind_center = int(len(list)/2)                           # Índice de la imagen central
    img_center =  list[ind_center]                          # Imagen central
    width = int(sum([im.shape[1] for im in list]) * 0.9)    # Ancho del mosaico
    height = list[0].shape[0] * 2                           # Alto del mosaico

    print("El mosaico resultante tiene tamaño ({}, {})".format(width, height))

    # Homografía central
    hom_center = identityHomography(img_center, width, height)
    homographies[ind_center] = hom_center
    res = cv2.warpPerspective(img_center, hom_center, (width, height), borderMode=cv2.BORDER_TRANSPARENT)

    # Empezamos por el centro y vamos hacia atrás
    for i in range(0, ind_center)[::-1]:
        h = getHomography(list[i], list[i+1])
        h = np.dot(homographies[i+1], h)
        homographies[i] = h
        res = cv2.warpPerspective(list[i], h, (width, height), dst=res, borderMode=cv2.BORDER_TRANSPARENT)

    # Empezamos por el centro y vamos hacia delante
    for i in range(ind_center+1, len(list)):
        h = getHomography(list[i], list[i-1])
        h = np.dot(homographies[i-1], h)
        homographies[i] = h
        res = cv2.warpPerspective(list[i], h, (width, height), dst=res, borderMode=cv2.BORDER_TRANSPARENT)

    return res

########################
###   BURT ADELSON   ###
########################

# Funcion que implementa una proyeccion cilindrica sobre una imagen,
# dada una distancia focal f y un factor de escalado s
def ProyeccionCilindrica(imagen, f, s):
    # Si tenemos una imagen en color, separamos en canales RGB y realizamos
    # el proceso de proyeccion canal a canal
    if len(imagen.shape) == 3:
        # Separamos en los distintos canales
        canales = cv2.split(imagen)
        canales_proyecciones = []
        for n in range(len(canales)):
            # Realizamos cada proyección cilindrica
            canales_proyecciones.append(ProyeccionCilindrica(canales[n],f,s))
        # Mezclamos los canales obteniendo las proyeccion para la imagen a color
        imagen_proyectada=cv2.merge(canales_proyecciones)

    else:
        # Creamos una nueva imagen del tamaño de la original
        imagen_proyectada = np.zeros(imagen.shape)

        # Recorremos la imagen y vamos aplicando la proyeccion, como sabemos
        # nuestras coordenadas empiezan en la esquina superior izquierda de
        # nuestra imagen
        centro_anchura = imagen.shape[1]/2
        centro_altura = imagen.shape[0]/2

        for i in range(imagen.shape[0]):
            for j in range(imagen.shape[1]):
                # Sacamos los indices x' e y'
                i_proyectada = floor(s*((i-centro_altura)/np.sqrt((j-centro_anchura)*(j-centro_anchura)+f*f)) + centro_altura)
                j_proyectada = floor(s*np.arctan((j-centro_anchura)/f) + centro_anchura)
                imagen_proyectada[i_proyectada][j_proyectada] = imagen[i][j]
        # Normalizamos los datos al tipo uint8
        imagen_proyectada = cv2.normalize(imagen_proyectada,imagen_proyectada,0,255, cv2.NORM_MINMAX,cv2.CV_8U)

    return imagen_proyectada

def listaProyeccionesCilindricas(list, f, s, title):
    proy = []

    print("Calculando las proyecciones cilíndricas de '" + title + "'")
    for i in range(len(list)):
        proy.append(ProyeccionCilindrica(list[i], f, s))

    return proy

# Funcion que implementa una proyeccion cilindrica sobre una imagen,
# dada una distancia focal f y un factor de escalado s
def ProyeccionEsferica(imagen, f, s):
    # Si tenemos una imagen en color, separamos en canales RGB y realizamos
    # el proceso de proyeccion canal a canal
    if len(imagen.shape) == 3:
        # Separamos en los distintos canales
        canales = cv2.split(imagen)
        canales_proyecciones = []
        for n in range(len(canales)):
            # Realizamos cada proyección cilindrica
            canales_proyecciones.append(ProyeccionEsferica(canales[n],f,s))
        # Mezclamos los canales obteniendo las proyeccion para la imagen a color
        imagen_proyectada=cv2.merge(canales_proyecciones)

    else:
        # Creamos una nueva imagen del tamaño de la original
        imagen_proyectada = np.zeros(imagen.shape)

        # Recorremos la imagen y vamos aplicando la proyeccion, como sabemos
        # nuestras coordenadas empiezan en la esquina superior izquierda de
        # nuestra imagen
        centro_anchura = imagen.shape[1]/2
        centro_altura = imagen.shape[0]/2

        for i in range(imagen.shape[0]):
            for j in range(imagen.shape[1]):
                # Sacamos los indices x' e y'
                i_proyectada = floor(s*np.arctan((i-centro_altura)/np.sqrt((j-centro_anchura)*(j-centro_anchura)+f*f)) + centro_altura)
                j_proyectada = floor(s*np.arctan((j-centro_anchura)/f) + centro_anchura)
                imagen_proyectada[i_proyectada][j_proyectada] = imagen[i][j]
        # Normalizamos los datos al tipo uint8
        imagen_proyectada = cv2.normalize(imagen_proyectada,imagen_proyectada,0,255, cv2.NORM_MINMAX,cv2.CV_8U)

    return imagen_proyectada

def listaProyeccionesEsfericas(list, f, s, title):
    proy = []

    print("Calculando las proyecciones esféricas de '" + title + "'")
    for i in range(len(list)):
        proy.append(ProyeccionEsferica(list[i], f, s))

    return proy

# Funcion que ajusta las imagenes al mismo tamaño y al formato uint32
def AjustarImagenes(imagen1, imagen2):
    # Obtenemos las dimensiones de las imágenes
    y1, x1 = imagen1.shape[0:2]
    y2, x2 = imagen2.shape[0:2]

    # Obtenemos las dimensiones mas pequeñas tanto para la x como para la y
    nueva_x = min(x1, x2)
    nueva_y = min(y1, y2)

    # Ajustamos ambas imágenes a las mismas dimensiones
    imagen1, imagen2 = AjustarTamImagenes(imagen1, imagen2, nueva_y, nueva_x)

    # Pasamas al formato correspondiente para operar sobre ellas
    imagen1 = np.uint32(imagen1)
    imagen2 = np.uint32(imagen2)

    return imagen1, imagen2

# Funcion que dadas dos imagenes y dimensiones, las recorta para adecuarlas
def AjustarTamImagenes(imagen1, imagen2, y_ajuste, x_ajuste):
    imagenes = [imagen1,imagen2]
    nuevas_imagenes = []
    # Recorremos las imágenes
    for n in range(len(imagenes)):
        # Obtenemos las dimensiones de la imagen
        y, x = imagenes[n].shape[0:2]
        # Calculamos la diferencia respecto a las nuevas dimensiones
        diferencia_y = max(y - y_ajuste, 0)
        diferencia_x = max(x - x_ajuste, 0)
        # Obtenemos las nuevas dimensiones (recorte a la imagen de los bordes)
        y0 = diferencia_y // 2 + diferencia_y % 2
        y1 = diferencia_y // 2
        x0 = diferencia_x // 2 + diferencia_x % 2
        x1 = diferencia_x // 2

        # Guardamos la nueva imagen
        nuevas_imagenes.append(imagenes[n][y0:y - y1, x0:x - x1])

    return (nuevas_imagenes[0], nuevas_imagenes[1])

def Mezcla(Laplaciana1, Laplaciana2):
    Laplaciana_final = []
    for i in range(len(Laplaciana1)):
        nivel = np.zeros(Laplaciana1[i].shape, Laplaciana1[i].dtype)
        mitad = Laplaciana1[i].shape[1] // 2
        nivel[:, :mitad, ...] = Laplaciana1[i][:, :mitad, ...]
        nivel[:, -mitad:, ...] = Laplaciana2[i][:, -mitad:, ...]
        if Laplaciana1[i].shape[1] % 2 == 1:
            # Numero de columnas impar -> media aritmética
            nivel[:, mitad, ...] = (Laplaciana1[i][:, mitad, ...] + Laplaciana2[i][:, mitad, ...])/2

        Laplaciana_final.append(nivel)
    return Laplaciana_final

def BurtAdelson(imagen1, imagen2):
    # Ajustamos las imágenes a formato uint32 y al mismo tamaño
    imagen1, imagen2 = AjustarImagenes(imagen1, imagen2)
    # Calculamos las pirámides Laplacianas
    laplaciana1 = PiramideLaplaciana(imagen1)
    laplaciana2 = PiramideLaplaciana(imagen2)

    # Calculamos la pirámide Laplaciana combinada
    laplaciana_mezcla = Mezcla(laplaciana1, laplaciana2)
    img_restaurada = RestaurarLaplaciana(laplaciana_mezcla)

    # Normalizamos los valores que salgan del rango [0,255]
    np.clip(img_restaurada, 0, 255, out=img_restaurada)
    # Transformamos el formato de la imagen para la visualización
    img_restaurada = np.uint8(img_restaurada)
    return img_restaurada

def limpiarImagen(imagen, imagenBA1, imagenBA2):
    # Si la imagen es a colo creamos una copia en blanco y negro
    if len(imagen.shape) == 3:
        copia_imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        copia_imagen = imagen

    # Calculamos las columnas enteras de 0
    columnas = np.any(copia_imagen.T != 0,  axis = 1)
    # Calculamos las filas enteras de 0
    filas = np.any(copia_imagen.T != 0, axis = 0)

    # Quitamos esas filas y columnas que sobran a la imagen
    imagen = imagen[:,columnas][filas,:]
    imagenBA1 = imagenBA1[:,columnas][filas,:]
    imagenBA2 = imagenBA2[:,columnas][filas,:]

    return imagen, imagenBA1, imagenBA2

def mosaicoBA(imagen1, imagen2):
    area1, area2 = getMosaic(imagen1, imagen2)

    mascaraBA = np.zeros((area1.shape[0], area1.shape[1]))
    mascaraBA[np.nonzero(area1)[0:2]] = 1
    mascaraBA[np.nonzero(area2)[0:2]] = 2

    mascaraBA, area1, area2 = limpiarImagen(mascaraBA, area1, area2)
    mascaraBA[mascaraBA==1]=0
    mascaraBA[mascaraBA==2]=1
    # kuek
    mosaico = BurtAdelson(area1,area2)

    return mosaico

def mosaico_nBA(lista_imagenes):
    centro = len(lista_imagenes)//2
    print("centro={}".format(centro))
    mosaico_der = mosaicoBA(lista_imagenes[centro], lista_imagenes[centro+1])
    mosaico_izq = mosaicoBA(lista_imagenes[centro-1], lista_imagenes[centro])
    pintaI(mosaico_der)
    pintaI(mosaico_izq)
    for n in range(centro, -1, -1):
        print("Entro izquierda: {}".format(n))
        mosaico_izq = mosaicoBA(mosaico_izq, lista_imagenes[n])
        pintaI(mosaico_izq)
    for n in range(centro, len(lista_imagenes)):
        print("Entro derecha: {}".format(n))
        mosaico_der = mosaicoBA(mosaico_der, lista_imagenes[n])
        pintaI(mosaico_der)

    mosaico_final = mosaicoBA(mosaico_izq, mosaico_der)

    return mosaico_final


#######################
###       MAIN      ###
#######################

""" Programa principal. """
if __name__ == "__main__":
    # Leemos las imágenes que necesitamos
    al1 = leer_imagen("imagenes/al1.png", 1)
    yos = [leer_imagen("imagenes/yosemite1.jpg", 1),
           leer_imagen("imagenes/yosemite2.jpg", 1),
           leer_imagen("imagenes/yosemite3.jpg", 1),
           leer_imagen("imagenes/yosemite4.jpg", 1)]
    al = [leer_imagen("imagenes/al1.png", 1),
          leer_imagen("imagenes/al2.png", 1),
          leer_imagen("imagenes/al3.png", 1)]
    alham = [leer_imagen("imagenes/alham1.png", 1),
             leer_imagen("imagenes/alham2.png", 1),
             leer_imagen("imagenes/alham3.png", 1),
             leer_imagen("imagenes/alham4.png", 1)]
    #yosProy = listaProyeccionesCilindricas(yos, 800, 800, "Yosemite")
    alProy = listaProyeccionesCilindricas(al, 800, 800, "Alhambra 1")
    #alhamProy = listaProyeccionesCilindricas(alham, 900, 900, "Alhambra 2")

    # Ejemplo para probar proyecciones cilíndricas
    #proyeccion_cilindrica = ProyeccionCilindrica(al1, 600, 600)
    #pintaI(proyeccion_cilindrica, 1, "Proyeccion cilindrica. f=600. s=600.", "VC Proyecto - BurtAdelson")
    #input("Pulsa 'Enter' para continuar\n")

    # Ejemplo para probar proyecciones esféricas
    #proyeccion_esferica = ProyeccionEsferica(al1, 600, 600)
    #pintaI(proyeccion_esferica, 1, "Proyeccion esférica. f=600. s=600.", "VC Proyecto - BurtAdelson")
    #input("Pulsa 'Enter' para continuar\n")

    # Ejemplo para probar un mosaico de yosemite
    #yosPan = mosaico_nBA(yosProy)
    #m1 = mosaicoBA(yosProy[1],yosProy[2])
    #m2 = mosaicoBA(yosProy[0],m1)
    #yosPan = mosaicoBA(m2,yosProy[3])
    #pintaI(yosPan, 1, "Mosaico de Yosemite.", "VC Proyecto - BurtAdelson")
    #input("Pulsa 'Enter' para continuar\n")

    # Ejemplo para probar un mosaico de la alhambra
    alPan = mosaico_nBA(alProy)
    pintaI(alPan, 1, "Mosaico de la Alhambra.", "VC Proyecto - BurtAdelson")

    # Ejemplo para probar un mosaico de la alhambra 2
    #alhamPan = mosaico_nBA(alhamProy)
    #pintaI(alhamPan, 1, "Mosaico de la Alhambra.", "VC Proyecto - BurtAdelson")
    """
    pyramid = laplacian_pyramid(al[0], 4)
    res1 = RestaurarLaplaciana_nuevo(pyramid)
    pintaI(res1)
    res2 = RestaurarLaplaciana(pyramid)
    pintaI(res2)
    """
