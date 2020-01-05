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


########################################
###   FUNCIONES DE OTRAS PRÁCTICAS   ###
########################################

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

    #cv2.imshow(image_title, image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

####################
###   PROYECTO   ###
####################

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


def correspondencias(imagen1, imagen2, criterio, elementos):
    # Creamos el detector-descriptor SIFT
    sift = cv2.xfeatures2d.SIFT_create()

    # Obtenemos los keypoints y los descriptores de las dos imagenes
    keypoints1, descriptores1 = sift.detectAndCompute(imagen1, None)
    keypoints2, descriptores2 = sift.detectAndCompute(imagen2, None)

    # Si hemos escogido el criterio 1, utilizamos BruteForce+CrossCheck
    if criterio == 1:
        # Creamos el objeto BFMatcher con CrossCheck
        bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)

        # Calculamos las correspondencias con los descriptores de las imagenes
        correspondencias = bf.match(descriptores1, descriptores2)

        # De todas las correspondencias obtenidas, escogemos n elementos
        # aleatorios para mostrarlos
        elem_correspondencias = sample(range(len(correspondencias)), elementos)

        seleccionados = []
        for i in range(len(elem_correspondencias)):
            seleccionados.append(correspondencias[i])

        # Mostramos los n elementos aleatorios
        imagen_final = cv2.drawMatches(img1=imagen1, keypoints1=keypoints1,
                                       img2=imagen2, keypoints2=keypoints2,
                                       matches1to2=seleccionados, outImg=None)

    # Si hemos escogido el criterio 2, utilizamos Lowe-Average-2NN
    elif criterio == 2:
        # Creamos el objeto BFMatcher sin CrossCheck
        bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)

        # Calculamos las correspondencias con los descriptores de las imagenes
        # y con k=2
        correspondencias = bf.knnMatch(descriptores1, descriptores2, k=2)

        # De todas las correspondencias obtenidas, escogemos n elementos
        # aleatorios para mostrarlos
        elem_correspondencias = sample(range(len(correspondencias)), elementos)

        seleccionados = []
        for i in elem_correspondencias:
            seleccionados.append(correspondencias[i])

        # Mostramos los n elementos aleatorios
        imagen_final = cv2.drawMatchesKnn(img1=imagen1, keypoints1=keypoints1,
                                          img2=imagen2, keypoints2=keypoints2,
                                          matches1to2=seleccionados, outImg=None)

    return (correspondencias, keypoints1, keypoints2, imagen_final)

#%%

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

# Funcion que realiza el proceso de pirámide Gaussiana
def PiramideGaussiana(imagen, niveles=8):
    # Guardamos la imagen original
    piramide = [imagen]
    actual = imagen
    # Recorremos las imágenes
    for i in range(niveles):
        # Establecemos el formato adecuado para OpenCV
        actual = np.uint8(actual)
        # Realizamos la convolucion y submuestreo
        actual = cv2.pyrDown(actual)
        # Recuperamos el formato
        actual = np.uint32(actual)
        # Guardamos la imagen
        piramide.append(actual)
    return piramide


def PiramideLaplaciana(gaussiana):
    piramide = []
    for i in range(len(gaussiana) - 1):
        # Cogemos los operandos
        actual = gaussiana[i]
        siguiente = gaussiana[i + 1]
        # Transformamos al formato adecuado para el upsample
        siguiente = np.float32(siguiente)
        # Realizamos la convolucion y aumento
        aumento = convolucionSeparable(siguiente, actual, False)
        # Recuperamos el formato
        siguiente = np.uint32(siguiente)
        # Realizamos la diferencia para obtener el nivel de la laplaciana
        laplacian = actual - aumento
        piramide.append(laplacian)

    # Guardamos el ultimo nivel de la gaussiana que es el primer nivel
    # de la laplaciana
    piramide.append(gaussiana[-1])
    return piramide


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


def Mezcla(Laplaciana1, Laplaciana2):
    Laplaciana_final = []
    for i in range(len(Laplaciana1)):
        #left = Laplaciana1[i]
        #right = Laplaciana2[i]
        nivel = np.zeros(Laplaciana1[i].shape, Laplaciana1[i].dtype)
        mitad = Laplaciana1[i].shape[1] // 2
        nivel[:, :mitad, ...] = Laplaciana1[i][:, :mitad, ...]
        nivel[:, -mitad:, ...] = Laplaciana2[i][:, -mitad:, ...]
        if Laplaciana1[i].shape[1] % 2 == 1:
            # Numero impar de columnas
            nivel[:, mitad, ...] = (Laplaciana1[i][:, mitad, ...] + Laplaciana2[i][:, mitad, ...])/2

        Laplaciana_final.append(nivel)
    return Laplaciana_final

def BurtAdelson(imagen1, imagen2):
    # Ajustamos las imágenes a formato uint32 y al mismo tamaño
    imagen1, imagen2 = AjustarImagenes(imagen1, imagen2)
    # Calculamos las pirámides Gaussianas
    gaussiana1 = PiramideGaussiana(imagen1)
    gaussiana2 = PiramideGaussiana(imagen2)
    # Calculamos las pirámides Laplacianas
    laplaciana1 = PiramideLaplaciana(gaussiana1)
    laplaciana2 = PiramideLaplaciana(gaussiana2)

    # Calculamos la pirámide Laplaciana combinada
    laplaciana_mezcla = Mezcla(laplaciana1, laplaciana2)
    img_restaurada = RestaurarLaplaciana(laplaciana_mezcla)

    # Normalizamos los valores que salgan del rango [0,255]
    np.clip(img_restaurada, 0, 255, out=img_restaurada)
    # Transformamos el formato de la imagen para la visualización
    img_restaurada = np.uint8(img_restaurada)
    plt.imsave('orapple.jpg', cv2.cvtColor(img_restaurada, cv2.COLOR_BGR2RGB))
    return img_restaurada


#%%

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
    # Calculamos la homografía que sítua a la imagen 2 en el centro del canvas
    homografia=np.matrix([[1,0,0],[0,1,0],[0,0,1]],
                         dtype=float)

    # Calculamos las correspondencias y keypoints de la imagen 2 (derecha) a la imagen 1 (izquierda)
    correspondencias_x, keypoints1, keypoints2 = correspondencias(imagen1, imagen2, 1, 20)[0:3]

    # Obtenemos los puntos de fuente y destino de los objetos DMatch y con un reshape
    # tal y como se muestra en https://docs.opencv.org/3.3.1/d1/de0/tutorial_py_feature_homography.html
    puntos_dst = np.float32([keypoints1[m.queryIdx].pt for m in correspondencias_x]).reshape(-1, 1, 2)
    puntos_src = np.float32([keypoints2[m.trainIdx].pt for m in correspondencias_x]).reshape(-1, 1, 2)

    # Calculamos la homografía de la imagen 2 a la imagen 1
    homografia_x = cv2.findHomography(puntos_src, puntos_dst, cv2.RANSAC, 1)[0]

    # Tamaño del canvas
    size = (imagen1.shape[1] + imagen2.shape[1], imagen1.shape[0])

    # Aplicamos la homografia que situa a la imagen 2 en el centro del canvas
    area1 = cv2.warpPerspective(imagen1, homografia, size)
    # Aplicamos la homografia que situa a la imagen 1 con respecto a la imagen 2
    area2 = cv2.warpPerspective(imagen2, homografia*homografia_x, dsize=size)

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
    mosaico_der = mosaicoBA(lista_imagenes[centro], lista_imagenes[centro+1])
    mosaico_izq = mosaicoBA(lista_imagenes[centro-1], lista_imagenes[centro])
    for n in range(centro, len(lista_imagenes)):
        mosaico_der = mosaicoBA(mosaico_der, lista_imagenes[n])
    for n in range(centro, 0, -1):
        mosaico_izq = mosaicoBA(mosaico_izq, lista_imagenes[n])

    mosaico_final = mosaicoBA(mosaico_izq, mosaico_der)

    return mosaico_final

def limpiarImagen1(imagen):
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

    return imagen

def mosaico_n(imagenes):
    # Calculamos la posicion de la imagen central
    centro = len(imagenes)//2

    # Calculamos la homografía que sítua a la imagen central en el centro del canvas
    homografia=np.matrix([[1,0,imagenes[0].shape[1]*centro],[0,1,imagenes[0].shape[0]*centro],[0,0,1]], dtype=float)

    # Calculamos las homografias hacia la izquierda de la central
    homografias_izq = []
    homografia_previa = homografia
    for i in range(centro, 0, -1):
        correspondencia, keypoints1, keypoints2 = correspondencias(imagenes[i], imagenes[i-1], 1, 1)[0:3]

        puntos_dst = np.float32([keypoints1[m.queryIdx].pt for m in correspondencia]).reshape(-1, 1, 2)
        puntos_src = np.float32([keypoints2[m.trainIdx].pt for m in correspondencia]).reshape(-1, 1, 2)

        homografia_i = cv2.findHomography(puntos_src, puntos_dst, cv2.RANSAC, 1)[0]

        homografia_previa = homografia_previa* homografia_i
        homografias_izq.append(homografia_previa)

    # Calculamos las homografias hacia la derecha de la central
    homografias_der = []
    homografia_previa = homografia
    for i in range(centro, len(imagenes) -1):
        correspondencia, keypoints1, keypoints2 = correspondencias(imagenes[i], imagenes[i+1], 1, 1)[0:3]

        puntos_dst = np.float32([keypoints1[m.queryIdx].pt for m in correspondencia]).reshape(-1, 1, 2)
        puntos_src = np.float32([keypoints2[m.trainIdx].pt for m in correspondencia]).reshape(-1, 1, 2)

        homografia_i = cv2.findHomography(puntos_src, puntos_dst, cv2.RANSAC, 1)[0]

        homografia_previa = homografia_previa* homografia_i
        homografias_der.append(homografia_previa)

    # Tamaño del canvas
    size_x = 0
    size_y = 0
    for i in imagenes:
        size_x += i.shape[1]
        size_y += i.shape[0]

    size = (size_x, size_y)

    # Aplicamos la homografia que situa a la imagen 2 en el centro del canvas
    mosaico = cv2.warpPerspective(imagenes[centro], homografia, size)

    # Aplicamos hacia la izquierda las homografias correspondientes
    indice_homografia = 0
    for i in range(centro, 0, -1):
        mosaico = cv2.warpPerspective(imagenes[i-1], homografias_izq[indice_homografia],
                                      borderMode=cv2.BORDER_TRANSPARENT, dst=mosaico,
                                      dsize=size)
        indice_homografia += 1

    indice_homografia = 0
    for i in range(centro, len(imagenes) -1):
        mosaico = cv2.warpPerspective(imagenes[i+1], homografias_der[indice_homografia],
                                      borderMode=cv2.BORDER_TRANSPARENT, dst=mosaico,
                                      dsize=size)
        indice_homografia += 1

    # Quitamos los bordes negros de la imagen
    mosaico = limpiarImagen1(mosaico)

    return mosaico

#######################
###       MAIN      ###
#######################

""" Programa principal. """
if __name__ == "__main__":
    # Leemos las imágenes que necesitamos
    al1 = leer_imagen("imagenes/al1.png", 1)
    yos = [leer_imagen("imagenes/yosemite1.jpg", 1),
           leer_imagen("imagenes/yosemite2.jpg", 1),
           leer_imagen("imagenes/yosemite3.jpg", 1)]
    al = [leer_imagen("imagenes/al1.png", 1),
          leer_imagen("imagenes/al2.png", 1),
          leer_imagen("imagenes/al3.png", 1)]
    alham = [leer_imagen("imagenes/alham1.png", 1),
             leer_imagen("imagenes/alham2.png", 1),
             leer_imagen("imagenes/alham3.png", 1),
             leer_imagen("imagenes/alham4.png", 1),
             leer_imagen("imagenes/alham5.png", 1)]
    yosProy = listaProyeccionesCilindricas(yos, 700, 700, "Yosemite")
    alProy = listaProyeccionesCilindricas(al, 700, 700, "Alhambra 1")
    alhamProy = listaProyeccionesCilindricas(alham, 900, 900, "Alhambra 2")

    # Ejemplo para probar proyecciones cilíndricas
    proyeccion_cilindrica = ProyeccionCilindrica(al1, 600, 600)
    pintaI(proyeccion_cilindrica, 1, "Proyeccion cilindrica. f=600. s=600.", "VC Proyecto - BurtAdelson")
    #input("Pulsa 'Enter' para continuar\n")

    # Ejemplo para probar proyecciones esféricas
    proyeccion_esferica = ProyeccionEsferica(al1, 600, 600)
    pintaI(proyeccion_esferica, 1, "Proyeccion esférica. f=600. s=600.", "VC Proyecto - BurtAdelson")
    #input("Pulsa 'Enter' para continuar\n")

    # Ejemplo para probar un mosaico de yosemite
    panorama = mosaico_nBA(yosProy)
    pintaI(panorama, 1, "Mosaico de Yosemite.", "VC Proyecto - BurtAdelson")
    #input("Pulsa 'Enter' para continuar\n")

    # Ejemplo para probar un mosaico de la playa
    panorama = mosaico_nBA(alham)
    pintaI(panorama, 1, "Mosaico de la Alhambra.", "VC Proyecto - BurtAdelson")

    """ BORRAR CUANDO ESTEMOS SEGUROS
    # Ejemplo para probar un mosaico de yosemite
    panorama = mosaico_nBA((ProyeccionCilindrica(cv2.imread("imagenes/yosemite1.jpg", 1), 700, 700),
                            ProyeccionCilindrica(cv2.imread("imagenes/yosemite2.jpg", 1), 700, 700),
                            ProyeccionCilindrica(cv2.imread("imagenes/yosemite3.jpg", 1), 700, 700)))

    #pintaI(panorama, 1, "Mosaico de Yosemite.", "VC Proyecto - BurtAdelson")
    representar("Mosaico de Yosemite", panorama, 1, 1)
    #input("Pulsa 'Enter' para continuar\n")

    # Ejemplo para probar un mosaico de la playa
    panorama = mosaico_nBA((ProyeccionCilindrica(cv2.imread("imagenes/alham1.png", 1), 900, 900),
                            ProyeccionCilindrica(cv2.imread("imagenes/alham2.png", 1), 900, 900),
                            ProyeccionCilindrica(cv2.imread("imagenes/alham3.png", 1), 900, 900),
                            ProyeccionCilindrica(cv2.imread("imagenes/alham4.png", 1), 900, 900),
                            ProyeccionCilindrica(cv2.imread("imagenes/alham5.png", 1), 900, 900)))

    #pintaI(panorama, 1, "Mosaico de la Alhambra.", "VC Proyecto - BurtAdelson")
    representar("Mosaico de la playa de la Herradura", panorama, 1, 1)
    """
