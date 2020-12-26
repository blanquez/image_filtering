# -*- coding: utf-8 -*-
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

pltfigure = 1

# Función que lee una imagen, en color o en blanco y negro
def leeImagen(filename,flagColor,size=100):
    image = cv.imread(filename,flagColor)
    if(len(image.shape)>2):
        image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
    
    return image

# Función que pinta varias imágenes en una sóla con sus títulos
def juntaMI(vim,titulo,size=100):
    global pltfigure
    
    imgFinal = plt.figure(pltfigure,dpi=size)
    axes=[]
    
    for i in range(len(vim)):        
        axes.append(imgFinal.add_subplot(1,len(vim),i+1))
        axes[-1].set_title(titulo[i])
        axes[-1].set_aspect('auto')
        #plt.axes().set_aspect('equal')
        plt.axis('off')
        plt.imshow(vim[i],cmap='gray')
    
    plt.show()
    pltfigure = pltfigure + 1

# Función que pinta una pirámide de imágenes
def printPiramide(pyramid,titulo,size=100):
    
    if len(img.shape)<2:
        rows, cols, dim = pyramid[0].shape
        composite_image = np.zeros((rows, cols + cols // 2 + 1, 3), dtype=np.double)
        composite_image[:rows, :cols, :] = pyramid[0]
        
    else:
        rows, cols = pyramid[0].shape
        composite_image = np.zeros((rows, cols + cols // 2 + 1), dtype=np.double)
        composite_image[:rows, :cols] = pyramid[0]

    i_row = 0
    for p in pyramid[1:]:
        n_rows, n_cols = p.shape[:2]
        composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
        i_row += n_rows

    fig, ax = plt.subplots(dpi=size)
    ax.imshow(composite_image, cmap='gray')
    plt.axis('off')
    plt.title(titulo)
    plt.show()

# Función que añade padding
def addPadding(img,kx,ky,replic=False):
    # Calculamos las nuevas medidas de la imagen
    x, y = img.shape
    new_x = x + 2*kx
    new_y = y + 2*ky
    
    # Creamos una matriz de 0s (negro) con las nuevas medidas
    out_img = np.zeros(((new_x),(new_y)))
    
    # Calculamos la esquina superior derecha de la imagen
    # original en matriz de 0s
    xx = (new_x - x) // 2
    yy = (new_y - y) // 2
    
    # Introducimos la imagen desde la esquina calculada y devolvemos
    # la matriz con la imagen incrustada en el centro
    out_img[xx:xx+x, yy:yy+y] = img
    
    # Si queremos bordes replicados tenemos que replicarlos
    if replic:
        # Replicamos las filas
        out_img[:kx,ky:-ky] = np.flip(img[:kx,:])
        out_img[-kx:,ky:-ky] = np.flip(img[-kx:,:])
        
        # Replicamos las filas
        out_img[:,:ky] = np.flip(out_img[:,ky:ky*2])
        out_img[:,-ky:] = np.flip(out_img[:,-ky*2:-ky])
    
    return out_img

# Función que elimina el padding
def removePadding(img,kx,ky):
    # Calculamos las medidas de la imagen sin padding
    x, y = img.shape
    new_x = x - 2*kx
    new_y = y - 2*ky
    
    # Creamos una matriz de las nuevas medidas
    out_img = np.zeros(((new_x),(new_y)))
    
    # Calculamos la esquina superior de la imagen sin padding
    xx = (x - new_x) // 2
    yy = (y - new_y) // 2
    
    # Copiamos la imagen sin padding a partir de la esquina calculada
    # en la matriz creada anteriormente, que será devuelta
    out_img = img[xx:xx+new_x, yy:yy+new_y]
    
    return out_img

# Función gaussiana
def f(x, sigma):
    return np.exp(-(x**2)/(2*sigma**2))

# Primera derivada de la función gaussiana 
def df(x, sigma):
    return -x * (np.exp(-(x**2) / (2*sigma**2)) / (sigma**2))

# Segunda derivada de la función gaussiana
def ddf(x, sigma):
    return (x**2-sigma**2) * (np.exp(-(x**2) / (2*sigma**2)) / sigma**4)

# Devuelve una máscara gaussiana(o primera o segunda derivada) según un sigma o tamaño dado
def gaussian_kernel(sigma=None,T=None,der=0):
    # Si no se selecciona la derivada correctamente devuelve un error
    if(der<0 or der>2):
        print("Error: con der se debe elegir entre la gaussiana(0) o alguna de sus derivadas(1,2)")
        return
    # Si se pasan los dos parámetros da error, ya que uno se calcula a partir del otro
    if((sigma != None and T != None) or (sigma == None and T == None)):
        print("Error: no se puede pasar ambos valores o ninguno, si no solo uno de ellos y automáticamente se calculará el otro")
        return
    elif(sigma == None):
        sigma = (T - 1) / 6
        k = int(3 * sigma)
    else:
        k = int(np.ceil(3 * sigma))
        T = 2 * k + 1
        
    mask = []
    
    # Según la función seleccionada se utiliza una u otra función
    # der=0: Filtro Gaussiano
    # der=1: Primera derivada del filtro Gaussiano
    # der=2: Segunda derivada del filtro Gaussiano
    if(der == 0):
        for i in range(-k,k+1):
            mask.append(f(i,sigma))
        mask = mask / np.sum(mask)
    elif(der == 1):
        for i in range(-k,k+1):
            mask.append(df(i,sigma))
        mask = np.multiply(mask, sigma)
    else:
        for i in range(-k,k+1):
            mask.append(ddf(i,sigma))
        mask = np.multiply(mask, sigma ** 2)
    
    return np.array(mask)

def convolucion(img,maskx,masky):
    
    # Como es convolucion, hay que invertir la máscara
    maskx = np.flip(maskx)
    masky = np.flip(masky)
    img_out = img.copy()
    
    kx = int((len(maskx)-1) / 2)
    ky = int((len(masky)-1) / 2)
    
    # Aplicamos la máscara por filas
    for i in range(ky, img_out.shape[0]-ky):
        for j in range(kx,img_out.shape[1]-kx):
            img_out[i, j] = np.sum(np.multiply(img[i, j-kx:j+kx+1], maskx))
    
    # Aplicamos la máscara por columnas
    
    img_out2 = img_out.copy()
    
    for i in range(ky, img_out.shape[0]-ky):
        for j in range(kx,img_out.shape[1]-kx):
            img_out2[i, j] = np.sum(np.multiply(img_out[i-ky:i+ky+1, j], masky))
               
    return img_out2

# Función que hibrida imágenes
def hybrid_imgs(img1,img2,sig1,sig2,img2_factor,tit):
    # Creamos las máscaras Gaussianas
    gauss1 = gaussian_kernel(sigma=sig1)
    gauss2 = gaussian_kernel(sigma=sig2)
    
    img1 = convolucion(img1, gauss1, gauss1)
    img2 = np.multiply(img2, img2_factor) - convolucion(img2, gauss2, gauss2)
    
    hybrid = cv.addWeighted(img1.astype(int),0.5,img2.astype(int),0.5,0)
    
    # Imprimimos las tres imágenes
    
    juntaMI([img1,hybrid,img2], ["Baja frecuencia","Híbrida","Alta frecuencia"])
    
    # Calculamos la máscara gaussiana
    
    gauss = gaussian_kernel(sigma=1)
    
    # Calculamos los 4 niveles de la pirámide gaussiana con bordes replicados
    
    k = int(len(gauss)-1/2)
    
    # Nivel 1
    print("Calculando nivel 1...")
    
    level1 = addPadding(hybrid, k, k, True)
    level1 = convolucion(level1, gauss, gauss)
    #level1 = cv.GaussianBlur(level1,(k,k),sig,sig)
    level1 = removePadding(level1, k, k)
    level1 = level1[::2, ::2]
    
    # Nivel 2
    print("Calculando nivel 2...")
    
    level2 = addPadding(level1, k, k, True)
    level2 = convolucion(level2, gauss, gauss)
    #level2 = cv.GaussianBlur(level2,(k,k),sig,sig)
    level2 = removePadding(level2, k, k)
    level2 = level2[::2, ::2]
    
    # Nivel 3
    print("Calculando nivel 3...")
    
    level3 = addPadding(level2, k, k, True)
    level3 = convolucion(level3, gauss, gauss)
    #level3 = cv.GaussianBlur(level3,(k,k),sig,sig)
    level3 = removePadding(level3, k, k)
    level3 = level3[::2, ::2]
    
    # Nivel 4
    print("Calculando nivel 4...")
    
    level4 = addPadding(level3, k, k, True)
    level4 = convolucion(level4, gauss, gauss)
    #level4 = cv.GaussianBlur(level4,(k,k),sig,sig)
    level4 = removePadding(level4, k, k)
    level4 = level4[::2, ::2]
    
    # Pasamos la matrices a enteros
    
    cv.normalize(level1,np.float32(level1),255.0,0.0,cv.NORM_MINMAX)
    level1 = level1.astype(int)
    cv.normalize(level2,np.float32(level2),255.0,0.0,cv.NORM_MINMAX)
    level2 = level2.astype(int)
    cv.normalize(level3,np.float32(level3),255.0,0.0,cv.NORM_MINMAX)
    level3 = level3.astype(int)
    cv.normalize(level4,np.float32(level4),255.0,0.0,cv.NORM_MINMAX)
    level4 = level4.astype(int)
    
    # Imprimimos la pirámide gaussiana
    
    print("Pirámide de 4 niveles")
    
    imgs=[hybrid,level1,level2,level3,level4]
    printPiramide(imgs,tit,150)

#########################################################################

# Apartados
# 1A)

def ej1A(sig):
    print("Apartado 1A\n")
    
    # Mostramos las funciones para comprobar que son correctas
    
    print("Gráfica de la Gaussiana y sus derivadas calculadas según las funciones propias")
    print("sigma = ",sig)
    
    x = np.linspace(-5,5)
    plt.plot(f(x,sig),label="f")
    plt.plot(df(x,sig), label="f'")
    plt.plot(ddf(x,sig), label="f''")
    plt.title("Función Gaussiana y sus derivadas")
    plt.legend()
    plt.show()

    input("\n--- Pulse cualquier tecla para continuar ---\n")
    
    print("Máscaras calculadas a partir de las funciones anteriores")
    print("sigma = ",sig)

    # Calculamos la máscara gaussiana 1D y sus derivadas
    gauss = gaussian_kernel(sigma=sig)
    dgauss = gaussian_kernel(sigma=sig,der=1)
    ddgauss = gaussian_kernel(sigma=sig,der=2)
    
    # Mostramos las máscaras
    x = [-3,-2,-1,0,1,2,3]
    plt.plot(x,gauss, label="Gaussiana")
    plt.plot(x,dgauss, label="1a de la Gaussiana")
    plt.plot(x,ddgauss, label="2a de la Gaussiana")
    plt.legend()
    plt.title("Máscaras creadas")
    plt.show()
    
    input("\n--- Pulse cualquier tecla para continuar ---\n")

# 1B)

def ej1B(img,sig):
    print("Apartado 1B\n")
    
    # Calculamos la máscara
    mask = gaussian_kernel(sigma=sig)
    
    # Añadimos el padding a la imagen
    k = len(mask)
    prueba = addPadding(img, k, k)
    pimg = addPadding(img, k, k, True)
    
    # Visualizamos el padding, tanto bordes negros como replicados
    
    print("Ambos tipos de padding")
    
    imgs=[img,prueba,pimg]
    tit=["Original","Bordes negros","Bordes replicados"]
    juntaMI(imgs,tit,150)
    
    input("\n--- Pulse cualquier tecla para continuar ---\n")
    
    print("Calculando convolución...")
    
    # Calculamos la convolución de la imagen
    gimg = convolucion(pimg,mask,mask)
    cvimg = cv.GaussianBlur(img,(k,k),sig,sig)
    
    # Eliminamos el padding a la imagen
    gimg = removePadding(gimg, k, k)
    
    # Normalizamos
    cv.normalize(gimg,gimg,255.0,0.0,cv.NORM_MINMAX)
    cv.normalize(cvimg,cvimg,255.0,0.0,cv.NORM_MINMAX)
    
    gimg = gimg.astype(int)
    cvimg = cvimg.astype(int)
    
    print("Comparación entre función propia y OpenCV")
    
    imgs=[img,gimg,cvimg]
    tit=["Original","Función propia","Gaussian Blur"]
    juntaMI(imgs,tit,200)
    
    #print("\nMatriz diferencia\n")
    
    #print(np.abs(gimg-cvimg))
    
    input("\n--- Pulse cualquier tecla para continuar ---\n")

# 1C)

def ej1C():
    print("Apartado 1C\n")
    
    # Probando la primera derivada con varios T(y por tanto varios sigmas)
    
    print("Enumeración de máscaras primera derivada de la Gaussiana con distintos tamaños y comparación con getDerivKernel")
    
    x = [0,1,2]
    dx_mask, dy_mask = cv.getDerivKernels(1,1,ksize=3,normalize=True)
    plt.scatter(x,gaussian_kernel(T=3,der=1), label="propia")
    plt.scatter(x,dx_mask, label="getDerivKernels")
    plt.legend()
    plt.title("1a derivada, T=3 -> sigma=0.33")
    plt.show()
    
    input("\n--- Pulse cualquier tecla para continuar ---\n")
    
    x = [0,1,2,3,4]
    dx_mask, dy_mask = cv.getDerivKernels(1,1,ksize=5,normalize=True)
    plt.scatter(x,gaussian_kernel(T=5,der=1), label="propia")
    plt.scatter(x,dx_mask, label="getDerivKernels")
    plt.legend()
    plt.title("1a derivada, T=5 -> sigma=0.66")
    plt.show()
    
    input("\n--- Pulse cualquier tecla para continuar ---\n")
    
    x = [0,1,2,3,4,5,6]
    dx_mask, dy_mask = cv.getDerivKernels(1,1,ksize=7,normalize=True)
    plt.scatter(x,gaussian_kernel(T=7,der=1), label="propia")
    plt.scatter(x,dx_mask, label="getDerivKernels")
    plt.legend()
    plt.title("1a derivada, T=7 -> sigma=1")
    plt.show()
    
    input("\n--- Pulse cualquier tecla para continuar ---\n")
    
    x = [0,1,2,3,4,5,6,7,8]
    dx_mask, dy_mask = cv.getDerivKernels(1,1,ksize=9,normalize=True)
    plt.scatter(x,gaussian_kernel(T=9,der=1), label="propia")
    plt.scatter(x,dx_mask, label="getDerivKernels")
    plt.legend()
    plt.title("1a derivada, T=9 -> sigma=1.33")
    plt.show()
    
    input("\n--- Pulse cualquier tecla para continuar ---\n")
    
    x = [0,1,2,3,4,5,6,7,8,9,10]
    dx_mask, dy_mask = cv.getDerivKernels(1,1,ksize=11,normalize=True)
    plt.scatter(x,gaussian_kernel(T=11,der=1), label="propia")
    plt.scatter(x,dx_mask, label="getDerivKernels")
    plt.legend()
    plt.title("1a derivada, T=11 -> sigma=1.66")
    plt.show()
    
    input("\n--- Pulse cualquier tecla para continuar ---\n")
    
    print("Aproximación usando un plot")
    
    # Aproximamos las curvas a partir de la máscara más grande que hemos probado
    
    plt.plot(x,gaussian_kernel(T=11,der=1), label="propia")
    plt.plot(x,dx_mask, label="getDerivKernels")
    plt.legend()
    plt.title("1a derivada, Tendencia aproximada")
    plt.show()
    
    input("\n--- Pulse cualquier tecla para continuar ---\n")
    
    print("Enumeración de máscaras segunda derivada de la Gaussiana con distintos tamaños y comparación con getDerivKernel")
    
    # Probando la segunda derivada con varios T(y por tanto varios sigmas)
    
    x = [0,1,2]
    dx_mask, dy_mask = cv.getDerivKernels(2,2,ksize=3,normalize=True)
    plt.scatter(x,gaussian_kernel(T=3,der=2), label="propia")
    plt.scatter(x,dx_mask, label="getDerivKernels")
    plt.legend()
    plt.title("2a derivada, T=3 -> sigma=0.33")
    plt.show()
    
    input("\n--- Pulse cualquier tecla para continuar ---\n")
    
    x = [0,1,2,3,4]
    dx_mask, dy_mask = cv.getDerivKernels(2,2,ksize=5,normalize=True)
    plt.scatter(x,gaussian_kernel(T=5,der=2), label="propia")
    plt.scatter(x,dx_mask, label="getDerivKernels")
    plt.legend()
    plt.title("2a derivada, T=5 -> sigma=0.66")
    plt.show()
    
    input("\n--- Pulse cualquier tecla para continuar ---\n")
    
    x = [0,1,2,3,4,5,6]
    dx_mask, dy_mask = cv.getDerivKernels(2,2,ksize=7,normalize=True)
    plt.scatter(x,gaussian_kernel(T=7,der=2), label="propia")
    plt.scatter(x,dx_mask, label="getDerivKernels")
    plt.legend()
    plt.title("2a derivada, T=7 -> sigma=1")
    plt.show()
    
    input("\n--- Pulse cualquier tecla para continuar ---\n")
    
    x = [0,1,2,3,4,5,6,7,8]
    dx_mask, dy_mask = cv.getDerivKernels(2,2,ksize=9,normalize=True)
    plt.scatter(x,gaussian_kernel(T=9,der=2), label="propia")
    plt.scatter(x,dx_mask, label="getDerivKernels")
    plt.legend()
    plt.title("2a derivada, T=9 -> sigma=1.33")
    plt.show()
    
    input("\n--- Pulse cualquier tecla para continuar ---\n")
    
    x = [0,1,2,3,4,5,6,7,8,9,10]
    dx_mask, dy_mask = cv.getDerivKernels(2,2,ksize=11,normalize=True)
    plt.scatter(x,gaussian_kernel(T=11,der=2), label="propia")
    plt.scatter(x,dx_mask, label="getDerivKernels")
    plt.legend()
    plt.title("2a derivada, T=11 -> sigma=1.66")
    plt.show()
    
    input("\n--- Pulse cualquier tecla para continuar ---\n")
    
    # Aproximamos las curvas a partir de la máscara más grande que hemos probado
    
    print("Aproximación usando plot")
    
    plt.plot(x,gaussian_kernel(T=11,der=2), label="propia")
    plt.plot(x,dx_mask, label="getDerivKernels")
    plt.legend()
    plt.title("2a derivada, Tendencia aproximada")
    plt.show()
    
    input("\n--- Pulse cualquier tecla para continuar ---\n")

# 1D)

def ej1D(img):
    print("Apartado 1D\n")

    laplacians = []
    titles = []
    
    # Sigma = 1 y bordes negros
    
    print("Calculando Laplaciana para sigma 1 y bordes negros...")
    
    # Calculamos las máscaras necesarias para la Laplaciana de la Gaussiana
    
    sig = 1
    
    gauss = gaussian_kernel(sigma=sig)
    kg = len(gauss)
    ddgauss = gaussian_kernel(sigma=sig,der=2)
    kdd = len(ddgauss)
    
    # Hacemos las convoluciones con las máscaras previamente calculadas
    
    gxx = addPadding(img, kdd, kg)
    gxx = convolucion(gxx,ddgauss,gauss)
    gxx = removePadding(gxx, kdd, kg)
    
    gyy = addPadding(img, kg, kdd)
    gyy = convolucion(gyy, gauss, ddgauss)
    gyy = removePadding(gyy, kg, kdd)
    
    # Calculamos la Laplaciana
    
    laplacian = gxx + gyy
    
    # Normalizamos
    
    cv.normalize(laplacian,laplacian,255.0,0.0,cv.NORM_MINMAX)
    laplacian = laplacian.astype(int)
    
    laplacians.append(laplacian)
    titles.append("S=1, B.Neg")
    
    # Sigma = 1 y bordes replicados
    
    print("Calculando Laplaciana para sigma 1 y bordes replicados...")
    
    # Hacemos las convoluciones con las máscaras previamente calculadas
    
    gxx = addPadding(img, kdd, kg, True)
    gxx = convolucion(gxx,ddgauss,gauss)
    gxx = removePadding(gxx, kdd, kg)
    
    gyy = addPadding(img, kg, kdd, True)
    gyy = convolucion(gyy, gauss, ddgauss)
    gyy = removePadding(gyy, kg, kdd)
    
    # Calculamos la Laplaciana
    
    laplacian = gxx + gyy
    
    cv.normalize(laplacian,laplacian,255.0,0.0,cv.NORM_MINMAX)
    laplacian = laplacian.astype(int)
    
    # Normalizamos
    
    laplacians.append(laplacian.astype(int))
    titles.append("S=1, B.Rep")
    
    # Sigma = 3 y bordes negros
    
    print("Calculando Laplaciana para sigma 3 y bordes negros...")
    
    # Calculamos las máscaras necesarias para la Laplaciana de la Gaussiana
    
    sig = 3
    
    gauss = gaussian_kernel(sigma=sig)
    kg = len(gauss)
    ddgauss = gaussian_kernel(sigma=sig,der=2)
    kdd = len(ddgauss)
    
    # Hacemos las convoluciones con las máscaras previamente calculadas
    
    gxx = addPadding(img, kdd, kg)
    gxx = convolucion(gxx,ddgauss,gauss)
    gxx = removePadding(gxx, kdd, kg)
    
    gyy = addPadding(img, kg, kdd)
    gyy = convolucion(gyy, gauss, ddgauss)
    gyy = removePadding(gyy, kg, kdd)
    
    # Calculamos la Laplaciana
    
    laplacian = gxx + gyy
    
    # Normalizamos
    
    cv.normalize(laplacian,laplacian,255.0,0.0,cv.NORM_MINMAX)
    laplacian = laplacian.astype(int)
    
    laplacians.append(laplacian.astype(int))
    titles.append("S=3, B.Neg")
    
    # Sigma = 3 y bordes replicados
    
    print("Calculando Laplaciana para sigma 3 y bordes replicados...")
    
    # Hacemos las convoluciones con las máscaras previamente calculadas
    
    gxx = addPadding(img, kdd, kg, True)
    gxx = convolucion(gxx,ddgauss,gauss)
    gxx = removePadding(gxx, kdd, kg)
    
    gyy = addPadding(img, kg, kdd, True)
    gyy = convolucion(gyy, gauss, ddgauss)
    gyy = removePadding(gyy, kg, kdd)
    
    # Calculamos la Laplaciana
    
    laplacian = gxx + gyy
    
    # Normalizamos
    
    cv.normalize(laplacian,laplacian,255.0,0.0,cv.NORM_MINMAX)
    laplacian = laplacian.astype(int)
    
    laplacians.append(laplacian.astype(int))
    titles.append("S=3, B.Rep")
    
    # Observamos los resultados
    
    print("Comparación de las Laplacianas en los distintos escenarios")
    juntaMI(laplacians, titles, 200)
    
    input("\n--- Pulse cualquier tecla para continuar ---\n")
    
    # Observamos las máscaras
    
    print("Máscara utilizada(sigma=",sig,")")
    mask = np.outer(ddgauss,gauss) + np.outer(gauss,ddgauss)
    
    ax = plt.axes(projection='3d')
    k = int(np.ceil(3*sig))
    X2, Y2 = np.meshgrid(range(-k,k+1),range(-k,k+1))
    ax.plot_surface(X2,Y2,mask,cmap='jet')
    plt.title("Máscara Laplaciana de la Gaussiana")
    plt.show()
    
    input("\n--- Pulse cualquier tecla para continuar ---\n")

# 2A)

def ej2A(img,sig):
    print("Apartado 2A\n")
    
    # Calculamos la máscara gaussiana
    
    gauss = gaussian_kernel(sigma=sig)
    
    # Calculamos los 4 niveles de la pirámide gaussiana con bordes replicados
    
    k = int(len(gauss)-1/2)
    
    # Nivel 1
    print("Calculando nivel 1...")
    
    level1 = addPadding(img, k, k, True)
    level1 = convolucion(level1, gauss, gauss)
    #level1 = cv.GaussianBlur(level1,(k,k),sig,sig)
    level1 = removePadding(level1, k, k)
    level1 = level1[::2, ::2]
    
    # Nivel 2
    print("Calculando nivel 2...")
    
    level2 = addPadding(level1, k, k, True)
    level2 = convolucion(level2, gauss, gauss)
    #level2 = cv.GaussianBlur(level2,(k,k),sig,sig)
    level2 = removePadding(level2, k, k)
    level2 = level2[::2, ::2]
    
    # Nivel 3
    print("Calculando nivel 3...")
    
    level3 = addPadding(level2, k, k, True)
    level3 = convolucion(level3, gauss, gauss)
    #level3 = cv.GaussianBlur(level3,(k,k),sig,sig)
    level3 = removePadding(level3, k, k)
    level3 = level3[::2, ::2]
    
    # Nivel 4
    print("Calculando nivel 4...")
    
    level4 = addPadding(level3, k, k, True)
    level4 = convolucion(level4, gauss, gauss)
    #level4 = cv.GaussianBlur(level4,(k,k),sig,sig)
    level4 = removePadding(level4, k, k)
    level4 = level4[::2, ::2]
    
    # Normalizamos
    
    cv.normalize(level1,np.float32(level1),255.0,0.0,cv.NORM_MINMAX)
    level1 = level1.astype(int)
    cv.normalize(level2,np.float32(level2),255.0,0.0,cv.NORM_MINMAX)
    level2 = level2.astype(int)
    cv.normalize(level3,np.float32(level3),255.0,0.0,cv.NORM_MINMAX)
    level3 = level3.astype(int)
    cv.normalize(level4,np.float32(level4),255.0,0.0,cv.NORM_MINMAX)
    level4 = level4.astype(int)
    
    # Imprimimos la pirámide gaussiana
    
    print("Pirámide Gaussiana de 4 niveles")
    
    imgs=[img,level1,level2,level3,level4]
    printPiramide(imgs,"Pirámide Gaussiana de 4 niveles",150)
    
    input("\n--- Pulse cualquier tecla para continuar ---\n")
   
# 2B)

def ej2B(img,sig):
    print("Apartado 2B")
    
    #Calculamos la máscara
    gauss = gaussian_kernel(sigma=sig)
    
    k = int(len(gauss)-1/2)
    
    # Ajustar los bordes para que no se pierda información al hacer el subsampling
    k = k + (16 - (k % 16))
    
    # Calculamos los 5 niveles de la pirámide gaussiana con bordes replicados
    
    # Nivel 1
    print("Calculando nivel 1...")
    
    level1 = addPadding(img, k, k, True)
    level1 = convolucion(level1, gauss, gauss)
    level1 = removePadding(level1, k, k)
    level1 = level1[::2, ::2]
    
    # Nivel 2
    print("Calculando nivel 2...")
    
    level2 = addPadding(level1, k, k, True)
    level2 = convolucion(level2, gauss, gauss)
    level2 = removePadding(level2, k, k)
    level2 = level2[::2, ::2]
    
    # Nivel 3
    print("Calculando nivel 3...")
    
    level3 = addPadding(level2, k, k, True)
    level3 = convolucion(level3, gauss, gauss)
    level3 = removePadding(level3, k, k)
    level3 = level3[::2, ::2]
    
    # Nivel 4
    print("Calculando nivel 4...")
    
    level4 = addPadding(level3, k, k, True)
    level4 = convolucion(level4, gauss, gauss)
    level4 = removePadding(level4, k, k)
    level4 = level4[::2, ::2]
    
    # Nivel 5
    print("Calculando nivel 5...")
    
    level5 = addPadding(level4, k, k, True)
    level5 = convolucion(level5, gauss, gauss)
    level5 = removePadding(level5, k, k)
    level5 = level5[::2, ::2]
    
    # Calculamos las Laplacianas de los primeros niveles
    print("Calculando pirámide Laplaciana")
    
    img = img - cv.resize(level1,(img.shape[1],img.shape[0]))
    level1 = level1 - cv.resize(level2,(level1.shape[1],level1.shape[0]))
    level2 = level2 - cv.resize(level3,(level2.shape[1],level2.shape[0]))
    level3 = level3 - cv.resize(level4,(level3.shape[1],level3.shape[0]))
    level4 = level4 - cv.resize(level5,(level4.shape[1],level4.shape[0]))
    
    # Pasamos la matrices a enteros
    
    cv.normalize(img,img,255.0,0.0,cv.NORM_MINMAX)
    img = img.astype(int)
    cv.normalize(level1,level1,255.0,0.0,cv.NORM_MINMAX)
    level1 = level1.astype(int)
    cv.normalize(level2,level2,255.0,0.0,cv.NORM_MINMAX)
    level2 = level2.astype(int)
    cv.normalize(level3,level3,255.0,0.0,cv.NORM_MINMAX)
    level3 = level3.astype(int)
    cv.normalize(level4,level4,255.0,0.0,cv.NORM_MINMAX)
    level4 = level4.astype(int)
    
    # Imprimimos la pirámide gaussiana
    
    print("Pirámide Gaussiana de 4 niveles")
    
    imgs=[img,level1,level2,level3,level4]
    printPiramide(imgs,"Pirámide Gaussiana de 4 niveles",150)
    
    input("\n--- Pulse cualquier tecla para continuar ---\n")
    
def ej3():
    print("Apartado 3")
    
    # Leemos las imágenes
    img11 = leeImagen("imagenes/marilyn.bmp", 0)
    img12 = leeImagen("imagenes/einstein.bmp", 0)
    img21 = leeImagen("imagenes/motorcycle.bmp", 0)
    img22 = leeImagen("imagenes/bicycle.bmp", 0)
    img31 = leeImagen("imagenes/fish.bmp", 0)
    img32 = leeImagen("imagenes/submarine.bmp", 0)
    
    # Calculamos y mostramos las imágenes híbridas

    print("Mari-stein")
    
    hybrid_imgs(img11,img12,2.7,2.2,1.8,"Mari-stein")
    
    input("\n--- Pulse cualquier tecla para continuar ---\n")
    
    print("What-cycle")
    
    hybrid_imgs(img21,img22,3,2,1.5,"What-cycle")
    
    input("\n--- Pulse cualquier tecla para continuar ---\n")
    
    print("Fish-marine")
    
    hybrid_imgs(img31,img32,4,1,2,"Fish-marine")
    
##################################################################

# Main
print("\nTRABAJO 1: FILTROS DE MÁSCARA")
    
# Leemos la imagen, en este caso el avión
img = leeImagen("imagenes/plane.bmp",0)

ej1A(1)
ej1B(img, 2)
ej1C()
ej1D(img)
ej2A(img, 1)
ej2B(img, 1)
ej3()