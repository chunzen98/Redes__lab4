# --------- IMPORTACIONES ---------
import scipy.io
from scipy import interpolate
from scipy.io import wavfile
from scipy.fftpack import fft,fftfreq,ifft

import numpy as np
import matplotlib.pyplot as plt

# ---------- BLOQUE FUNCIONES -------------
'''Recibe la senal y su frecuencia de muestreo, interpola los valores de la funcion, retorna la senal interpolada'''
def interpolacionAM(signal,frecuencia):
    tiempo=np.linspace(0, len(signal)/frecuencia, num=len(signal))
    interpolada = interpolate.interp1d(tiempo, signal)
    y2 = interpolada(tiempo)
    return y2

'''Recibe la senal, su frecuencia y el coeficiente de modulacion. Modula la senal en am segun la senal portadora
portadora y el coeficiente dado. Retorna la senal modulada'''    
def modulacionAM(signal,frecuencia,interpolada,coeficiente):
    #Obtencion de senales portadora y modulada
    largoInterpolada = len(interpolada)
    tiempoModulada = np.linspace(0,largoInterpolada/frecuencia,num=largoInterpolada)
    tiempo = np.linspace(0,len(signal)/frecuencia,num=len(signal))
    senalPortadora = np.cos(2*np.pi*frecuencia*3*tiempoModulada)
    senalModulada =  interpolada*signal*coeficiente

    #Graficos de senales
    plt.plot(tiempo,signal,'c')
    plt.xlabel('Tiempo(s)')
    plt.ylabel('Original')
    plt.title('Señal Original en el tiempo')
    
    tiempoPortadora = np.linspace(0,len(senalPortadora)/frecuencia,num=len(senalPortadora))
    plt.figure(2)
    plt.xlabel('Tiempo(s)')
    plt.ylabel('Portadora')
    plt.title('Portadora en el tiempo')
    plt.plot(tiempoPortadora,senalPortadora)

    plt.figure(3)
    plt.plot(tiempoModulada,senalModulada)
    plt.xlabel('Tiempo(s)')
    plt.ylabel('Modulada')
    plt.title('Señal Modulada en el tiempo')
    plt.show()

    return senalModulada



# ---------- BLOQUE PRINCIPAL -------------

#Obtencion de datos de senal
fs,audio = wavfile.read("handel.wav")

# Largo de la senal
largo = float(len(audio))

# Tiempo de la senal
t = largo / fs

interpolada = interpolacionAM(audio,fs)
modulada = modulacionAM(audio,fs,interpolada,1)

#Grafico de senales


