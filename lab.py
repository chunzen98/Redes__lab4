# --------- IMPORTACIONES ---------
import scipy.io
from scipy import interpolate
from scipy.io import wavfile
from scipy import signal as sig #Espectograma
from scipy.fftpack import fft,fftfreq
from scipy.signal import filtfilt

import numpy as np
import matplotlib.pyplot as plt

# ---------- BLOQUE FUNCIONES -------------

##Recibe la senal y su frecuencia de muestreo, interpola los valores de la funcion, retorna la senal interpolada
def interpolacionAM(signal,frecuencia):
    tiempo = np.linspace(0, len(signal)/frecuencia, num=len(signal))
    interpolada = interpolate.interp1d(tiempo, signal)
    tiempo2 = np.linspace(0, len(signal)/frecuencia, num=len(signal) * 4)
    y2 = interpolada(tiempo)
    return y2

##Recibe la senal, su frecuencia y el coeficiente de modulacion. Modula la senal en am segun la senal portadora
##portadora y el coeficiente dado. Retorna la senal modulada    
def modulacionAM(signal,frecuencia,interpolada,coeficiente):
    #Obtencion de senales portadora y modulada
    largoInterpolada = len(interpolada)
    fsAM = frecuencia * 3
    tiempoModulada = np.linspace(0,largoInterpolada/frecuencia,num=largoInterpolada)
    tiempo = np.linspace(0,len(signal)/frecuencia,num=len(signal))
    senalPortadora = np.cos(2*np.pi*fsAM*tiempoModulada)
    senalModulada =  interpolada*senalPortadora*coeficiente


    plt.figure("Mod. AM " + str(float(coeficiente) * 100) + "%")
    #Graficos de senales
    plt.subplot(311)
    plt.plot(tiempo,signal,"c")
    plt.xlabel("Tiempo(s)")
    plt.ylabel("Original")
    plt.title("Señal Original en el tiempo")
    
    tiempoPortadora = np.linspace(0,len(senalPortadora)/frecuencia,num=len(senalPortadora))

    plt.subplot(312)
    plt.xlabel("Tiempo(s)")
    plt.ylabel("Portadora")
    plt.title("Portadora en el tiempo")
    plt.plot(tiempoPortadora,senalPortadora)

    plt.subplot(313)
    plt.plot(tiempoModulada,senalModulada, "r")
    plt.xlabel("Tiempo(s)")
    plt.ylabel("Modulada")
    plt.title("Señal Modulada en el tiempo")

    plt.suptitle("Modulacion AM " + str(coeficiente * 100) + "%", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    return senalModulada, fsAM

#Recibe la senal, su frecuencia, la interpolada y el coeficiente de modulacion.
#Modula (FM) la senal segun el coeficiente dado. Retorna la senal modulada y la frecuencia usada
def modulacionFM(signal,frecuencia,interpolada,coeficiente):
    largoInterpolada = len(interpolada)
    fsFM = frecuencia * 3
    tiempoModulada = np.linspace(0,largoInterpolada/frecuencia,num=largoInterpolada)
    tiempo = np.linspace(0,len(signal)/frecuencia,num=len(signal))
    senalPortadora = np.cos(2*np.pi*fsFM*tiempoModulada)
    senalModulada = np.cos(2*np.pi*fsFM*tiempoModulada + (np.cumsum(interpolada)/frecuencia)*coeficiente)


    plt.figure("Mod. FM " + str(float(coeficiente) * 100) + "%")
    plt.subplot(311)
    plt.plot(tiempo,signal,"c")
    plt.xlabel("Tiempo(s)")
    plt.ylabel("Original")
    plt.title("Señal Original en el tiempo")

    plt.subplot(312)
    tiempoPortadora = np.linspace(0,len(senalPortadora)/frecuencia, num=len(senalPortadora))
    plt.plot(tiempoPortadora,senalPortadora)
    plt.xlabel("Tiempo(s)")
    plt.ylabel("Portadora")
    plt.title("Portadora en el tiempo ")
    
    
    plt.subplot(313)
    plt.plot(tiempoModulada,senalModulada,"r")
    plt.xlabel("Tiempo(s)")
    plt.ylabel("Modulada")
    plt.title("Señal Modulada en el tiempo fm")
    
    plt.suptitle("Modulacion FM " + str(coeficiente * 100) + "%", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    return senalModulada, fsFM

#Recibe la senal, la senal modulada en FM y AM, las frecuencias usadas en las modulaciones.
#Obtiene las transformadas de Fourier de las senales entregadas y las grafica.
#Retorna las transformadas de las senales.
def transFourier(audio, AM, FM, fs, fsAM, fsFM):

    largo = len(audio)
    
    dt = 1 / float(fs)
    dtAM = 1 / float(fsAM)
    dtFM = 1 / float(fsFM)
    
    yFourier = abs(fft(audio) / largo) #valor abs

    yAMFourier = abs(fft(AM) / largo) #valor abs

    yFMFourier = abs(fft(FM) / largo) #valor abs

    fsFourier = fftfreq(int(largo), dt)
    fsAMFourier = fftfreq(int(largo), dtAM)
    fsFMFourier = fftfreq(int(largo), dtFM)


    plt.figure("Transformada de fourier")

    plt.subplot(311)
    plt.plot(fsFourier, yFourier, "r")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Amplitud [dB]")
    plt.title("Transformada de Fourier original")
##    plt.text(2, 0.65, "Ancho de banda: " + str(max(fsFourier)))

    plt.subplot(312)
    plt.plot(fsAMFourier, yAMFourier, "b")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Amplitud [dB]")
    plt.title("Transformada de Fourier AM")
##    plt.text(2, 0.65, "Ancho de banda: " + str(max(fsAMFourier)))

    plt.subplot(313)
    plt.plot(fsFMFourier, yFMFourier, "c")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Amplitud [dB]")
    plt.title("Transformada de Fourier FM")
##    plt.text(2, 0.65, "Ancho de banda: " + str(max(fsFMFourier)))

    plt.suptitle("Transformada de Fourier de Modulacion", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    print("Ancho de banda Original: " + str(max(fsFourier)))
    print("Ancho de banda AM: " + str(max(fsAMFourier)))
    print("Ancho de banda FM: " + str(max(fsFMFourier)))
        
    return yFourier, yAMFourier, yFMFourier

#Recibe la senal modulada en AM, la frecuencia y la interpolada.
#Demodula la senal AM obteniendo la original.
#retorna la senal demodulada.
def demodulacionAM(signal,frecuencia,interpolada):
    #Obtencion de senales portadora y modulada
    largoInterpolada = len(interpolada)
    fsAM = frecuencia * 3
    tiempoModulada = np.linspace(0,largoInterpolada/frecuencia,num=largoInterpolada)
    tiempo = np.linspace(0,len(signal)/frecuencia,num=len(signal))
    senalPortadora = np.cos(2*np.pi*fsAM*tiempoModulada)
    senalDemodulada = signal * senalPortadora

    #Filtro pasa bajo
    b, a = sig.butter(3,4000,"low",fs=frecuencia)
    demodulada = filtfilt(b, a,audio)
    
    return senalDemodulada

# ---------- BLOQUE PRINCIPAL -------------

#Obtencion de datos de senal
fs,audio = wavfile.read("handel.wav")
largo = float(len(audio)) # Largo de la senal
t = largo / fs # Tiempo de la senal


#2)

interpolada = interpolacionAM(audio,fs)

#100%
moduladaAM, fsAM = modulacionAM(audio,fs,interpolada,1)
moduladaFM, fsFM = modulacionFM(audio,fs,interpolada,1)
#15%
moduladaAM15, fsAM15 = modulacionAM(audio,fs,interpolada,0.15)
moduladaFM15, fsFM15 = modulacionFM(audio,fs,interpolada,0.15)
#125%
moduladaAM125, fsAM125 = modulacionAM(audio,fs,interpolada,1.25)
moduladaFM125, fsFM125 = modulacionFM(audio,fs,interpolada,1.25)

yFourier, yAMFourier, yFMFourier = transFourier(audio, moduladaAM, moduladaFM, fs, fsAM, fsFM)


#3)
# Demodulacion AM
demoduladaAM = demodulacionAM(moduladaAM, fs, interpolada)
demoduladaAM15 = demodulacionAM(moduladaAM15, fs, interpolada)
demoduladaAM125 = demodulacionAM(moduladaAM125, fs, interpolada)

# Grafico demodulacion
tiempo = np.linspace(0,len(audio)/fs, num=len(audio))

plt.figure("Demodulacion 100%")

plt.subplot(211)
plt.plot(tiempo, audio,"c")
plt.xlabel("Tiempo(s)")
plt.ylabel("Amplitud [dB]")
plt.title("Señal Original en el tiempo")

plt.subplot(212)
plt.plot(tiempo, demoduladaAM,"r")
plt.xlabel("Tiempo(s)")
plt.ylabel("Amplitud [dB]")
plt.title("Señal Demodulada")    
    
plt.suptitle("Demodulación 100%", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.88)


plt.figure("Demodulacion 15%")

plt.subplot(211)
plt.plot(tiempo, audio,"c")
plt.xlabel("Tiempo(s)")
plt.ylabel("Amplitud [dB]")
plt.title("Señal Original en el tiempo")

plt.subplot(212)
plt.plot(tiempo, demoduladaAM15,"r")
plt.xlabel("Tiempo(s)")
plt.ylabel("Amplitud [dB]")
plt.title("Señal Demodulada")    
    
plt.suptitle("Demodulación 15%", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.88)


plt.figure("Demodulacion 125%")

plt.subplot(211)
plt.plot(tiempo, audio,"c")
plt.xlabel("Tiempo(s)")
plt.ylabel("Amplitud [dB]")
plt.title("Señal Original en el tiempo")

plt.subplot(212)
plt.plot(tiempo, demoduladaAM125,"r")
plt.xlabel("Tiempo(s)")
plt.ylabel("Amplitud [dB]")
plt.title("Señal Demodulada")    
    
plt.suptitle("Demodulación 125%", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.88)



demoduladaAM = np.asarray(demoduladaAM, dtype=np.int16)
demoduladaAM15 = np.asarray(demoduladaAM15, dtype=np.int16)
demoduladaAM125 = np.asarray(demoduladaAM125, dtype=np.int16)



print("Escribiendo audios...")
wavfile.write("demodulad100%.wav", fs, demoduladaAM)
wavfile.write("demodulad15%.wav", fs, demoduladaAM15)
wavfile.write("demodulad125%.wav", fs, demoduladaAM125)
print("Audio .wav escrito exitosamente.")


##plt.show()
