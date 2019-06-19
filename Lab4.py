    # --------- IMPORTACIONES ---------
import scipy.io
from scipy import interpolate
from scipy.io import wavfile
from scipy.fftpack import fft,fftfreq,ifft

import numpy as np
import matplotlib.pyplot as plt


# --------- FUNCIONES ---------

def graph_portadora(portadora,rate):
    plt.xlabel('Tiempo (s)')
    plt.ylabel('f(t)')
    Tiempo = np.linspace(0, len(portadora) / rate, num=len(portadora))
    plt.plot(Tiempo, portadora)
    plt.xlim(0,0.015)
    plt.show()

def GraficoAM(signal,frecuencia,audio, filename):
    plt.xlabel('Tiempo (s)')
    plt.ylabel('f(t)')
    Tiempo = np.linspace(0, len(audio) / frecuencia, num=len(audio))
    plt.plot(Tiempo, audio, 'g', label = "Senal audio "+filename)
    plt.plot(Tiempo, signal, label = "Senal modulada")
    plt.legend()
    plt.xlim(0, 0.1)
    plt.show()

def graph_AM(signal,rate,data,portadora,filename):
    plt.ion()
    plt.figure("Portadora")
    graph_portadora(portadora,rate)
    plt.pause(0.0001)
    plt.figure("Modulacion AM")
    GraficoAM(signal, rate, data, filename)
    plt.pause(0.0001)
    plt.show(block=True)

def interpolacionAM(signal,frecuencia):
    Tiempo=np.linspace(0, len(signal)/frecuencia, num=len(signal))
    interpolada = interpolate.interp1d(Tiempo, signal)
    Tiempo2 = np.linspace(0, len(signal)/frecuencia, len(signal)*4)
    y2 = interpolada(Tiempo2)
    return y2


def modulation_am_time():
    rate,data=wavfile.read("handel.wav")
    signal_interp_AM = interpolacionAM(data,rate)
    largo_AM = len(signal_interp_AM)
    tiempo_AM = np.linspace(0, largo_AM/rate, num = largo_AM)
    portadora = np.cos(2*np.pi*27000000*tiempo_AM)
    y = signal_interp_AM * portadora
    graph_AM(y,rate,signal_interp_AM,portadora,"handel.wav")


# --------- BLOQUE PRINCIPAL ---------


# 1)
# --------- DATOS DE LA SENAL ---------

# Importacion de los datos de la senal
fs,audio = wavfile.read("handel.wav")

# Largo de la senal
largo = float(len(audio))
# Tiempo de la senal
t = largo / fs
dt = 1 / float(fs)
# Generar arreglo
tArreglo = np.linspace(0, t, int(largo))

# Muestra por pantalla de datos de senal
print("Datos de senal original")
print("frecuencia: ",fs)
print("tiempo: ",t)

### 2)
### --------- GRAFICO DE SENAL ---------
##plt.figure(1)
##plt.plot(tArreglo, audio, "b")
##plt.xlabel("Tiempo [s]")
##plt.ylabel("Amplitud [dB]")
##plt.title("Grafico en el dominio del tiempo", )
##
##
### 3)
### --------- TRANSFORMADA DE FOURIER ---------
##audioFourier = fft(audio)
##yFourier = audioFourier / largo
##yFourierPos = abs(fft(audio) / largo) #valor abs
##fsFourier = fftfreq(int(largo), dt)
##
### 3.a)
### Grafico transformada de Fourier
##plt.figure(2)
##plt.plot(fsFourier, yFourierPos, "r")
##plt.xlabel("Frecuencia [Hz]")
##plt.ylabel("Amplitud [dB]")
##plt.title("Transformada de Fourier grafico")

# --------- ESCRITURA ARCHIVO .WAV ---------
# Ajuste datos para que sean aceptadas en la funcion wavfile.write
#fourierInvFiltro = np.asarray(yFourierInvFiltro, dtype=np.int16)

print("Escribiendo audios...")
#wavfile.write("FourierInv.wav", fs, fourierInv)
#wavfile.write("FourierInvFiltro.wav", fs, fourierInvFiltro)
print("Audio .wav escrito exitosamente.")




Tiempo = np.linspace(0, len(audio)/fs, num=len(audio))
interpolada = interpolate.interp1d(Tiempo, audio)
Tiempo2 = np.linspace(0, len(audio)/fs, len(audio)*4)
y2 = interpolada(Tiempo2)



#largo_AM = len(signal_interp_AM)
tiempo_AM = np.linspace(0, len(Tiempo), 1)
portadora = np.cos(2*np.pi*100*tiempo_AM)



print(len(y2))

plt.figure(3)
plt.plot(tiempo_AM, portadora)
#plt.xlim(0, 100)

plt.show() # Mostrar todos los graficos
