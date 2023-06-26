"""
# En 16 bits enteros.

import pyaudio
import numpy as np
import threading
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class AudioStreamer:
    def __init__(self, sample_rate=44100, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.stop_event = threading.Event()
        self.plot_data = np.zeros(chunk_size)

    def start_stream(self):
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=self.sample_rate,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=self.chunk_size)

        noise_thread = threading.Thread(target=self._generate_white_noise)
        audio_thread = threading.Thread(target=self._read_audio)
        self.plot_thread = threading.Thread(target=self._realtime_plot)

        noise_thread.start()
        audio_thread.start()
        self.plot_thread.start()

    def stop_stream(self):
        self.stop_event.set()

    def _generate_white_noise(self):
        while not self.stop_event.is_set():
            white_noise = np.random.randint(-32767, 32767, self.chunk_size, dtype=np.int16)
            self.stream.write(white_noise.tobytes())

    def _read_audio(self):
        while not self.stop_event.is_set():
            audio_data = self.stream.read(self.chunk_size)
            self.plot_data = np.frombuffer(audio_data, dtype=np.int16)

    def _realtime_plot(self):
        root = tk.Tk()
        root.title("Real-Time Plot")
        fig = Figure(figsize=(5, 4), dpi=100)
        plot = fig.add_subplot(111)
        plot.set_ylim(-32767, 32767)
        line, = plot.plot(self.plot_data)
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack()

        while not self.stop_event.is_set():
            line.set_ydata(self.plot_data)
            fig.canvas.draw()
            fig.canvas.flush_events()

        root.destroy()

    def close(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

# Funciones para los botones de la interfaz
def start_stream():
    global audio_streamer
    audio_streamer = AudioStreamer()
    audio_streamer.start_stream()

def stop_stream():
    global audio_streamer
    audio_streamer.stop_stream()
    audio_streamer.close()

# Crear la interfaz gráfica
root = tk.Tk()
root.title("Audio Streaming")
root.geometry("400x200")

start_button = tk.Button(root, text="Start Stream", command=start_stream)
start_button.pack(pady=10)

stop_button = tk.Button(root, text="Stop Stream", command=stop_stream)
stop_button.pack(pady=10)

root.mainloop()





#En 32 bits flotantes:

import pyaudio
import numpy as np
import threading
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class AudioStreamer:
    def __init__(self, sample_rate=44100, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.stop_event = threading.Event()
        self.plot_data = np.zeros(chunk_size, dtype=np.float32)

    def start_stream(self):
        self.stream = self.p.open(format=pyaudio.paFloat32,
                                  channels=1,
                                  rate=self.sample_rate,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=self.chunk_size)

        noise_thread = threading.Thread(target=self._generate_white_noise)
        audio_thread = threading.Thread(target=self._read_audio)
        self.plot_thread = threading.Thread(target=self._realtime_plot)

        noise_thread.start()
        audio_thread.start()
        self.plot_thread.start()

    def stop_stream(self):
        self.stop_event.set()

    def _generate_white_noise(self):
        while not self.stop_event.is_set():
            white_noise = np.random.uniform(-1.0, 1.0, self.chunk_size).astype(np.float32)
            self.stream.write(white_noise.tobytes())

    def _read_audio(self):
        while not self.stop_event.is_set():
            audio_data = self.stream.read(self.chunk_size)
            self.plot_data = np.frombuffer(audio_data, dtype=np.float32)

    def _realtime_plot(self):
        root = tk.Tk()
        root.title("Posicionamiento de auriculares")
        fig = Figure(figsize=(5, 4), dpi=100)
        plot = fig.add_subplot(111)
        plot.set_ylim(-1.0, 1.0)
        line, = plot.plot(self.plot_data)
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack()

        while not self.stop_event.is_set():
            line.set_ydata(self.plot_data)
            fig.canvas.draw()
            fig.canvas.flush_events()

        root.destroy()

    def close(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

# Funciones para los botones de la interfaz
def start_stream():
    global audio_streamer
    audio_streamer = AudioStreamer()
    audio_streamer.start_stream()

def stop_stream():
    global audio_streamer
    audio_streamer.stop_stream()
    audio_streamer.close()

# Crear la interfaz gráfica
root = tk.Tk()
root.title("Audio Streaming")
root.geometry("400x200")

start_button = tk.Button(root, text="Start Stream", command=start_stream)
start_button.pack(pady=10)

stop_button = tk.Button(root, text="Stop Stream", command=stop_stream)
stop_button.pack(pady=10)

root.mainloop()


"""

import pyaudio
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class AudioStreamer:
    def __init__(self, sample_rate=44100, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.stop_event = threading.Event()
        self.plot_data = np.zeros((2, chunk_size), dtype=np.float32)
        self.fft_data = np.zeros(chunk_size // 2, dtype=np.float32)
        self.x_axis = np.linspace(0, sample_rate / 2, chunk_size // 2)
        self.devices = self.get_audio_devices()
        self.selected_device = tk.StringVar()
        self.selected_device.set(self.devices[0])

    def get_audio_devices(self):
        device_info_list = []
        for i in range(self.p.get_device_count()):
            device_info = self.p.get_device_info_by_index(i)
            device_info_list.append(device_info['name'])
        return device_info_list

    def start_stream(self):
        selected_device = self.selected_device.get()
        device_index = self.devices.index(selected_device)
        self.stream = self.p.open(format=pyaudio.paFloat32,
                                  channels=2,  # Tomar dos canales mono
                                  rate=self.sample_rate,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=self.chunk_size,
                                  input_device_index=device_index)  # Índice del dispositivo de entrada

        noise_thread = threading.Thread(target=self._generate_white_noise)
        audio_thread = threading.Thread(target=self._read_audio)
        self.plot_thread = threading.Thread(target=self._realtime_plot)

        noise_thread.start()
        audio_thread.start()
        self.plot_thread.start()

    def stop_stream(self):
        self.stop_event.set()

    def _generate_white_noise(self):
        while not self.stop_event.is_set():
            white_noise = np.random.uniform(-1.0, 1.0, (2, self.chunk_size)).astype(np.float32)
            self.stream.write(white_noise.tobytes())

    def _read_audio(self):
        while not self.stop_event.is_set():
            audio_data = self.stream.read(self.chunk_size)
            audio_data = np.frombuffer(audio_data, dtype=np.float32)
            self.plot_data[0] = audio_data[::2]  # Primer canal
            self.plot_data[1] = audio_data[1::2]  # Segundo canal
            self.fft_data = np.abs(np.fft.fft(audio_data))[:self.chunk_size // 2]  # Tomar solo la mitad derecha

    def _realtime_plot(self):
        root = tk.Tk()
        root.title("Audio Streaming")
        fig = Figure(figsize=(10, 6), dpi=100)
        waveform_plot = fig.add_subplot(211)
        waveform_plot.set_ylim(-1.0, 1.0)
        waveform_line1, = waveform_plot.plot(self.plot_data[0])
        waveform_line2, = waveform_plot.plot(self.plot_data[1])
        fft_plot = fig.add_subplot(212)
        fft_plot.set_xlim(0, self.sample_rate // 2)
        fft_plot.set_ylim(0, 1000)
        fft_line, = fft_plot.plot(self.x_axis, self.fft_data)
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack()

        while not self.stop_event.is_set():
            waveform_line1.set_ydata(self.plot_data[0])
            waveform_line2.set_ydata(self.plot_data[1])
            fft_line.set_ydata(self.fft_data)
            fig.canvas.draw()
            fig.canvas.flush_events()

        root.destroy()

    def close(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

# Funciones para los botones de la interfaz
def start_stream():
    global audio_streamer
    audio_streamer = AudioStreamer()
    audio_streamer.start_stream()

def stop_stream():
    global audio_streamer
    audio_streamer.stop_stream()
    audio_streamer.close()

if __name__ == '__main__':

    root = tk.Tk()
    root.title("Audio Streaming")
    root.geometry("400x200")

    # Crear instancia de AudioStreamer
    audio_streamer = AudioStreamer()

    # Crear lista desplegable para seleccionar el dispositivo de audio
    device_label = tk.Label(root, text="Dispositivo de audio:")
    device_label.pack(pady=10)

    device_optionmenu = ttk.OptionMenu(root, audio_streamer.selected_device, *audio_streamer.devices)
    device_optionmenu.pack()

    # Botones de inicio y detención del stream
    start_button = tk.Button(root, text="Start Stream", command=start_stream)
    start_button.pack(pady=10)

    stop_button = tk.Button(root, text="Stop Stream", command=stop_stream)
    stop_button.pack(pady=10)

    root.mainloop()



