import pyaudio
import numpy as np
import sys
from scipy.stats import spearmanr
import threading
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class AudioStreamer:
    def __init__(self, sample_rate=44100, chunk_size=2048):
        # stream parameters:
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.eps = sys.float_info.epsilon
        self.left_cal = 1
        self.right_cal = 1
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.stop_event = threading.Event()
        # calculation and plotting:
        self.pref = 20*10**(-6)
        self.plot_data = np.zeros((2, chunk_size), dtype=np.float32)
        self.fft_data = np.zeros((2, chunk_size // 2), dtype=np.float32)
        self.x_axis_fft = np.fft.fftfreq(chunk_size, 1 / sample_rate)[:chunk_size // 2]
        self.ftick = [20, 31.5, 63, 125, 250, 500, 1000, 2000, 3000, 4000, 6000, 8000, 16000, 20000]
        self.labels = ['20', '31.5', '63', '125', '250', '500', '1k', '2k', '3k', '6k', '4k', '8k', '16k', '20k']
        self.devices = self.get_audio_devices()
        self.selected_input_device = tk.StringVar()
        self.selected_input_device.set(self.devices[0])
        self.selected_output_device = tk.StringVar()
        self.selected_output_device.set(self.devices[1])
        self.cal_channels = ['Izquierdo', 'Izquierdo', 'Derecho']
        self.selected_channel = tk.StringVar()
        self.selected_channel.set(self.cal_channels[0])
        self.signal_type = tk.StringVar()
        self.signals = ['White noise', 'White noise', 'Square wave']
        self.signal_type.set(self.signals[0])
        

    def get_audio_devices(self):
        device_info_list = []
        for i in range(self.p.get_device_count()):
            device_info = self.p.get_device_info_by_index(i)
            device_info_list.append(device_info['name'])
        return device_info_list
    
    def set_calibration(self, cal):

        if self.selected_channel.get() == 'Izquierdo':
            self.left_cal = cal
        else:
            self.right_cal = cal

    def start_stream(self):

        if self.stop_event.is_set():
            self.stop_event = threading.Event()
        
        input_device_index = self.devices.index(self.selected_input_device.get())
        output_device_index = self.devices.index(self.selected_output_device.get())
        self.stream = self.p.open(format=pyaudio.paFloat32,
                                  channels=2,  # Tomar dos canales mono
                                  rate=self.sample_rate,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=self.chunk_size,
                                  input_device_index=input_device_index,
                                  output_device_index=output_device_index)  # Índice del dispositivo de entrada

        if self.signal_type.get() == 'White noise':
            noise_thread = threading.Thread(target=self._generate_white_noise)
            noise_thread.start()
        elif self.signal_type.get() == 'Square wave':
            square_thread = threading.Thread(target=self._generate_square_wave)
            square_thread.start()

        audio_thread = threading.Thread(target=self._read_audio)
        self.plot_thread = threading.Thread(target=self._realtime_plot)

        
        audio_thread.start()
        self.plot_thread.start()


    def stop_stream(self):
        self.stop_event.set()

    def _generate_white_noise(self):
        while not self.stop_event.is_set():
            white_noise = np.random.uniform(-1.0, 1.0, (2, self.chunk_size)).astype(np.float32)
            #self.stream.write(white_noise.tobytes())
    
    def _generate_square_wave(self):
        f = 30
        duty_cycle= 0.5
        i = 0
        while not self.stop_event.is_set():
            t = np.arange(i, i+self.chunk_size/self.sample_rate, 1/self.sample_rate)
            square_wave = np.where((t * f) % 1 < duty_cycle, 1, -1).astype(np.float32)
            i += self.chunk_size
            self.stream.write(square_wave.tobytes())

    def _read_audio(self):
        while not self.stop_event.is_set():
            audio_data = self.stream.read(self.chunk_size)
            audio_data = np.frombuffer(audio_data, dtype=np.float32)
            self.plot_data[0] = audio_data[::2] / self.left_cal  # Primer canal (Izquierdo)
            self.plot_data[1] = audio_data[1::2] / self.right_cal # Segundo canal (Derecho)
            #Calculo de la fft:
            self.fft_data[0] = ((2/self.chunk_size) * np.abs(np.fft.fft(self.plot_data[0])))[:self.chunk_size // 2]  # Mitad derecha del primer canal dividido por la compensación de amplitud de fft
            self.fft_data[1] = ((2/self.chunk_size) *np.abs(np.fft.fft(self.plot_data[1])))[:self.chunk_size // 2]  # Mitad derecha del segundo canal dividido por la compensación de amplitud de fft
            #Paso a dB:
            self.fft_data[0] = 20*np.log10((self.fft_data[0] / self.pref) + self.eps)
            self.fft_data[1] = 20*np.log10((self.fft_data[1] / self.pref) + self.eps)

    def _realtime_plot(self):
        root = tk.Tk()
        root.title("Posicionamiento de auriculares")
        root.iconbitmap('logo.ico')
        fig = Figure(figsize=(10, 7), dpi=100)
        waveform_plot = fig.add_subplot(211)
        waveform_plot.set_ylim(-1.0, 1.0)
        waveform_line1, = waveform_plot.plot(self.plot_data[0], 'b', label=f'Max: {np.max(self.plot_data[0]):.1f}')
        waveform_line2, = waveform_plot.plot(self.plot_data[1], 'r', label=f'Max: {np.max(self.plot_data[1]):.1f}')
        waveform_plot.legend()
        fft_plot = fig.add_subplot(212)
        spearman = spearmanr(self.fft_data[0], self.fft_data[1])
        area = np.trapz(y=self.fft_data[0]-self.fft_data[1], x=self.x_axis_fft)
        fft_plot.set_title(f"r: {spearman.correlation:.2f} , p: {spearman.pvalue:.2f} , area: {area:.2f}")
        fft_line1, = fft_plot.semilogx(self.x_axis_fft, self.fft_data[0], 'b', label=f'Nivel: {self._global_level(self.fft_data[0]):.1f} dBSPL')
        fft_line2, = fft_plot.semilogx(self.x_axis_fft, self.fft_data[1], 'r', label=f'Nivel: {self._global_level(self.fft_data[1]):.1f} dBSPL')
        fft_plot.set_xticks(self.ftick)
        fft_plot.set_xticklabels(self.labels, rotation=45)
        fft_plot.set_xlim(0, self.sample_rate // 2)
        fft_plot.set_ylim(-20, 100)
        fft_plot.set_xlim(2000, 20000)
        fft_plot.legend()
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack()

        while not self.stop_event.is_set():
            waveform_line1.set_ydata(self.plot_data[0])
            waveform_line2.set_ydata(self.plot_data[1])
            waveform_line1.set_label(f'Max: {np.max(self.plot_data[0]):.1f}')
            waveform_line2.set_label(f'Max: {np.max(self.plot_data[1]):.1f}')
            waveform_plot.legend()
            spearman = spearmanr(self.fft_data[0], self.fft_data[1])
            area = np.trapz(y=self.fft_data[0]-self.fft_data[1], x=self.x_axis_fft)
            fft_plot.set_title(f"r: {spearman.correlation:.2f} , p: {spearman.pvalue:.2f} , area: {area:.2f}")
            fft_line1.set_ydata(self.fft_data[0])
            fft_line2.set_ydata(self.fft_data[1])
            fft_line1.set_label(f'Nivel: {self._global_level(self.fft_data[0]):.1f} dBSPL')
            fft_line2.set_label(f'Nivel: {self._global_level(self.fft_data[1]):.1f} dBSPL')
            fft_plot.legend()
            fig.canvas.draw()
            fig.canvas.flush_events()

        root.destroy()

    def _global_level(self, data):
        #Calculo del nivel global:

        #Convierto los valores en dBSPL a escala lineal
        linear_values = [10**(dB/10) for dB in data]
        #Calculo el promedio de los valores lineales
        linear_mean_value = np.mean(linear_values)
        #Vuelvo a pasar a dB:
        return 10 * np.log10(linear_mean_value)

    def close(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()

# Funciones para los botones de la interfaz
def start_stream():
    global audio_streamer
    audio_streamer.start_stream()

def stop_stream():
    global audio_streamer
    audio_streamer.stop_stream()
    audio_streamer.close()

def close_window():
    global audio_streamer

    audio_streamer.p.terminate()
    root.destroy()

def record_calibration():
    global audio_streamer

    duration = 5

    input_device_index = audio_streamer.devices.index(audio_streamer.selected_input_device.get())
    
    stream = audio_streamer.p.open(format=pyaudio.paFloat32,
                                   channels=2,  # Tomar dos canales mono
                                   rate=audio_streamer.sample_rate,
                                   input=True,
                                   frames_per_buffer=audio_streamer.chunk_size,
                                   input_device_index=input_device_index,)  # Índice del dispositivo de entrada

    frames_izq, frames_der = [], []
    for _ in range(0, int(audio_streamer.sample_rate / audio_streamer.chunk_size * duration)):
        data = stream.read(audio_streamer.chunk_size)
        data_np = np.frombuffer(data, dtype=np.float32)
        frames_izq.append(data_np[::2])  # Canal izquierdo
        frames_der.append(data_np[1::2])  # Canal derecho
    
    stream.stop_stream()
    stream.close()

    cal_izq = np.concatenate(frames_izq)
    cal_der = np.concatenate(frames_der)
    

    if audio_streamer.selected_channel.get() == 'Izquierdo':

        cal = np.sqrt(np.mean(cal_izq**2))
        audio_streamer.set_calibration(cal)
        print(cal)
        print('Grabación de calibración izquierda finalizada!')

    else:

        cal = np.sqrt(np.mean(cal_der**2))
        audio_streamer.set_calibration(cal)
        print(cal)
        print('Grabación de calibración derecha finalizada!')
    


if __name__ == '__main__':

    root = tk.Tk()
    root.title("Posicionamiento de auriculares")
    root.geometry("400x450")
    root.iconbitmap('logo.ico')

    # Crear instancia de AudioStreamer
    audio_streamer = AudioStreamer()

    # Crear lista desplegable para seleccionar el dispositivo de entrada de audio
    device_input_label = tk.Label(root, text="Dispositivo de entrada:")
    device_input_label.pack(pady=10)

    input_optionmenu = ttk.OptionMenu(root, audio_streamer.selected_input_device, *audio_streamer.devices)
    input_optionmenu.pack()

    # Crear lista desplegable para seleccionar el dispositivo de salida de audio
    device_output_label = tk.Label(root, text="Dispositivo de salida:")
    device_output_label.pack(pady=10)

    output_optionmenu = ttk.OptionMenu(root, audio_streamer.selected_output_device, *audio_streamer.devices)
    output_optionmenu.pack()

    #Calibración del sistema:
    # Crear lista desplegable para seleccionar el dispositivo de audio
    cal_channel = tk.Label(root, text="Canal a calibrar:")
    cal_channel.pack(pady=10)

    channel_optionmenu = ttk.OptionMenu(root, audio_streamer.selected_channel, *audio_streamer.cal_channels)
    channel_optionmenu.pack()

    cal = tk.Button(root, text="Calibración", command=record_calibration)
    cal.pack(pady=10)

    # Botones de inicio y detención del stream
    signal_optionmenu = ttk.OptionMenu(root, audio_streamer.signal_type, *audio_streamer.signals)
    signal_optionmenu.pack()

    start_button = tk.Button(root, text="Start Stream", command=start_stream)
    start_button.pack(pady=10)

    stop_button = tk.Button(root, text="Stop Stream", command=stop_stream)
    stop_button.pack(pady=10)

    close_button = tk.Button(root, text="Close", command=close_window)
    close_button.pack(pady=10)

    root.mainloop()