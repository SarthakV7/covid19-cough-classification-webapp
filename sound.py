import pyaudio, wave
import numpy as np
from scipy.io.wavfile import write
import datetime

class Sound(object):
    def __init__(self):
        self.format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = 48000
        self.chunk = 1024
        self.duration = 6
        self.path = str(datetime.datetime.now()).replace('.','').replace(':','').replace(' ','-') + '.wav'
        self.device = 0
        self.frames = []
        self.audio = pyaudio.PyAudio()

    def record(self):
        self.audio = pyaudio.PyAudio()
        stream = self.audio.open(
                        format=self.format,
                        channels=self.channels,
                        rate=self.sample_rate,
                        input=True,
                        frames_per_buffer=self.chunk,
                        input_device_index=self.device)
        self.frames = []
        for i in range(0, int(self.sample_rate / self.chunk * self.duration)):
            data = stream.read(self.chunk)
            self.frames.append(data)
        stream.stop_stream()
        stream.close()
        self.audio.terminate()
        self.save()
        return self.path

    def save(self):
        waveFile = wave.open(self.path, 'wb')
        waveFile.setnchannels(self.channels)
        waveFile.setsampwidth(self.audio.get_sample_size(self.format))
        waveFile.setframerate(self.sample_rate)
        waveFile.writeframes(b''.join(self.frames))
        waveFile.close()

sound = Sound()
