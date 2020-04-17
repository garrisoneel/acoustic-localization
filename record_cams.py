import wave
import pyaudio
import cv2
import threading
import time


class camThread(threading.Thread):
    def __init__(self, previewName, camID, frame_width=1920, frame_height=1080, fps=30):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps

        self.audio_FORMAT = pyaudio.paInt16
        self.audio_CHANNELS = 2
        self.audio_RATE = 44100
        self.audio_CHUNK = 1024
        self.audio_RECORD_SECONDS = 5
        self.WAVE_OUTPUT_FILENAME = "file.wav"

    def run(self):
        print("Recording {}".format(self.previewName))
        record_vid(self.previewName, self.camID, self.frame_width, self.frame_height, self.fps)

    def record_vid(self, frame_width, frame_height, fps):
        p = pyaudio.PyAudio()
        audio_stream = p.open(format = self.audio_FORMAT,
                        channels = self.audio_CHANNELS,
                        rate = self.audio_RATE,
                        input = True,
                        frames_per_buffer = self.audio_CHUNK,
                        input_device_index = self.camID)
        
        audio = []
        out = cv2.VideoWriter('{}_{}.avi'.format(self.previewName,time.strftime("%Y%m%d-%H%M%S")),cv2.VideoWriter_fourcc('M','J','P','G'), fps, (self.frame_width, self.frame_height))
        cam = cv2.VideoCapture(self.camID)
        audio_stream.start_stream()
        if cam.isOpened():  # try to get the first frame
            rval, frame = cam.read()
            audio_signal = audio_stream.read(self.audio_CHUNK)
        else:
            rval = False
            print("Video is not being captured.")

        while rval:
            out.write(frame)
            audio.append(audio_signal)
            rval, frame = cam.read()
            audio_signal = audio_stream.read(self.audio_CHUNK)
            key = cv2.waitKey(20)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

        cam.release()
        out.release()
        audio_stream.close()
        wf = wave.open(self.WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(self.audio_CHANNELS)
        wf.setsampwidth(p.get_sample_size(self.audio_FORMAT))
        wf.setframerate(self.audio_RATE)
        wf.writeframes(b''.join(audio))
        wf.close()

# Create two threads as follows
thread1 = camThread("Camera 1", 0)
thread2 = camThread("Camera 2", 1)
thread1.start()
thread2.start()