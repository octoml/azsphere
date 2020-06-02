import pyaudio
import wave
import tensorflow as tf
import numpy as np
from tensorflow.contrib.framework.python.ops import audio_ops

# Set chunk size of 1024 samples per data frame
CHUNK = 1000
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 1
WAVE_OUTPUT_FILENAME = "myfile.wav" 


def process_audio(audio_data):
    with tf.Session() as sess:
        data_input = tf.compat.v1.placeholder(dtype=tf.float32, shape=(16000, 1), name=None)
        spectrum = audio_ops.audio_spectrogram(input=data_input,
                                            window_size=640,
                                            stride=320,
                                            magnitude_squared=True,
                                            name='AudioSpectrogram')
        final = audio_ops.mfcc(spectrogram=spectrum, 
                                sample_rate=RATE, 
                                upper_frequency_limit=4000.0, 
                                lower_frequency_limit=20.0, 
                                filterbank_channel_count=40, 
                                dct_coefficient_count=10, 
                                name='Mfcc')
    
        data_out = sess.run(final,
        feed_dict={data_input: audio_data})

    return data_out


# Create an interface to PortAudio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

audio_data = np.zeros((RATE * RECORD_SECONDS, 1))
process_audio(audio_data)

sess = tf.Session()
data_input = tf.compat.v1.placeholder(dtype=tf.float32, shape=(16000, 1), name=None)
spectrum = audio_ops.audio_spectrogram(input=data_input,
                                    window_size=640,
                                    stride=320,
                                    magnitude_squared=True,
                                    name='AudioSpectrogram')
final = audio_ops.mfcc(spectrogram=spectrum,
                        sample_rate=RATE, 
                        upper_frequency_limit=4000.0, 
                        lower_frequency_limit=20.0, 
                        filterbank_channel_count=40, 
                        dct_coefficient_count=10, 
                        name='Mfcc')

print("** recording")
while(True):
    frame = np.zeros((RATE * RECORD_SECONDS, 1))
    print(type(frame))
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data_chunk = stream.read(CHUNK)
        print(type(data_chunk))
        # frame[int(i*CHUNK): int((i+1)*CHUNK)] = float()
        print(i)
    break
    print(frame)
    break





print("* done recording")
stream.stop_stream()
stream.close()
p.terminate()

# wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
# wf.setnchannels(CHANNELS)
# wf.setsampwidth(p.get_sample_size(FORMAT))
# wf.setframerate(RATE)
# wf.writeframes(b''.join(frames))
# wf.close()

