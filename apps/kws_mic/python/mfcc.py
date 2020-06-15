import argparse
import os

import tensorflow as tf
try:
    tf_compat_v1 = tf.compat.v1
except ImportError:
    tf_compat_v1 = tf

DESIRED_SAMPLES = 16000

def prepare_data(filename):
    from tensorflow.contrib.framework.python.ops import audio_ops
    from tensorflow.python.ops import io_ops
    import scipy.io.wavfile

    desired_channels = 1
    with tf.Session(graph=tf.Graph()) as sess:
        wav_filename_placeholder = tf.placeholder(tf.string, [])
        wav_loader = io_ops.read_file(wav_filename_placeholder)
        wav_decoder = tf.audio.decode_wav(wav_loader,
                            desired_channels=desired_channels,
                            desired_samples=16000,
                            name='decoded_sample_data')

        spectrum = audio_ops.audio_spectrogram(input=wav_decoder[0],
                                            window_size=640,
                                            stride=320,
                                            magnitude_squared=True,
                                            name='AudioSpectrogram')
        final = audio_ops.mfcc(spectrogram=spectrum, 
                               sample_rate=wav_decoder[1], 
                               upper_frequency_limit=4000.0, 
                               lower_frequency_limit=20.0, 
                               filterbank_channel_count=40, 
                               dct_coefficient_count=10, 
                               name='Mfcc')

        mfcc = sess.run(final,
        feed_dict={wav_filename_placeholder: filename}) 
        mfcc = mfcc.flatten()

    samplerate, wav_data = scipy.io.wavfile.read(filename)
    return mfcc, wav_data

def generate_wav_header(opts, filename, data, dtype):
    if data.dtype != dtype:
        data = data.astype(dtype)

    if data.size < DESIRED_SAMPLES:
        raise ValueError('Data is short!')

    data_str = "int16_t WAVE_DATA[] = {"
    for (ii, item) in enumerate(data):
        if ii < DESIRED_SAMPLES-1:
            data_str += f"{item},"
        elif ii == DESIRED_SAMPLES-1:
            data_str += f"{item}"
        else:
            break

    data_str += "};"
    with open(os.path.join(opts.out_dir, filename), 'w') as f:
        f.write(data_str)

def generate_mfcc_log(opts, filename, data, dtype):
    if data.dtype != dtype:
        data = data.astype(dtype)

    data_str = "int8_t MFCC_RESULT[] = {"
    for (ii, item) in enumerate(data):
        if dtype=='float32':
            data_str += f"{item:.5f},\n"
        else:
            data_str += f"{item},\n"
    data_str += "};"

    with open(os.path.join(opts.out_dir, filename), 'w') as f:
        f.write(data_str)

def mfcc_compare(opts, mfcc_expect):
    mfcc_len = 490
    with open(opts.compare, 'r') as f:
        line = f.readline()
        while(line):
            if "Extract Features Done!" in line:
                break
            line = f.readline()
        
        mfcc_out = []
        for i in range(mfcc_len):
            line = f.readline()
            mfcc_out.append(float(line))

    for i in range(mfcc_len):
        if abs(mfcc_expect[i] - mfcc_out[i]) > 1e-3:
            print(f'Ind: {i}\tEXP: {mfcc_expect[i]:.5f}\tOUT: {mfcc_out[i]:.5f}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out-dir', default='build')
    parser.add_argument('-f', '--wav-file', default=None)
    parser.add_argument('-c', '--compare', default=None)

    opts = parser.parse_args()

    file_path = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(file_path, opts.out_dir)
    opts.out_dir = build_dir
    if not os.path.exists(build_dir):
        os.mkdir(build_dir)

    if opts.wav_file:
        mfcc, wav_data = prepare_data(opts.wav_file)

        generate_wav_header(opts, 'wav_data.h', wav_data, 'int16')
        generate_mfcc_log(opts, 'mfcc_data.h', mfcc, 'float32')
        
        if opts.compare:
            mfcc_compare(opts, mfcc)


