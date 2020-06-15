
import argparse
import os
import numpy as np
import wave
import struct

def mic_parse(opts):
  adc_offset = 2681
  raw_mic = []
  with open(opts.file, 'r') as f:
    line = f.readline()
    while line:
      line = f.readline()
      if 'Pressed' in line or len(line) == 0 or line=="\n":
        continue
      data = np.int16(line)
      raw_mic.append(data)

  samples = np.array(raw_mic)
  print(np.mean(samples))
  samples = samples - adc_offset
  sample_rate = 16000
  out_f = 'mic_out.wav'
  if not os.path.exists(opts.out_dir):
    os.makedirs(opts.out_dir)

  # import matplotlib.pyplot as plt
  # plt.plot(samples)
  # plt.show()
  
  print(f'INFO: num of samples: {len(samples)}')
  obj = wave.open(os.path.join(opts.out_dir, out_f), 'w')
  obj.setnchannels(1)
  obj.setsampwidth(2)
  obj.setframerate(sample_rate)
  for i, item in enumerate(samples):
    data = struct.pack('<h', item)
    obj.writeframesraw(data)
  obj.close()
  print(f'INFO: Samples max:{max(samples)}, min: {min(samples)}')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-o', '--out-dir', default='build')
  parser.add_argument('-f', '--file', default=None, help='Recorded data from serial port.')
  opts = parser.parse_args()
  mic_parse(opts)