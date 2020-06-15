#include <string.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <inttypes.h>
#include <float.h>
#include <arm_math.h>

#include "include/mfcc.h"
#include "include/config.h"

mfcc* MFCC_Init(int num_mfcc_features, int frame_len, int mfcc_dec_bits){
  #if AS_DEBUG
  UART_Print(debug, "MFCC Init!\n");
  #endif
  mfcc* new_mfcc = (mfcc *)malloc(sizeof(mfcc));
  if (!new_mfcc) {
    return NULL;
  }

  new_mfcc->num_mfcc_features = num_mfcc_features;
  new_mfcc->frame_len = frame_len;
  new_mfcc->dec_bits = mfcc_dec_bits;

  // Round-up to nearest power of 2.
  new_mfcc->frame_len_padded = pow(2, ceil((log(frame_len)/log(2))));

  new_mfcc->frame =  (float *)malloc(new_mfcc->frame_len_padded * sizeof(float));
  if (!new_mfcc->frame) {
    return NULL;
  }

  new_mfcc->buffer = (float *)malloc(new_mfcc->frame_len_padded * sizeof(float));
  if (!new_mfcc->buffer) {
    return NULL;
  }
  new_mfcc->mel_energies = (float *)malloc(NUM_FBANK_BINS * sizeof(float));
  if (!new_mfcc->mel_energies) {
    return NULL;
  }

  //create window function
  new_mfcc->window_func = (float *)malloc(new_mfcc->frame_len * sizeof(float));
  if (!new_mfcc->window_func) {
    return NULL;
  }
  for (int i = 0; i<new_mfcc->frame_len; i++)
    new_mfcc->window_func[i] = 0.5 - 0.5*cos(M_2PI * ((float)i) / (new_mfcc->frame_len));

  //create mel filterbank
  new_mfcc->fbank_filter_first = (int32_t *)malloc(NUM_FBANK_BINS * sizeof(int32_t));
    if (!new_mfcc->fbank_filter_first) {
    return NULL;
  }

  new_mfcc->fbank_filter_last = (int32_t *)malloc(NUM_FBANK_BINS * sizeof(int32_t));
  if (!new_mfcc->fbank_filter_last) {
    return NULL;
  }

  new_mfcc->mel_fbank = MFCC_Create_MEL_FBank(new_mfcc);
  if (!new_mfcc->mel_fbank) {
    return NULL;
  }

  //create DCT matrix
  new_mfcc->dct_matrix = MFCC_Create_DCT_Matrix(NUM_FBANK_BINS, new_mfcc->num_mfcc_features);
  if (!new_mfcc->dct_matrix) {
    return NULL;
  }

  //initialize FFT
  new_mfcc->rfft = (arm_rfft_fast_instance_f32 *)malloc(sizeof(arm_rfft_fast_instance_f32));
  if (!new_mfcc->rfft) {
    return NULL;
  }

  #if AS_DEBUG
  UART_Printf(debug, "frame_len_padded: %d\n", new_mfcc->frame_len_padded);
  #endif
  arm_status status = arm_rfft_fast_init_f32(new_mfcc->rfft, new_mfcc->frame_len_padded);
  #if AS_DEBUG
  UART_Printf(debug, "arm status: %d\n", status);
  #endif
  if (status != ARM_MATH_SUCCESS) {
    return NULL;
  }
  return new_mfcc;
}

void MFCC_Close(mfcc* mfcc) {
  free(mfcc->frame);
  free(mfcc->buffer);
  free(mfcc->mel_energies);
  free(mfcc->window_func);
  free(mfcc->fbank_filter_first);
  free(mfcc->fbank_filter_last);
  free(mfcc->dct_matrix);
  free(mfcc->rfft);
  for(int i=0;i<NUM_FBANK_BINS;i++)
    free(mfcc->mel_fbank[i]);
  free(mfcc->mel_fbank);
  free(mfcc);
}

float * MFCC_Create_DCT_Matrix(int32_t input_length, int32_t coefficient_count) {

  float * M = (float *) malloc(input_length * coefficient_count * sizeof(float));
  float normalizer;
  arm_sqrt_f32(2.0/(float)input_length, &normalizer);
  for (int32_t k = 0; k < coefficient_count; k++) {
    for (int32_t n = 0; n < input_length; n++) {
      M[k * input_length + n] = normalizer * cos(((double)M_PI)/input_length * (n + 0.5) * k);
    }
  }
  return M;
}

float ** MFCC_Create_MEL_FBank(mfcc* mfcc) {
  int32_t num_fft_bins = mfcc->frame_len_padded / 2;
  float fft_bin_width = ((float)SAMP_FREQ) / mfcc->frame_len_padded;
  float mel_low_freq = MelScale(MEL_LOW_FREQ);
  float mel_high_freq = MelScale(MEL_HIGH_FREQ); 
  float mel_freq_delta = (mel_high_freq - mel_low_freq) / (NUM_FBANK_BINS+1);

  float *this_bin = (float *)malloc(num_fft_bins * sizeof(float));

  // float ** mel_fbank =  new float*[NUM_FBANK_BINS];
  float ** mel_fbank =  (float **) malloc(NUM_FBANK_BINS * sizeof(float *));

  for (int32_t bin=0; bin<NUM_FBANK_BINS; bin++) {

    float left_mel = mel_low_freq + bin * mel_freq_delta;
    float center_mel = mel_low_freq + (bin + 1) * mel_freq_delta;
    float right_mel = mel_low_freq + (bin + 2) * mel_freq_delta;

    int32_t first_index = -1, last_index = -1;

    for (int32_t i =0; i<num_fft_bins; i++) {

      float freq = (fft_bin_width * i);  // center freq of this fft bin.
      float mel = MelScale(freq);
      this_bin[i] = 0.0;

      if (mel > left_mel && mel < right_mel) {
        float weight;
        if (mel <= center_mel) {
          weight = (mel - left_mel) / (center_mel - left_mel);
        } else {
          weight = (right_mel-mel) / (right_mel-center_mel);
        }
        this_bin[i] = weight;
        if (first_index == -1)
          first_index = i;
        last_index = i;
      }
    }

    mfcc->fbank_filter_first[bin] = first_index;
    mfcc->fbank_filter_last[bin] = last_index;
    mel_fbank[bin] = (float *) malloc((last_index - first_index + 1) * sizeof(float)); 

    int32_t j = 0;
    //copy the part we care about
    for (int32_t i = first_index; i <= last_index; i++) {
      mel_fbank[bin][j++] = this_bin[i];
    }
  }
  free(this_bin);
  return mel_fbank;
}

int8_t MFCC_Compute(mfcc* mfcc, const int16_t * audio_data, float* mfcc_out) {
  int32_t i, j, bin;

  //TensorFlow way of normalizing .wav data to (-1,1)
  for (i = 0; i < mfcc->frame_len; i++) {
    mfcc->frame[i] = (float)audio_data[i]/(1<<15); 
  }
  //Fill up remaining with zeros
  memset(&mfcc->frame[mfcc->frame_len], 0, sizeof(float) * (mfcc->frame_len_padded-mfcc->frame_len));

  for (i = 0; i < mfcc->frame_len; i++) {
    mfcc->frame[i] *= mfcc->window_func[i];
  }
  
  //Compute FFT
  arm_rfft_fast_f32(mfcc->rfft, mfcc->frame, mfcc->buffer, 0);

  //Convert to power spectrum
  //frame is stored as [real0, realN/2-1, real1, im1, real2, im2, ...]
  int32_t half_dim = mfcc->frame_len_padded/2;
  float first_energy = mfcc->buffer[0] * mfcc->buffer[0],
        last_energy =  mfcc->buffer[1] * mfcc->buffer[1];  // handle this special case
  for (i = 1; i < half_dim; i++) {
    float real = mfcc->buffer[i*2], im = mfcc->buffer[i*2 + 1];
    mfcc->buffer[i] = real*real + im*im;
  }
  mfcc->buffer[0] = first_energy;
  mfcc->buffer[half_dim] = last_energy;  
 
  float sqrt_data;
  //Apply mel filterbanks
  for (bin = 0; bin < NUM_FBANK_BINS; bin++) {
    j = 0;
    float mel_energy = 0;
    int32_t first_index = mfcc->fbank_filter_first[bin];
    int32_t last_index = mfcc->fbank_filter_last[bin];

    for (i = first_index; i<=last_index; i++) {
      arm_status status = arm_sqrt_f32(mfcc->buffer[i], &sqrt_data);
      if (status != ARM_MATH_SUCCESS) {
        #if AS_DEBUG
        UART_Printf(debug, "arm_sqrt_f32 status: %d\r\n", status);
        #endif
        return status;
      }
      mel_energy += (sqrt_data) * mfcc->mel_fbank[bin][j++];
    }

    mfcc->mel_energies[bin] = mel_energy;

    //avoid log of zero
    if (mel_energy == 0.0)
      mfcc->mel_energies[bin] = FLT_MIN;
  }

  //Take log
  for (bin = 0; bin < NUM_FBANK_BINS; bin++)
    mfcc->mel_energies[bin] = logf(mfcc->mel_energies[bin]);

  //Take DCT. Uses matrix mul.
  for (i = 0; i < mfcc->num_mfcc_features; i++) {
    float sum = 0.0;
    for (j = 0; j < NUM_FBANK_BINS; j++) {
      sum += mfcc->dct_matrix[i*NUM_FBANK_BINS+j] * mfcc->mel_energies[j];
    }

    //Input is Qx.mfcc_dec_bits (from quantization step)
    //Mehrdad: commented these to match X86 result
    // sum *= (0x1<<mfcc->dec_bits);
    // sum = round(sum); 
    // if(sum >= 127)
    //   mfcc_out[i] = 127;
    // else if(sum <= -128)
    //   mfcc_out[i] = -128;
    // else
    mfcc_out[i] = sum; 
  }
  return 0;
}

