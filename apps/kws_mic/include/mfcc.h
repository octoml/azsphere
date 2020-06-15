#ifndef AS_MFCC_H_
#define AS_MFCC_H_

#include <stdint.h>
#include <string.h>
#include <math.h>
#include "arm_math.h"

#define SAMP_FREQ           16000
#define NUM_FBANK_BINS      40
#define MEL_LOW_FREQ        20
#define MEL_HIGH_FREQ       4000
#define M_2PI               6.283185307179586476925286766559005
#define M_PI                (double) M_2PI/2

typedef struct mfcc{
  int num_mfcc_features;
  int frame_len;
  int frame_len_padded;
  int dec_bits;
  float * frame;
  float * buffer;
  float * mel_energies;
  float * window_func;
  int32_t * fbank_filter_first;
  int32_t * fbank_filter_last;
  float ** mel_fbank;
  float * dct_matrix;
  arm_rfft_fast_instance_f32 * rfft;
} mfcc;

mfcc* MFCC_Init(int num_mfcc_features, int frame_len, int mfcc_dec_bits);
void MFCC_Close(mfcc* mfcc);
float * MFCC_Create_DCT_Matrix(int32_t input_length, int32_t coefficient_count);
float ** MFCC_Create_MEL_FBank();
int8_t MFCC_Compute(mfcc* mfcc, const int16_t* data, float* mfcc_out);


static inline float InverseMelScale(float mel_freq) {
  return 700.0f * (expf (mel_freq / 1127.0f) - 1.0f);
}

static inline float MelScale(float freq) {
  return 1127.0f * logf (1.0f + freq / 700.0f);
}

#endif  /* AS_MFCC_H_ */
