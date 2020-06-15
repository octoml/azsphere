#ifndef AS_KWS_H_
#define AS_KWS_H_

#include <stdint.h>
#include "mfcc.h"

#define NUM_FRAMES        3
#define REC_WINDOW        49
#define FRAME_LEN_MS      40
#define FRAME_SHIFT_MS    20
#define NUM_MFCC_COEFFS   10
#define SAMP_FREQ         16000
#define MFCC_DEC_BITS     1
#define FRAME_LEN         ((int16_t)(SAMP_FREQ * 0.001 * FRAME_LEN_MS))
#define FRAME_SHIFT       ((int16_t)(SAMP_FREQ * 0.001 * FRAME_SHIFT_MS))

typedef struct kws{
  float* mfcc_buffer;
  int num_frames;
  int num_mfcc_features;
  int frame_len;
  int frame_shift;
  int num_out_classes;
  int audio_block_size;
  int audio_buffer_size;
  int mfcc_buffer_size;
  int recording_win;
  int sliding_window_len;
  mfcc* mfcc_;
} kws;

kws* KWS_Init();
int8_t KWS_Extract_Features(kws *kws, int16_t *audio);
int8_t KWS_Extract_Features_Frame(kws *kws, int16_t *audio_buffer, int32_t mfcc_buffer_head);

#endif /* AS_KWS_H_ */