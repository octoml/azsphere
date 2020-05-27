#ifndef AS_KWS_H_
#define AS_KWS_H_

#include "mfcc.h"

#define NUM_FRAMES        49
#define NUM_MFCC_COEFFS   10
#define SAMP_FREQ         16000
#define FRAME_LEN_MS      40
#define MFCC_DEC_BITS     1
#define FRAME_LEN         ((int16_t)(SAMP_FREQ * 0.001 * FRAME_LEN_MS))
#define FRAME_SHIFT_MS    20
#define FRAME_SHIFT       ((int16_t)(SAMP_FREQ * 0.001 * FRAME_SHIFT_MS))

class KWS {

public:
  KWS(int16_t* audio_data_buffer);
  ~KWS();
  void extract_features();
  int16_t* audio_buffer;
  q7_t *mfcc_buffer;
  // q7_t *output;
  // q7_t *predictions;
  // q7_t *averaged_output;
  int num_frames;
  int num_mfcc_features;
  int frame_len;
  int frame_shift;
  int num_out_classes;
  int audio_block_size;
  int audio_buffer_size;
  void init_kws();
  
private:
  MFCC *mfcc;
  int mfcc_buffer_size;
  int recording_win;
  int sliding_window_len;

};

#endif /* AS_KWS_H_ */