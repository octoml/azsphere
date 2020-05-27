#include <stdlib.h>
#include "kws.h"

KWS::KWS(int16_t* audio_data_buffer) {
  audio_buffer = audio_data_buffer;
  recording_win = NUM_FRAMES;
  sliding_window_len = 1;
  init_kws();
}

KWS::~KWS() {

}

void KWS::init_kws()
{
  num_mfcc_features = NUM_MFCC_COEFFS;
  num_frames = NUM_FRAMES;
  frame_len = FRAME_LEN;
  frame_shift = FRAME_SHIFT;
  int mfcc_dec_bits = MFCC_DEC_BITS;
  num_out_classes = 12;
  //TODO: fix this
  // mfcc = new MFCC(num_mfcc_features, frame_len, mfcc_dec_bits);
  mfcc->init(num_mfcc_features, frame_len, mfcc_dec_bits);
  mfcc_buffer = (q7_t *)malloc(num_frames * num_mfcc_features * sizeof(q7_t));
  // output = new q7_t[num_out_classes];
  // averaged_output = new q7_t[num_out_classes];
  // predictions = new q7_t[sliding_window_len*num_out_classes];
  // audio_block_size = recording_win*frame_shift;
  // audio_buffer_size = audio_block_size + frame_len - frame_shift;
}

void KWS::extract_features() {
  if(num_frames>recording_win) {
    //move old features left 
    memmove(mfcc_buffer,mfcc_buffer+(recording_win*num_mfcc_features),(num_frames-recording_win)*num_mfcc_features);
  }
  //compute features only for the newly recorded audio
  int32_t mfcc_buffer_head = (num_frames-recording_win)*num_mfcc_features; 
  for (uint16_t f = 0; f < recording_win; f++) {
    mfcc->mfcc_compute(audio_buffer+(f*frame_shift),&mfcc_buffer[mfcc_buffer_head]);
    mfcc_buffer_head += num_mfcc_features;
  }
}