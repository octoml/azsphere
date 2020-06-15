#include <stdlib.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>

#include "include/kws.h"
#include "include/config.h"

kws* KWS_Init()
{
  kws* new_kws = (kws *)malloc(sizeof(kws));
  if (!new_kws) {
    return NULL;
  }

  new_kws->recording_win = REC_WINDOW;
  new_kws->sliding_window_len = 1;

  new_kws->num_mfcc_features = NUM_MFCC_COEFFS;
  new_kws->num_frames = NUM_FRAMES;
  new_kws->frame_len = FRAME_LEN;
  new_kws->frame_shift = FRAME_SHIFT;
  int dec_bits = MFCC_DEC_BITS;
  new_kws->num_out_classes = 12;

  #if AS_DEBUG
  UART_Printf(debug, "frame_len: %d\r\n", new_kws->frame_len);
  #endif
  new_kws->mfcc_ = MFCC_Init(new_kws->num_mfcc_features, new_kws->frame_len, dec_bits);
  if (!new_kws->mfcc_) {
    return NULL;
  }
  #if AS_DEBUG
  UART_Print(debug, "MFCC Init Done!\n");
  #endif

  new_kws->mfcc_buffer_size = new_kws->recording_win * new_kws->num_mfcc_features;
  new_kws->mfcc_buffer = (float *)malloc(new_kws->mfcc_buffer_size * sizeof(float));
  if (!new_kws->mfcc_buffer) {
    return NULL;
  }
  memset(new_kws->mfcc_buffer, 0, new_kws->mfcc_buffer_size * sizeof(float));
  return new_kws;
}

int8_t KWS_Extract_Features(kws *kws, int16_t *audio) {
  #if AS_DEBUG
  UART_Print(debug, "Extracting feature started\n");
  #endif
  if(kws->num_frames > kws->recording_win) {
    //move old features left 
    memmove(kws->mfcc_buffer, kws->mfcc_buffer+(kws->recording_win*kws->num_mfcc_features), (kws->num_frames-kws->recording_win)*kws->num_mfcc_features);
  }
  //compute features only for the newly recorded audio
  int32_t mfcc_buffer_head = 0;
  int8_t status;
  int16_t* audio_buffer;
  int audio_buffer_counter = 0;
  for (uint16_t f = 0; f<kws->recording_win; f++) {
    if ((f % kws->num_frames) == 0) {
      //update audio buffer
      audio_buffer = (int16_t *)(audio + (audio_buffer_counter * kws->frame_len/2));
      audio_buffer_counter += kws->num_frames;
    }
    #if AS_DEBUG
    UART_Printf(debug, "MFCC_Compute: %d\n", f);
    #endif
    status = MFCC_Compute(kws->mfcc_, audio_buffer + ((f%kws->num_frames)*kws->frame_shift), &kws->mfcc_buffer[mfcc_buffer_head]);
    if (status < 0){
      #if AS_DEBUG
      UART_Printf(debug, "MFCC_Compute failed!\r\n");
      #endif
      return -1;
    }
    mfcc_buffer_head += kws->num_mfcc_features;
  }
  return 0;
}

int8_t KWS_Extract_Features_Frame(kws *kws, int16_t *audio_buffer, int32_t mfcc_buffer_head) {
  int8_t status;
  int32_t head = mfcc_buffer_head;
  for (uint16_t f = 0; f<kws->num_frames; f++) {
    #if AS_DEBUG
    UART_Printf(debug, "MFCC_Compute: %d\n", f);
    #endif
    status = MFCC_Compute(kws->mfcc_, audio_buffer + (f * kws->frame_shift), &kws->mfcc_buffer[head]);
    if (status < 0){
      #if AS_DEBUG
      UART_Printf(debug, "MFCC_Compute failed!\r\n");
      #endif
      return -1;
    }
    head += kws->num_mfcc_features;
    if (head >= (kws->num_mfcc_features*kws->recording_win)) {
      break;
    }
  }
  return 0;
}