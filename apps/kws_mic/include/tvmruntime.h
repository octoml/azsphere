#ifndef AS_TVMRUNTIME_H_
#define AS_TVMRUNTIME_H_

#include "include/bundle.h"

#define in_dim0     1
#define in_dim1     49
#define in_dim2     10

#define out_dim0    1
#define out_dim1    12

// extern int* tvm_handle;

int* TVMInit(char *params_data, uint64_t params_size, char *graph_data);
int TVMCallback(int* handle, void* inputData, float* result);
uint8_t TVMMaxIndex(float* tvmOutput);

#endif /* AS_TVMRUNTIME_H_ */