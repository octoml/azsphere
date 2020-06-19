#include <stdio.h>
#include <stdlib.h>

#include "include/bundle.h"
#include "include/runtime.h"

TVM_DLL void * tvm_runtime_create(const char * json_data,
                                  const char * params_data,
                                  const uint64_t params_size) {
  int64_t device_type = kDLCPU;
  int64_t device_id = 0;

  TVMByteArray params;
  params.data = params_data;
  params.size = params_size;

  TVMContext ctx;
  ctx.device_type = (DLDeviceType)device_type;
  ctx.device_id = device_id;

  // declare pointers
  void * (*SystemLibraryCreate)();
  TVMGraphRuntime * (*TVMGraphRuntimeCreate)(const char *, const TVMModuleHandle, const TVMContext *);
  int (*TVMGraphRuntime_LoadParams)(TVMModuleHandle, const char *, const uint32_t);

  // get pointers
  TVMFuncGetGlobal("runtime.SystemLib", (TVMFunctionHandle*)&SystemLibraryCreate);
  TVMFuncGetGlobal("tvm.graph_runtime.create", (TVMFunctionHandle*)&TVMGraphRuntimeCreate);

  // run modules
  TVMModuleHandle mod_syslib = SystemLibraryCreate();
  TVMModuleHandle mod = TVMGraphRuntimeCreate(json_data, mod_syslib, &ctx);
  TVMModGetFunction(mod, "load_params", 0, (TVMFunctionHandle*)&TVMGraphRuntime_LoadParams);
  TVMGraphRuntime_LoadParams(mod, params.data, params.size);
  
  return mod;
}

TVM_DLL void tvm_runtime_destroy(void * runtime) {
  void (*TVMGraphRuntimeRelease)(TVMModuleHandle *);
  TVMFuncGetGlobal("tvm.graph_runtime.release", (TVMFunctionHandle*)&TVMGraphRuntimeRelease);
  TVMGraphRuntimeRelease(&runtime);
}

TVM_DLL void tvm_runtime_set_input(void * runtime, const char * name, DLTensor * tensor) {
  void (*TVMGraphRuntime_SetInput)(TVMModuleHandle, const char *, DLTensor*);
  TVMFuncGetGlobal("tvm.graph_runtime.set_input", (TVMFunctionHandle*)&TVMGraphRuntime_SetInput);
  TVMGraphRuntime_SetInput(runtime, name, tensor);
}

TVM_DLL void tvm_runtime_run(void * runtime) {
  void (*TVMGraphRuntime_Run)(TVMModuleHandle runtime);
  TVMFuncGetGlobal("tvm.graph_runtime.run", (TVMFunctionHandle*)&TVMGraphRuntime_Run);
  TVMGraphRuntime_Run(runtime);
}

TVM_DLL void tvm_runtime_get_output(void * runtime, int32_t index, DLTensor * tensor) {
  int (*TVMGraphRuntime_GetOutput)(TVMModuleHandle, const int32_t, DLTensor *);
  TVMFuncGetGlobal("tvm.graph_runtime.get_output", (TVMFunctionHandle*)&TVMGraphRuntime_GetOutput);
  TVMGraphRuntime_GetOutput(runtime, index, tensor);
}