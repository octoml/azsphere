#ifndef TVM_APPS_BUNDLE_DEPLOY_BUNDLE_H_
#define TVM_APPS_BUNDLE_DEPLOY_BUNDLE_H_

#include <tvm/runtime/c_runtime_api.h>

TVM_DLL void * tvm_runtime_create(const char * json_data,
                                  const char * params_data,
                                  const uint64_t params_size);

TVM_DLL void tvm_runtime_destroy(void * runtime);

TVM_DLL void tvm_runtime_set_input(void * runtime,
                                   const char * name, 
                                   DLTensor * tensor);

TVM_DLL void tvm_runtime_run(void * runtime);

TVM_DLL void tvm_runtime_get_output(void * runtime,
                                    int32_t index, 
                                    DLTensor * tensor);

#endif /* TVM_APPS_BUNDLE_DEPLOY_BUNDLE_H_ */