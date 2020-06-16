/* Explicitly declare posix_memalign function */
#if _POSIX_C_SOURCE < 200112L
#undef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

/*! Support low-level debugging in MISRA-C runtime */
#define TVM_CRT_DEBUG 0

/*! Maximum supported dimension in NDArray */
#define TVM_CRT_MAX_NDIM 6
/*! Maximum supported arguments in generated functions */
#define TVM_CRT_MAX_ARGS 10

/*! Maximum inputs in a GraphRuntimeNode */
#define GRAPH_RUNTIME_NODE_MAX_INPUTS       4               //default 300
/*! Maximum supported contexts in a GraphRuntime */
#define GRAPH_RUNTIME_MAX_CONTEXTS          1               //default 1
/*! Maximum supported nodes in a GraphRuntime */
#define GRAPH_RUNTIME_MAX_NODES             62              //default 400
/*! Maximum input nodes in a GraphRuntime */
#define GRAPH_RUNTIME_MAX_INPUT_NODES       39              //default 300
/*! Maximum nodes in a GraphRuntime for quick entry indexing */
#define GRAPH_RUNTIME_MAX_NODE_ROW_PTR      63              //default 300
/*! Maximum output entries in a GraphRuntime */
#define GRAPH_RUNTIME_MAX_OUTPUTS           1               //default 300

// #define GRAPH_RUNTIME_NODE_NAME_MAX         5               //default 80
// #define GRAPH_RUNTIME_FUNC_NAME_MAX         110             //default 120
// #define GRAPH_RUNTIME_CTRL_DEPTH_SIZE       1               //default 200

#include <runtime/crt/crt_runtime_api.c>
#include <runtime/crt/crt_backend_api.c>
#include <runtime/crt/graph_runtime.c>
#include <runtime/crt/load_json.c>
#include <runtime/crt/ndarray.c>