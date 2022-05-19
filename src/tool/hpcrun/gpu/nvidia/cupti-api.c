// -*-Mode: C++;-*- // technically C99

// * BeginRiceCopyright *****************************************************
//
// --------------------------------------------------------------------------
// Part of HPCToolkit (hpctoolkit.org)
//
// Information about sources of support for research and development of
// HPCToolkit is at 'hpctoolkit.org' and in 'README.Acknowledgments'.
// --------------------------------------------------------------------------
//
// Copyright ((c)) 2002-2022, Rice University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of Rice University (RICE) nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// This software is provided by RICE and contributors "as is" and any
// express or implied warranties, including, but not limited to, the
// implied warranties of merchantability and fitness for a particular
// purpose are disclaimed. In no event shall RICE or contributors be
// liable for any direct, indirect, incidental, special, exemplary, or
// consequential damages (including, but not limited to, procurement of
// substitute goods or services; loss of use, data, or profits; or
// business interruption) however caused and on any theory of liability,
// whether in contract, strict liability, or tort (including negligence
// or otherwise) arising in any way out of the use of this software, even
// if advised of the possibility of such damage.
//
// ******************************************************* EndRiceCopyright *

//***************************************************************************
//
// File:
//   cupti-api.c
//
// Purpose:
//   implementation of wrapper around NVIDIA's CUPTI performance tools API
//
//***************************************************************************

//***************************************************************************
// system includes
//***************************************************************************

#include <errno.h>     // errno
#include <fcntl.h>     // open
#include <sys/stat.h>  // mkdir
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>
#include <math.h>
#include <time.h>

#ifndef HPCRUN_STATIC_LINK
#include <dlfcn.h>
#undef _GNU_SOURCE
#define _GNU_SOURCE
#include <link.h>          // dl_iterate_phdr
#include <linux/limits.h>  // PATH_MAX
#include <string.h>        // strstr
#endif



//***************************************************************************
// workaround for cuptiFlushAll hang
//***************************************************************************

#define CUPTI_FLUSH_HANG_WORKAROUND 1
#define CUPTI_FLUSH_HANG_WORKAROUND_TEST 0

#if CUPTI_FLUSH_HANG_WORKAROUND
#include <setjmp.h>
#include <signal.h>
#include <unistd.h>

#include <messages/messages.h>
#include <utilities/linuxtimer.h>


//----------------------------------------------
// flush alarm enabled
//----------------------------------------------
__thread jmp_buf flush_jump_buf;

int flush_signal;

#define FLUSH_ALARM_SECONDS 4

#define FLUSH_ALARM_INIT() \
  linuxtimer_t flush_alarm

#define FLUSH_ALARM_SIGALLOC() \
  flush_signal = linuxtimer_newsignal()

#define FLUSH_ALARM_SET()						\
  linuxtimer_create(&flush_alarm, CLOCK_REALTIME, flush_signal);	\
  monitor_sigaction(linuxtimer_getsignal(&flush_alarm),			\
		    &flush_alarm_handler, 0, NULL);			\
  linuxtimer_set(&flush_alarm, FLUSH_ALARM_SECONDS, 0, 0)

#define FLUSH_ALARM_CLEAR()			\
  linuxtimer_set(&flush_alarm, 0, 0, 0)

#define FLUSH_ALARM_FIRED() \
  setjmp(flush_jump_buf)

#define FLUSH_ALARM_FINI()			\
  linuxtimer_delete(&flush_alarm)

static int
flush_alarm_handler(int sig, siginfo_t* siginfo, void* context)
{
  STDERR_MSG("hpcrun: NVIDIA's CUPTI event flush didn't return; some GPU event data may be lost.");
  longjmp(flush_jump_buf, 1);
  return 0; /* keep compiler happy, but can't get here */
}

#else

//----------------------------------------------
// flush alarm disabled
//----------------------------------------------
#define FLUSH_ALARM_INIT()
#define FLUSH_ALARM_SIGALLOC()
#define FLUSH_ALARM_SET()
#define FLUSH_ALARM_CLEAR()
#define FLUSH_ALARM_FIRED() 0
#define FLUSH_ALARM_FINI()

#endif

#if CUPTI_FLUSH_HANG_WORKAROUND_TEST
#define FLUSH_ALARM_TEST()			\
  sleep(20)
#else
#define FLUSH_ALARM_TEST()
#endif



//***************************************************************************
// local includes
//***************************************************************************

#include <include/gpu-binary.h>

#include <lib/prof-lean/spinlock.h>

#include <hpcrun/files.h>
#include <hpcrun/hpcrun_stats.h>
#include <hpcrun/main.h> // hpcrun_force_dlopen
#include <hpcrun/safe-sampling.h>

#include <hpcrun/gpu/gpu-activity-channel.h>
#include <hpcrun/gpu/gpu-application-thread-api.h>
#include <hpcrun/gpu/gpu-monitoring-thread-api.h>
#include <hpcrun/gpu/gpu-correlation-channel.h>
#include <hpcrun/gpu/gpu-correlation-id.h>
#include <hpcrun/gpu/gpu-op-placeholders.h>
#include <hpcrun/gpu/gpu-cct.h>
#include <hpcrun/gpu/gpu-operation-multiplexer.h>
#include <hpcrun/gpu/gpu-range.h>
#include <hpcrun/gpu/gpu-metrics.h>

#include <hpcrun/ompt/ompt-device.h>

#include <hpcrun/sample-sources/libdl.h>
#include <hpcrun/sample-sources/nvidia.h>

#include <hpcrun/utilities/hpcrun-nanotime.h>

#include <hpcrun/thread_data.h>
#include <hpcrun/tool_state.h>

#include "cuda-api.h"
#include "cupti-api.h"
#include "cupti-gpu-api.h"
#include "cubin-hash-map.h"
#include "cubin-id-map.h"

//#include "sample_sources_all.h"

#ifdef NEW_CUPTI
#include "cubin-crc-map.h"
#include "cupti-range.h"
#include "cupti-subscribers.h"
#include "cupti-pc-sampling-api.h"
#include "cupti-unwind-map.h"
#include "cupti-cct-trie.h"
#include "cupti-cct-map.h"
#endif

//******************************************************************************
// macros
//******************************************************************************


#define DEBUG 0
#include <hpcrun/gpu/gpu-print.h>


#define CUPTI_LIBRARY_LOCATION "/lib64/libcupti.so"
#define CUPTI_PATH_FROM_CUDA "extras/CUPTI"


#define HPCRUN_CUPTI_ACTIVITY_BUFFER_SIZE (16 * 1024 * 1024)
#define HPCRUN_CUPTI_ACTIVITY_BUFFER_ALIGNMENT (8)

#define CUPTI_FN_NAME(f) DYN_FN_NAME(f)

#define CUPTI_FN(fn, args) \
  static CUptiResult (*CUPTI_FN_NAME(fn)) args

#define HPCRUN_CUPTI_CALL(fn, args)  \
{  \
  CUptiResult status = CUPTI_FN_NAME(fn) args;  \
  if (status != CUPTI_SUCCESS) {  \
    cupti_error_report(status, #fn);  \
  }  \
}

#define HPCRUN_CUPTI_CALL_NOERROR(fn, args)  \
{  \
  CUPTI_FN_NAME(fn) args;  \
}


#define DISPATCH_CALLBACK(fn, args) if (fn) fn args

#define FORALL_CUPTI_ROUTINES(macro)             \
  macro(cuptiActivityConfigurePCSampling)        \
  macro(cuptiActivityDisable)                    \
  macro(cuptiActivityDisableContext)             \
  macro(cuptiActivityEnable)                     \
  macro(cuptiActivityEnableContext)              \
  macro(cuptiActivityFlushAll)                   \
  macro(cuptiActivitySetAttribute)               \
  macro(cuptiActivityGetNextRecord)              \
  macro(cuptiActivityGetNumDroppedRecords)       \
  macro(cuptiActivityPopExternalCorrelationId)   \
  macro(cuptiActivityPushExternalCorrelationId)  \
  macro(cuptiActivityRegisterCallbacks)          \
  macro(cuptiGetTimestamp)                       \
  macro(cuptiEnableDomain)                       \
  macro(cuptiEnableCallback)                     \
  macro(cuptiFinalize)                           \
  macro(cuptiGetResultString)                    \
  macro(cuptiSubscribe)                          \
  macro(cuptiUnsubscribe)



//******************************************************************************
// types
//******************************************************************************

typedef void (*cupti_error_callback_t)
(
 const char *type,
 const char *fn,
 const char *error_string
);


typedef CUptiResult (*cupti_activity_enable_t)
(
 CUpti_ActivityKind activity
);


typedef cct_node_t *(*cupti_correlation_callback_t)
(
);


typedef void (*cupti_load_callback_t)
(
 CUcontext context,
 uint32_t cubin_id,
 const void *cubin,
 size_t cubin_size
);


typedef struct {
  CUpti_BuffersCallbackRequestFunc buffer_request;
  CUpti_BuffersCallbackCompleteFunc buffer_complete;
} cupti_activity_buffer_state_t;

//******************************************************************************
// forward declarations
//******************************************************************************

static void
cupti_error_callback_dummy
(
 const char *type,
 const char *fn,
 const char *error_string
);


static cct_node_t *
cupti_correlation_callback_dummy
(
 uint64_t id
);



//******************************************************************************
// static data
//******************************************************************************

static int cupti_backoff_base = 4;
static int cupti_correlation_threshold = -1;
static bool cupti_sync_yield = false;
static bool cupti_fast_unwind = false;

static spinlock_t files_lock = SPINLOCK_UNLOCKED;

static __thread bool cupti_thread_activity_flag = false;
static __thread bool cupti_runtime_api_flag = false;
static __thread cct_node_t *cupti_kernel_ph = NULL;
static __thread cct_node_t *cupti_trace_ph = NULL;

#ifdef NEW_CUPTI
static __thread cct_node_t *cupti_prev_api_node = NULL;
static __thread cct_node_t *cupti_prev_kernel_node = NULL;
static __thread cct_node_t *cupti_prev_prev_kernel_node = NULL;
#endif

static __thread uint64_t cupti_runtime_correlation_id = 0;
static __thread uint64_t cupti_driver_correlation_id = 0;

static bool cupti_correlation_enabled = false;

static cupti_correlation_callback_t cupti_correlation_callback =
  cupti_correlation_callback_dummy;

static cupti_error_callback_t cupti_error_callback =
  cupti_error_callback_dummy;

static cupti_activity_buffer_state_t cupti_activity_enabled = { 0, 0 };

static cupti_load_callback_t cupti_load_callback = 0;

static cupti_load_callback_t cupti_unload_callback = 0;

static CUpti_SubscriberHandle cupti_subscriber;

#ifdef NEW_CUPTI
static uint64_t CUPTI_CORRELATION_ID_NULL = 0;

static uint64_t slow_unwinds = 0;
static uint64_t fast_unwinds = 0;
static uint64_t total_unwinds = 0;
static uint64_t correct_unwinds = 0;
#endif

//----------------------------------------------------------
// cupti function pointers for late binding
//----------------------------------------------------------

CUPTI_FN
(
 cuptiActivityEnable,
 (
  CUpti_ActivityKind kind
 )
);


CUPTI_FN
(
 cuptiActivityDisable,
 (
 CUpti_ActivityKind kind
 )
);


CUPTI_FN
(
 cuptiActivityEnableContext,
 (
  CUcontext context,
  CUpti_ActivityKind kind
 )
);


CUPTI_FN
(
 cuptiActivityDisableContext,
 (
  CUcontext context,
  CUpti_ActivityKind kind
 )
);


CUPTI_FN
(
 cuptiActivityConfigurePCSampling,
 (
  CUcontext ctx,
  CUpti_ActivityPCSamplingConfig *config
 )
);


CUPTI_FN
(
 cuptiActivityRegisterCallbacks,
 (
  CUpti_BuffersCallbackRequestFunc funcBufferRequested,
  CUpti_BuffersCallbackCompleteFunc funcBufferCompleted
 )
);


CUPTI_FN
(
 cuptiActivityPushExternalCorrelationId,
 (
  CUpti_ExternalCorrelationKind kind,
  uint64_t id
 )
);


CUPTI_FN
(
 cuptiActivityPopExternalCorrelationId,
 (
  CUpti_ExternalCorrelationKind kind,
  uint64_t *lastId
 )
);


CUPTI_FN
(
 cuptiActivityGetNextRecord,
 (
  uint8_t* buffer,
  size_t validBufferSizeBytes,
  CUpti_Activity **record
 )
);


CUPTI_FN
(
 cuptiActivityGetNumDroppedRecords,
 (
  CUcontext context,
  uint32_t streamId,
  size_t *dropped
 )
);


CUPTI_FN
(
  cuptiActivitySetAttribute,
  (
   CUpti_ActivityAttribute attribute,
   size_t *value_size,
   void *value
  )
);


CUPTI_FN
(
 cuptiActivityFlushAll,
 (
  uint32_t flag
 )
);


CUPTI_FN
(
 cuptiGetTimestamp,
 (
  uint64_t *timestamp
 )
);


CUPTI_FN
(
 cuptiGetTimestamp,
 (
  uint64_t* timestamp
 )
);


CUPTI_FN
(
 cuptiEnableDomain,
 (
  uint32_t enable,
  CUpti_SubscriberHandle subscriber,
  CUpti_CallbackDomain domain
 )
);


CUPTI_FN
(
 cuptiEnableCallback,
 (
  uint32_t enable,
  CUpti_SubscriberHandle subscriber,
  CUpti_CallbackDomain domain,
  CUpti_CallbackId cbid
 )
);


CUPTI_FN
(
 cuptiFinalize,
 (
  void
 )
);


CUPTI_FN
(
 cuptiGetResultString,
 (
  CUptiResult result,
  const char **str
 )
);


CUPTI_FN
(
 cuptiSubscribe,
 (
  CUpti_SubscriberHandle *subscriber,
  CUpti_CallbackFunc callback,
  void *userdata
 )
);


CUPTI_FN
(
 cuptiUnsubscribe,
 (
  CUpti_SubscriberHandle subscriber
 )
);



//******************************************************************************
// private operations
//******************************************************************************

#ifndef HPCRUN_STATIC_LINK
int
cuda_path
(
 struct dl_phdr_info *info,
 size_t size,
 void *data
)
{
  char *buffer = (char *) data;
  const char *suffix = strstr(info->dlpi_name, "libcudart");
  if (suffix) {
    // CUDA library organization after 9.0
    suffix = strstr(info->dlpi_name, "targets");
    if (!suffix) {
      // CUDA library organization in 9.0 or earlier
      suffix = strstr(info->dlpi_name, "lib64");
    }
  }
  if (suffix){
    int len = suffix - info->dlpi_name;
    strncpy(buffer, info->dlpi_name, len);
    buffer[len] = 0;
    return 1;
  }
  return 0;
}


static void
cupti_set_default_path(char *buffer)
{
  strcpy(buffer, CUPTI_INSTALL_PREFIX "/" CUPTI_LIBRARY_LOCATION);
}


static int
library_path_resolves(const char *buffer)
{
  struct stat sb;
  return stat(buffer, &sb) == 0;
}


const char *
cupti_path
(
 void
)
{
  const char *path = "libcupti.so";

  static char buffer[PATH_MAX];
  buffer[0] = 0;

#ifdef NEW_CUPTI
  // XXX(Keren): Don't use the default cupti library under CUDA_HOME because
  // since NVIDIA delivers us independent cupti libraries that contain bug
  // fixes.
  cupti_set_default_path(buffer);
  if (library_path_resolves(buffer)) {
    fprintf(stderr, "NOTE: Using builtin path for NVIDIA's CUPTI tools "
      "library %s.\n", buffer);
    path = buffer;
  }
#else
  int resolved = 0;
  // open an NVIDIA library to find the CUDA path with dl_iterate_phdr
  // note: a version of this file with a more specific name may
  // already be loaded. thus, even if the dlopen fails, we search with
  // dl_iterate_phdr.
  void *h = monitor_real_dlopen("libcudart.so", RTLD_LOCAL | RTLD_LAZY);

  if (dl_iterate_phdr(cuda_path, buffer)) {
    // invariant: buffer contains CUDA home
    int zero_index = strlen(buffer);
    strcat(buffer, CUPTI_LIBRARY_LOCATION);

    if (library_path_resolves(buffer)) {
      path = buffer;
      resolved = 1;
    } else {
      buffer[zero_index] = 0;
      strcat(buffer, CUPTI_PATH_FROM_CUDA CUPTI_LIBRARY_LOCATION);

      if (library_path_resolves(buffer)) {
        path = buffer;
        resolved = 1;
      } else {
        buffer[zero_index - 1] = 0;
        fprintf(stderr, "NOTE: CUDA root at %s lacks a copy of NVIDIA's CUPTI "
          "tools library.\n", buffer);
      }
    }
  }

  if (!resolved) {
    cupti_set_default_path(buffer);
    if (library_path_resolves(buffer)) {
      fprintf(stderr, "NOTE: Using builtin path for NVIDIA's CUPTI tools "
        "library %s.\n", buffer);
      path = buffer;
      resolved = 1;
    }
  }

  if (h) monitor_real_dlclose(h);
#endif

  return path;
}

#endif

int
cupti_bind
(
 void
)
{
#ifndef HPCRUN_STATIC_LINK
  // dynamic libraries only availabile in non-static case
  hpcrun_force_dlopen(true);
  CHK_DLOPEN(cupti, cupti_path(), RTLD_NOW | RTLD_GLOBAL);
  hpcrun_force_dlopen(false);

#define CUPTI_BIND(fn) \
  CHK_DLSYM(cupti, fn);

  FORALL_CUPTI_ROUTINES(CUPTI_BIND);

#undef CUPTI_BIND

  return DYNAMIC_BINDING_STATUS_OK;
#else
  return DYNAMIC_BINDING_STATUS_ERROR;
#endif // ! HPCRUN_STATIC_LINK
}


static cct_node_t *
cupti_correlation_callback_dummy // __attribute__((unused))
(
 uint64_t id
)
{
  return NULL;
}


static void
cupti_error_callback_dummy // __attribute__((unused))
(
 const char *type,
 const char *fn,
 const char *error_string
)
{
  
  EEMSG("FATAL: hpcrun failure: failure type = %s, "
      "function %s failed with error %s", type, fn, error_string);
  EEMSG("See the 'FAQ and Troubleshooting' chapter in the HPCToolkit manual for guidance");
  exit(1);
}


static void
cupti_error_report
(
 CUptiResult error,
 const char *fn
)
{
  const char *error_string;
  CUPTI_FN_NAME(cuptiGetResultString)(error, &error_string);
  cupti_error_callback("CUPTI result error", fn, error_string);
}


//******************************************************************************
// private operations
//******************************************************************************

static bool
cupti_write_cubin
(
 const char *file_name,
 const void *cubin,
 size_t cubin_size
)
{
  int fd;
  errno = 0;
  fd = open(file_name, O_WRONLY | O_CREAT | O_EXCL, 0644);
  if (errno == EEXIST) {
    close(fd);
    return true;
  }
  if (fd >= 0) {
    // Success
    if (write(fd, cubin, cubin_size) != cubin_size) {
      close(fd);
      return false;
    } else {
      close(fd);
      return true;
    }
  } else {
    // Failure to open is a fatal error.
    hpcrun_abort("hpctoolkit: unable to open file: '%s'", file_name);
    return false;
  }
}


#ifdef NEW_CUPTI

void
cupti_load_callback_cuda
(
 CUcontext context,
 uint32_t cubin_id,
 const void *cubin,
 size_t cubin_size
)
{
  TMSG(CUPTI, "Load cubin %u", cubin_id);

  // Compute hash for cubin and store it into a map
  uint64_t cubin_crc = cupti_cubin_crc_get(cubin, cubin_size);

  // Create file name
  char file_name[PATH_MAX];
  size_t used = 0;
  used += sprintf(&file_name[used], "%s", hpcrun_files_output_directory());
  used += sprintf(&file_name[used], "%s", "/" GPU_BINARY_DIRECTORY "/");
  mkdir(file_name, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  used += sprintf(&file_name[used], "%"PRIu64"", cubin_crc);
  used += sprintf(&file_name[used], "%s", GPU_BINARY_SUFFIX);
  TMSG(CUDA_CUBIN, "cubin_crc %s", file_name);

  // Write a file if does not exist
  bool file_flag;
  spinlock_lock(&files_lock);
  file_flag = cupti_write_cubin(file_name, cubin, cubin_size);
  spinlock_unlock(&files_lock);

  if (file_flag) {
    char device_file[PATH_MAX];
    sprintf(device_file, "%s", file_name);
    uint32_t hpctoolkit_module_id;
    load_module_t *module = NULL;
    hpcrun_loadmap_lock();
    if ((module = hpcrun_loadmap_findByName(device_file)) == NULL) {
      hpctoolkit_module_id = hpcrun_loadModule_add(device_file);
    } else {
      hpctoolkit_module_id = module->id;
    }
    hpcrun_loadmap_unlock();
    TMSG(CUDA_CUBIN, "cubin_crc %d -> hpctoolkit_module_id %d", cubin_crc, hpctoolkit_module_id);
    cubin_crc_map_entry_t *crc_entry = cubin_crc_map_lookup(cubin_crc);
    if (crc_entry == NULL) {
      Elf_SymbolVector *vector = computeCubinFunctionOffsets(cubin, cubin_size);
      cubin_crc_map_insert(cubin_crc, hpctoolkit_module_id, vector);
    }
    cubin_id_map_entry_t *id_entry = cubin_id_map_lookup(cubin_id);
    if (id_entry == NULL) {
      Elf_SymbolVector *vector = computeCubinFunctionOffsets(cubin, cubin_size);
      cubin_id_map_insert(cubin_id, hpctoolkit_module_id, vector);
    }
  }
}

#else

void
cupti_load_callback_cuda
(
 CUcontext context,
 uint32_t cubin_id,
 const void *cubin,
 size_t cubin_size
)
{
  // Compute hash for cubin and store it into a map
  cubin_hash_map_entry_t *entry = cubin_hash_map_lookup(cubin_id);
  unsigned char *hash;
  unsigned int hash_len;
  if (entry == NULL) {
    cubin_hash_map_insert(cubin_id, cubin, cubin_size);
    entry = cubin_hash_map_lookup(cubin_id);
  }
  hash = cubin_hash_map_entry_hash_get(entry, &hash_len);

  // Create file name
  char file_name[PATH_MAX];
  char hash_string[PATH_MAX];
  size_t used = 0;
  size_t i;
  for (i = 0; i < hash_len; ++i) {
    used += sprintf(&hash_string[used], "%02x", hash[i]);
  }

  // Create full path for the CUBIN
  gpu_binary_path_generate(hash_string, file_name);

  // Write a file if does not exist
  bool file_flag;
  spinlock_lock(&files_lock);
  file_flag = gpu_binary_store(file_name, cubin, cubin_size);
  spinlock_unlock(&files_lock);

  if (file_flag) {
    char device_file[PATH_MAX];
    sprintf(device_file, "%s", file_name);
    uint32_t hpctoolkit_module_id;
    load_module_t *module = NULL;
    hpcrun_loadmap_lock();
    if ((module = hpcrun_loadmap_findByName(device_file)) == NULL) {
      hpctoolkit_module_id = hpcrun_loadModule_add(device_file);
    } else {
      hpctoolkit_module_id = module->id;
    }
    hpcrun_loadmap_unlock();
    TMSG(CUDA_CUBIN, "cubin_id %d -> hpctoolkit_module_id %d", cubin_id, hpctoolkit_module_id);
    cubin_id_map_entry_t *entry = cubin_id_map_lookup(cubin_id);
    if (entry == NULL) {
      Elf_SymbolVector *vector = computeCubinFunctionOffsets(cubin, cubin_size);
      cubin_id_map_insert(cubin_id, hpctoolkit_module_id, vector);
    }
  }
}

#endif


void
cupti_unload_callback_cuda
(
 CUcontext context,
 uint32_t cubin_id,
 const void *cubin,
 size_t cubin_size
)
{
#ifdef NEW_CUPTI
  TMSG(CUDA_CUBIN, "Context %p cubin_id %d unload", context, cubin_id);
  if (context != NULL) {
    // Flush records but not stop context.
    // No need to lock because the current operation is not on GPU
    cupti_range_last();
  }
#endif
  //cubin_id_map_delete(cubin_id);
}


static ip_normalized_t
cupti_func_ip_resolve
(
 CUfunction function
)
{
  hpctoolkit_cufunc_st_t *cufunc = (hpctoolkit_cufunc_st_t *)(function);
  hpctoolkit_cumod_st_t *cumod = (hpctoolkit_cumod_st_t *)cufunc->cumod;
  uint32_t function_index = cufunc->function_index;
  uint32_t cubin_id = cumod->cubin_id;
  ip_normalized_t ip_norm = cubin_id_transform(cubin_id, function_index, 0);
  TMSG(CUPTI_TRACE, "Decode function_index %u cubin_id %u", function_index, cubin_id);
  return ip_norm;
}


static void
ensure_kernel_ip_present
(
 cct_node_t *kernel_ph, 
 ip_normalized_t kernel_ip
)
{
  // if the phaceholder was previously inserted, it will have a child
  // we only want to insert a child if there isn't one already. if the
  // node contains a child already, then the gpu monitoring thread 
  // may be adding children to the splay tree of children. in that case 
  // trying to add a child here (which will turn into a lookup of the
  // previously added child, would race with any insertions by the 
  // GPU monitoring thread.
  //
  // INVARIANT: avoid a race modifying the splay tree of children by 
  // not attempting to insert a child in a worker thread when a child 
  // is already present
  if (hpcrun_cct_children(kernel_ph) == NULL) {
    cct_node_t *kernel = 
      hpcrun_cct_insert_ip_norm(kernel_ph, kernel_ip, true);
    hpcrun_cct_retain(kernel);
  }
}

#ifdef NEW_CUPTI

static void
cupti_resource_subscriber_callback
(
 void *userdata,
 CUpti_CallbackDomain domain,
 CUpti_CallbackId cb_id,
 const void *cb_info
)
{
  const CUpti_ResourceData *rd = (const CUpti_ResourceData *)cb_info;
  CUpti_ModuleResourceData *mrd = (CUpti_ModuleResourceData *)rd->resourceDescriptor;
  int pc_sampling_frequency = cupti_pc_sampling_frequency_get();
  if (cb_id == CUPTI_CBID_RESOURCE_MODULE_LOADED) {
    TMSG(CUPTI, "Context %p loaded module id %d, cubin size %" PRIu64 ", cubin %p",
      rd->context, mrd->moduleId, mrd->cubinSize, mrd->pCubin);
    DISPATCH_CALLBACK(cupti_load_callback, (rd->context, mrd->moduleId, mrd->pCubin, mrd->cubinSize));
  } else if (cb_id == CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING) {
    TMSG(CUPTI, "Context %p unloaded module id %d, cubin size %" PRIu64 ", cubin %p",
      rd->context, mrd->moduleId, mrd->cubinSize, mrd->pCubin);
    DISPATCH_CALLBACK(cupti_unload_callback, (rd->context, mrd->moduleId, mrd->pCubin, mrd->cubinSize));
  } else if (cb_id == CUPTI_CBID_RESOURCE_CONTEXT_CREATED) {
    TMSG(CUPTI, "Context %p created", rd->context);
    if (pc_sampling_frequency != CUPTI_PC_SAMPLING_PERIOD_NULL) {
      cupti_pc_sampling_enable2(rd->context);
      cupti_pc_sampling_config(rd->context, pc_sampling_frequency);
    }
  } else if (cb_id == CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING) {
    TMSG(CUPTI, "Context %lu destroyed", rd->context);
    if (pc_sampling_frequency != CUPTI_PC_SAMPLING_PERIOD_NULL) {
      cupti_range_last();
      cupti_pc_sampling_disable2(rd->context);
      cupti_pc_sampling_free(rd->context);
    }
  }
}


static void
cupti_callback_init
(
)
{
  // stop flag is only set if a driver or a runtime api has been called
  cupti_thread_activity_flag_set();

  if (cupti_pc_sampling_frequency_get() != CUPTI_PC_SAMPLING_PERIOD_NULL) {
    // channel is only initialized if a driver or a runtime api has been called
    gpu_operation_multiplexer_my_channel_init();
  }
}

//******************************************************************************
// Runtime and driver API callbacks
//******************************************************************************

static uint64_t get_timestamp() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  uint64_t nanotime = ((uint64_t)tv.tv_usec + (((uint64_t)tv.tv_sec) * 1000000)) * 1000;
  return nanotime;
}

static __thread uint64_t unwind_time = 0;

static cct_node_t *
cupti_unwind
(
 gpu_op_placeholder_flags_t flags,
 long rsp, 
 void *args
)
{
  uint64_t start_time = 0;
  if (debug_flag_get(DBG_PREFIX(CUPTI_CCT))) {
    start_time = get_timestamp();
  }

  if (cupti_fast_unwind_get()) {
    // Slow path to generate a cct
    cct_node_t *node = cupti_correlation_callback();

    if (debug_flag_get(DBG_PREFIX(CUPTI_CCT))) {
      slow_unwinds += 1;
      total_unwinds += 1;
      unwind_time += get_timestamp() - start_time;
    }
    return node;
  }

  // Fast path to generate a cct
  unwind_key_t unwind_key;
  unwind_key.stack_length = rsp;
  unwind_key.prev_kernel = cupti_prev_kernel_node;
  unwind_key.prev_prev_kernel = cupti_prev_prev_kernel_node;
  unwind_key.prev_api = cupti_prev_api_node;

  if (gpu_op_placeholder_flags_is_set(flags, gpu_placeholder_type_kernel)) {
    CUfunction function_ptr = (CUfunction)args;
    unwind_key.function_id = cupti_func_ip_resolve(function_ptr);
  } else {
    unwind_key.function_id.lm_id = 0;
    unwind_key.function_id.lm_ip = flags;
  }

  cupti_unwind_map_entry_t *entry = cupti_unwind_map_lookup(unwind_key);

  // If not matched, unwind and memoize.
  // If memoized, generated a random number and see if it falls into the 
  // backoff range. If yes, unwind the cct and check if the two api nodes
  // match, if not, backoff is descreased. Otherwise, backoff is increased. 
  cct_node_t *api_node = NULL;
  if (entry == NULL) {
    api_node = cupti_correlation_callback();
    cupti_unwind_map_insert(unwind_key, api_node);
    if (debug_flag_get(DBG_PREFIX(CUPTI_CCT))) {
      fast_unwinds += 1;
    }
  } else {
    api_node = cupti_unwind_map_entry_cct_node_get(entry);
    int backoff = cupti_unwind_map_entry_backoff_get(entry);
    if (backoff < cupti_correlation_threshold_get()) {
      int threshold = pow(cupti_backoff_base_get(), backoff);
      int left = rand() % threshold;
      if (left == 0) {
        if (debug_flag_get(DBG_PREFIX(CUPTI_CCT))) {
          slow_unwinds += 1;
        }
        cct_node_t *actual_node = cupti_correlation_callback();
        if (actual_node != api_node) {
          api_node = actual_node;
          cupti_unwind_map_entry_cct_node_update(entry, actual_node);
          cupti_unwind_map_entry_backoff_update(entry, 0);
        } else {
          cupti_unwind_map_entry_backoff_update(entry, backoff + 1);
        }
      } else if (debug_flag_get(DBG_PREFIX(CUPTI_CCT))) {
        fast_unwinds += 1;
      }
    } else if (debug_flag_get(DBG_PREFIX(CUPTI_CCT))) {
      fast_unwinds += 1;
    }
  }

  if (debug_flag_get(DBG_PREFIX(CUPTI_CCT))) {
    unwind_time += get_timestamp() - start_time;
    cct_node_t *actual_node = cupti_correlation_callback();
    total_unwinds++;
    if (actual_node == api_node) {
      correct_unwinds++; 
    }
  }
  return api_node;
}


static cct_node_t *
cupti_api_node_get
(
 gpu_op_placeholder_flags_t flags,
 uint64_t correlation_id,
 const void *cb_info
)
{
  // Query key for the unwind map
  const CUpti_CallbackData *cd = (const CUpti_CallbackData *) cb_info;
  // TODO(Keren): Add stack length fetch for powerpc and arm
  register long rsp asm("rsp");
  cct_node_t *api_node = cupti_unwind(flags, rsp, *(CUfunction *)(cd->functionParams));
  
  // Update prev indicators
  if (gpu_op_placeholder_flags_is_set(flags, gpu_placeholder_type_kernel)) {
    cupti_prev_prev_kernel_node = cupti_prev_kernel_node;
    cupti_prev_kernel_node = api_node;
  } else {
    cupti_prev_api_node = api_node;
  }

  return api_node;
}


static gpu_op_ccts_t
cupti_api_enter_callback_cuda
(
 gpu_op_placeholder_flags_t flags,
 CUpti_CallbackId cb_id,
 const void *cb_info
)
{
  cupti_callback_init();

  // In the serialized mode or range profilng is not enabled, range_id is always zero
  uint32_t range_id = gpu_range_id_get();

  // A driver API cannot be implemented by other driver APIs, so we get an id
  // and unwind when the API is entered
  uint64_t correlation_id = CUPTI_CORRELATION_ID_NULL;
  if (cupti_runtime_api_flag_get()) {
    // runtime API RA
    // driver API dA dB
    /*  ------[   RA   ]-----
     *    --------dA-------
     *           /|\
     *            |
     */
    correlation_id = cupti_runtime_correlation_id_get();
    if (correlation_id == CUPTI_CORRELATION_ID_NULL) {
      correlation_id = gpu_correlation_id();
      cupti_runtime_correlation_id_set(correlation_id);
      cupti_correlation_id_push(correlation_id);
      TMSG(CUPTI_TRACE, "Runtime push externalId %lu (cb_id = %u, range_id = %u)", correlation_id, cb_id, range_id);
    }
  } else {
    /* Without a runtime API
     *    -dA-      -dB-
     *    /|\
     *     |
     */
    correlation_id = gpu_correlation_id();
    cupti_correlation_id_push(correlation_id);
    TMSG(CUPTI_TRACE, "Driver push externalId %lu (cb_id = %u, range_id = %u)", correlation_id, cb_id, range_id);
  }

  cupti_driver_correlation_id_set(correlation_id);
  cct_node_t *api_node = cupti_api_node_get(flags, correlation_id, cb_info);

  if (debug_flag_get(DBG_PREFIX(CUPTI_CCT)) && 
    !gpu_op_placeholder_flags_is_set(flags, gpu_placeholder_type_kernel)) {
    cupti_cct_map_insert(api_node, range_id);
  }

  gpu_op_ccts_t gpu_op_ccts;

  hpcrun_safe_enter();

  gpu_op_ccts_insert(api_node, &gpu_op_ccts, flags);

  cupti_gpu_monitors_apply_enter(api_node);

  hpcrun_safe_exit();

  // Generate a notification entry
  uint64_t cpu_submit_time = hpcrun_nanotime();
  gpu_correlation_channel_produce(correlation_id, &gpu_op_ccts, cpu_submit_time);

  return gpu_op_ccts;
}


static void
cupti_api_exit_callback_cuda
(
 CUpti_CallbackId cb_id
)
{
  uint64_t correlation_id = cupti_runtime_correlation_id_get();
  uint32_t range_id = gpu_range_id_get();

  if (correlation_id == CUPTI_CORRELATION_ID_NULL) {
    correlation_id = cupti_correlation_id_pop();
    // Runtime API has not been set before, must be the exit of a driver API
    TMSG(CUPTI_TRACE, "Driver pop externalId %lu (cb_id = %u, range_id = %u)", correlation_id, cb_id, range_id);
  } else if (!cupti_runtime_api_flag_get()) {
    /* cupti_runtime_api_flag_get() == false
     * ---[    RA   ]-------
     *        -dA- /|\
     *              |
     */
    correlation_id = cupti_correlation_id_pop();
    TMSG(CUPTI_TRACE, "Runtime pop externalId %lu (cb_id = %u, range_id = %u)", correlation_id, cb_id, range_id);
  }
  /* else
   * ---[       RA      ]-------
   *        -dA1--dA2-
   *              /|\
   *               |
   *                     
   * Still in a runtime API, don't pop this id
   */

  cupti_gpu_monitors_apply_exit();
}


// General driver callback
static void
cupti_driver_api_subscriber_callback_cuda
(
 gpu_op_placeholder_flags_t flags,
 CUpti_CallbackDomain domain,
 CUpti_CallbackId cb_id,
 const void *cb_info
)
{
  const CUpti_CallbackData *cd = (const CUpti_CallbackData *)cb_info;
  if (!cupti_runtime_api_flag_get() && !ompt_runtime_status_get()) {
    // Directly calls driver APIs
    if (cd->callbackSite == CUPTI_API_ENTER) {
      if (cupti_pc_sampling_frequency_get() == CUPTI_PC_SAMPLING_PERIOD_NULL) {
        // In the pc sampling mode, we don't capture other GPU activities
        cupti_api_enter_callback_cuda(flags, cb_id, cb_info);
      }
    } else {
      if (cupti_pc_sampling_frequency_get() == CUPTI_PC_SAMPLING_PERIOD_NULL) {
        // In the pc sampling mode, we don't capture other GPU activities
        cupti_api_exit_callback_cuda(cb_id);
      }
    }
  } else if (cupti_runtime_api_flag_get()) {
    // Runtime API calls driver APIs
    uint32_t range_id = gpu_range_id_get();
    if (cd->callbackSite == CUPTI_API_ENTER) {
      TMSG(CUPTI_TRACE, "Driver enter (cb_id = %u, range_id = %u)", cb_id, range_id);
    } else {
      TMSG(CUPTI_TRACE, "Driver exit (cb_id = %u, range_id = %u)", cb_id, range_id);
    }
  }
}


// Driver kernel callback
static void
cupti_driver_api_subscriber_callback_cuda_kernel
(
 gpu_op_placeholder_flags_t flags,
 CUpti_CallbackDomain domain,
 CUpti_CallbackId cb_id,
 const void *cb_info
)
{
  const CUpti_CallbackData *cd = (const CUpti_CallbackData *) cb_info;
  if (cd->callbackSite == CUPTI_API_ENTER) {
    gpu_application_thread_process_activities();

    // CUfunction is the first param
    // XXX(Keren): cannot parse this kind of kernel launch
    // cb_id = CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernelMultiDevice)
    CUfunction function_ptr = *(CUfunction *)(cd->functionParams);
    ip_normalized_t kernel_ip = cupti_func_ip_resolve(function_ptr);

    if (ompt_runtime_status_get()) {
      // Update kernel_ip for the ompt API
      // XXX(Keren): range profiling is not applicable for ompt
      cct_node_t *ompt_trace_node = ompt_trace_node_get();
      if (ompt_trace_node != NULL) {
        ensure_kernel_ip_present(ompt_trace_node, kernel_ip);
      }
    } else {
      gpu_op_ccts_t gpu_op_ccts = cupti_api_enter_callback_cuda(flags, cb_id, cb_info);

      cupti_kernel_ph_set(gpu_op_ccts_get(&gpu_op_ccts, gpu_placeholder_type_kernel));
      ensure_kernel_ip_present(cupti_kernel_ph_get(), kernel_ip);

      cupti_trace_ph_set(gpu_op_ccts_get(&gpu_op_ccts, gpu_placeholder_type_trace));
      ensure_kernel_ip_present(cupti_trace_ph_get(), kernel_ip);

      // Ranges are only divided by kernels but not other GPU APIs
      gpu_range_enter(cupti_kernel_ph_get(), cupti_driver_correlation_id_get());
    }
  } else if (!ompt_runtime_status_get()) {
    cupti_api_exit_callback_cuda(cb_id);

    gpu_range_exit();
  }
}


// General runtime callback
static void
cupti_runtime_api_subscriber_callback_cuda
(
 gpu_op_placeholder_flags_t flags,
 CUpti_CallbackDomain domain,
 CUpti_CallbackId cb_id,
 const void *cb_info
)
{
  const CUpti_CallbackData *cd = (const CUpti_CallbackData *)cb_info;
  if (cd->callbackSite == CUPTI_API_ENTER) {
    // Enter a CUDA runtime api
    cupti_runtime_api_flag_set();
    if (cupti_pc_sampling_frequency_get() == CUPTI_PC_SAMPLING_PERIOD_NULL) {
      // In the pc sampling mode, we don't capture other GPU activities
      cupti_api_enter_callback_cuda(flags, cb_id, cb_info);
    }
  } else {
    // Exit an CUDA runtime api
    cupti_runtime_api_flag_unset();
    if (cupti_pc_sampling_frequency_get() == CUPTI_PC_SAMPLING_PERIOD_NULL) {
      //   In the pc sampling mode, we don't capture other GPU activities
      cupti_api_exit_callback_cuda(cb_id);
    }
    cupti_runtime_correlation_id_set(CUPTI_CORRELATION_ID_NULL);
  }
}


static void
cupti_runtime_api_subscriber_callback_cuda_kernel
(
 gpu_op_placeholder_flags_t flags,
 CUpti_CallbackDomain domain,
 CUpti_CallbackId cb_id,
 const void *cb_info
)
{
  const CUpti_CallbackData *cd = (const CUpti_CallbackData *)cb_info;
  if (cd->callbackSite == CUPTI_API_ENTER) {
    // Enter a CUDA runtime api
    // For GPU kernels, we memoize a runtime API's correlation id and use it for its driver APIs
    uint64_t correlation_id = gpu_correlation_id();
    cupti_correlation_id_push(correlation_id);
    cupti_runtime_correlation_id_set(correlation_id);
    cupti_runtime_api_flag_set();
  } else {
    // Exit an CUDA runtime api
    cupti_correlation_id_pop();
    cupti_runtime_correlation_id_set(CUPTI_CORRELATION_ID_NULL);
    cupti_runtime_api_flag_unset();
    cupti_kernel_ph_set(NULL);
    cupti_trace_ph_set(NULL);
  }
}


static void
cupti_subscriber_callback_cuda
(
 void *userdata,
 CUpti_CallbackDomain domain,
 CUpti_CallbackId cb_id,
 const void *cb_info
)
{
  if (is_tool_active()) {
    return;
  }

  if (domain == CUPTI_CB_DOMAIN_RESOURCE) {
    cupti_resource_subscriber_callback(userdata, domain, cb_id, cb_info);
  } else if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
    gpu_op_placeholder_flags_t flags = cupti_driver_flags_get(cb_id);

    if (gpu_op_placeholder_flags_is_set(flags, gpu_placeholder_type_kernel)) {
      cupti_driver_api_subscriber_callback_cuda_kernel(flags, domain, cb_id, cb_info);
    } else if (flags) {
      cupti_driver_api_subscriber_callback_cuda(flags, domain, cb_id, cb_info);
    }
  } else if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
    gpu_op_placeholder_flags_t flags = cupti_runtime_flags_get(cb_id);

    if (gpu_op_placeholder_flags_is_set(flags, gpu_placeholder_type_kernel)) {
      cupti_runtime_api_subscriber_callback_cuda_kernel(flags, domain, cb_id, cb_info);
    } else if (flags) {
      cupti_runtime_api_subscriber_callback_cuda(flags, domain, cb_id, cb_info);
    }
  }
}

#else

static void
cupti_subscriber_callback_cuda
(
 void *userdata,
 CUpti_CallbackDomain domain,
 CUpti_CallbackId cb_id,
 const void *cb_info
)
{
  if (is_tool_active()) {
    return;
  }

  if (domain == CUPTI_CB_DOMAIN_RESOURCE) {
    const CUpti_ResourceData *rd = (const CUpti_ResourceData *) cb_info;
    if (cb_id == CUPTI_CBID_RESOURCE_MODULE_LOADED) {
      CUpti_ModuleResourceData *mrd = (CUpti_ModuleResourceData *)
        rd->resourceDescriptor;

      TMSG(CUPTI, "Context %p loaded module id %d, cubin size %" PRIu64 ", cubin %p",
        rd->context, mrd->moduleId, mrd->cubinSize, mrd->pCubin);
      DISPATCH_CALLBACK(cupti_load_callback, (rd->context, mrd->moduleId, mrd->pCubin, mrd->cubinSize));
    } else if (cb_id == CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING) {
      CUpti_ModuleResourceData *mrd = (CUpti_ModuleResourceData *)
        rd->resourceDescriptor;

      TMSG(CUPTI, "Context %p unloaded module id %d, cubin size %" PRIu64 ", cubin %p",
        rd->context, mrd->moduleId, mrd->cubinSize, mrd->pCubin);
      DISPATCH_CALLBACK(cupti_unload_callback, (rd->context, mrd->moduleId, mrd->pCubin, mrd->cubinSize));
    } else if (cb_id == CUPTI_CBID_RESOURCE_CONTEXT_CREATED) {
      TMSG(CUPTI, "Context %p created", rd->context);
      int pc_sampling_frequency = cupti_pc_sampling_frequency_get();
      if (pc_sampling_frequency != CUPTI_PC_SAMPLING_PERIOD_NULL) {
        cupti_pc_sampling_enable(rd->context, pc_sampling_frequency);
      }
      if (cupti_sync_yield_get()) {
        cuda_sync_yield();
      }
    } else if (cb_id == CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING) {
      TMSG(CUPTI, "Context %lu destroyed", rd->context);
      int pc_sampling_frequency = cupti_pc_sampling_frequency_get();
      if (pc_sampling_frequency != CUPTI_PC_SAMPLING_PERIOD_NULL) {
        cupti_pc_sampling_disable(rd->context);
      }
    }
  } else if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
    // stop flag is only set if a driver or runtime api called
    cupti_thread_activity_flag_set();

    //gpu_operation_multiplexer_my_channel_init();

    const CUpti_CallbackData *cd = (const CUpti_CallbackData *) cb_info;
		PRINT("\nDriver API:  -----------------%s\n", cd->functionName );

    bool ompt_runtime_api_flag = ompt_runtime_status_get();

    bool is_valid_op = false;
    gpu_op_placeholder_flags_t gpu_op_placeholder_flags = 0;
    ip_normalized_t kernel_ip;

    switch (cb_id) {
      // synchronize apis
      case CUPTI_DRIVER_TRACE_CBID_cuCtxSynchronize:
      case CUPTI_DRIVER_TRACE_CBID_cuEventSynchronize:
      case CUPTI_DRIVER_TRACE_CBID_cuStreamSynchronize:
      case CUPTI_DRIVER_TRACE_CBID_cuStreamSynchronize_ptsz:
      case CUPTI_DRIVER_TRACE_CBID_cuStreamWaitEvent:
      case CUPTI_DRIVER_TRACE_CBID_cuStreamWaitEvent_ptsz:
        {
          gpu_op_placeholder_flags_set(&gpu_op_placeholder_flags,
            gpu_placeholder_type_sync);
          is_valid_op = true;
          break;
        }
      // copyin apis
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2_ptds:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2_ptsz:
        {
          gpu_op_placeholder_flags_set(&gpu_op_placeholder_flags,
            gpu_placeholder_type_copyin);
          is_valid_op = true;
          break;
        }
      // copyout apis
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2_ptds:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2_ptsz:
        {
          gpu_op_placeholder_flags_set(&gpu_op_placeholder_flags,
            gpu_placeholder_type_copyout);
          is_valid_op = true;
          break;
        }
      // copy apis
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoA:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoD:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoA:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoH:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoA:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpy2D:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DUnaligned:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpy3D:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoAAsync:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoHAsync:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DAsync:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DAsync:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpy_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoH_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoHAsync_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoD_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoA_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoA_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpy2D_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DUnaligned_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DAsync_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpy3D_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DAsync_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoA_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoAAsync_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpy:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyPeer:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyPeerAsync:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DPeer:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DPeerAsync:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2_ptds:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoA_v2_ptds:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoD_v2_ptds:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoA_v2_ptds:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoH_v2_ptds:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoA_v2_ptds:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpy2D_v2_ptds:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DUnaligned_v2_ptds:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpy3D_v2_ptds:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpy_ptds:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyPeer_ptds:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DPeer_ptds:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync_ptsz:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoAAsync_v2_ptsz:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoHAsync_v2_ptsz:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2_ptsz:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DAsync_v2_ptsz:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DAsync_v2_ptsz:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyPeerAsync_ptsz:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DPeerAsync_ptsz:
        {
          gpu_op_placeholder_flags_set(&gpu_op_placeholder_flags,
            gpu_placeholder_type_copy);
          is_valid_op = true;
          break;
        }
        // kernel apis
      case CUPTI_DRIVER_TRACE_CBID_cuLaunch:
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchGrid:
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchGridAsync:
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz:
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel:
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz:
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernelMultiDevice:
        {
          gpu_op_placeholder_flags_set(&gpu_op_placeholder_flags,
            gpu_placeholder_type_kernel);

          gpu_op_placeholder_flags_set(&gpu_op_placeholder_flags,
            gpu_placeholder_type_trace);

          is_valid_op = true;

          if (cd->callbackSite == CUPTI_API_ENTER) {
            gpu_application_thread_process_activities();
            // XXX(Keren): cannot parse this kind of kernel launch
           //if (cb_id != CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernelMultiDevice)
            // CUfunction is the first param
            CUfunction function_ptr = *(CUfunction *)(cd->functionParams);
            kernel_ip = cupti_func_ip_resolve(function_ptr);
          }
          break;
        }
      default:
        break;
    }

    bool is_kernel_op = gpu_op_placeholder_flags_is_set(gpu_op_placeholder_flags,gpu_placeholder_type_kernel);

//		PRINT("DRIVER: is_valid_op = %d \t is_kernel = %d \t cupti_runtime_api_flag = %d \t ompt_runtime_api_flag = %d | callback_site = %d\n",
//					 is_valid_op, is_kernel_op, cupti_runtime_api_flag, ompt_runtime_api_flag, cd->callbackSite);

    // If we have a valid operation and is not in the interval of a cuda/ompt runtime api
    if (is_valid_op && !cupti_runtime_api_flag && !ompt_runtime_api_flag) {
      if (cd->callbackSite == CUPTI_API_ENTER) {
        uint64_t correlation_id = gpu_correlation_id();
        // A driver API cannot be implemented by other driver APIs, so we get an id
        // and unwind when the API is entered
        cupti_correlation_id_push(correlation_id);

        cct_node_t *api_node = cupti_correlation_callback();

        gpu_op_ccts_t gpu_op_ccts;

        hpcrun_safe_enter();

        gpu_op_ccts_insert(api_node, &gpu_op_ccts, gpu_op_placeholder_flags);

        if (is_kernel_op) {
          cct_node_t *kernel_ph = gpu_op_ccts_get(&gpu_op_ccts, gpu_placeholder_type_kernel);

          gpu_cct_insert(kernel_ph, kernel_ip);

          cct_node_t *trace_ph = gpu_op_ccts_get(&gpu_op_ccts, gpu_placeholder_type_trace);

          gpu_cct_insert(trace_ph, kernel_ip);
        }


        hpcrun_safe_exit();

        // Generate notification entry
        uint64_t cpu_submit_time = hpcrun_nanotime();

        gpu_correlation_channel_produce(correlation_id, &gpu_op_ccts, cpu_submit_time);

        TMSG(CUPTI_TRACE, "Driver push externalId %lu (cb_id = %u)", correlation_id, cb_id);
      } else if (cd->callbackSite == CUPTI_API_EXIT) {
        uint64_t correlation_id __attribute__((unused)); // not used if PRINT omitted
        correlation_id = cupti_correlation_id_pop();
        TMSG(CUPTI_TRACE, "Driver pop externalId %lu (cb_id = %u)", correlation_id, cb_id);
      }
    } else if (is_kernel_op && cupti_runtime_api_flag && cd->callbackSite ==
      CUPTI_API_ENTER) {
      if (cupti_kernel_ph != NULL) {
        gpu_cct_insert(cupti_kernel_ph, kernel_ip);
      }
      if (cupti_trace_ph != NULL) {
        gpu_cct_insert(cupti_trace_ph, kernel_ip);
      }
    } else if (is_kernel_op && ompt_runtime_api_flag && cd->callbackSite ==
      CUPTI_API_ENTER) {
      cct_node_t *ompt_trace_node = ompt_trace_node_get();
      if (ompt_trace_node != NULL) {
        gpu_cct_insert(ompt_trace_node, kernel_ip);
      }
    }
  } else if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
    // stop flag is only set if a driver or runtime api called
    cupti_thread_activity_flag_set();

    //gpu_operation_multiplexer_my_channel_init();

    const CUpti_CallbackData *cd = (const CUpti_CallbackData *)cb_info;
		PRINT("\nRuntime API:  -----------------%s\n", cd->functionName );

    bool is_valid_op = false;
    bool is_kernel_op = false;
    switch (cb_id) {
      // FIXME(Keren): do not support memory allocate and free for
      // current CUPTI version
      //case CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020:
      //case CUPTI_RUNTIME_TRACE_CBID_cudaMallocPitch_v3020:
      //case CUPTI_RUNTIME_TRACE_CBID_cudaMallocArray_v3020:
      //case CUPTI_RUNTIME_TRACE_CBID_cudaMalloc3D_v3020:
      //case CUPTI_RUNTIME_TRACE_CBID_cudaMalloc3DArray_v3020:
      //  {
      //    cuda_state = cuda_placeholders.cuda_memalloc_state;
      //    is_valid_op = true;
      //    break;
      //  }
      // cuda synchronize apis
      case CUPTI_RUNTIME_TRACE_CBID_cudaEventSynchronize_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_ptsz_v7000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaStreamWaitEvent_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020:
      // cuda copy apis
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyPeer_v4000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyPeerAsync_v4000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3DPeer_v4000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3DPeerAsync_v4000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3D_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3DAsync_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3D_ptds_v7000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3DAsync_ptsz_v7000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3DPeer_ptds_v7000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3DPeerAsync_ptsz_v7000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2D_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToArray_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DToArray_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromArray_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DFromArray_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyArrayToArray_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DArrayToArray_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToSymbol_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromSymbol_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToArrayAsync_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromArrayAsync_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DAsync_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DToArrayAsync_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DFromArrayAsync_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToSymbolAsync_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromSymbolAsync_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_ptds_v7000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2D_ptds_v7000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToArray_ptds_v7000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DToArray_ptds_v7000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromArray_ptds_v7000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DFromArray_ptds_v7000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyArrayToArray_ptds_v7000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DArrayToArray_ptds_v7000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToSymbol_ptds_v7000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromSymbol_ptds_v7000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_ptsz_v7000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToArrayAsync_ptsz_v7000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromArrayAsync_ptsz_v7000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DAsync_ptsz_v7000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DToArrayAsync_ptsz_v7000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DFromArrayAsync_ptsz_v7000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToSymbolAsync_ptsz_v7000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromSymbolAsync_ptsz_v7000:
        {
          is_valid_op = true;
          break;
        }
      // cuda kernel apis
      case CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_ptsz_v7000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000:
      #if CUPTI_API_VERSION >= 10
      case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_v9000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_ptsz_v9000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernelMultiDevice_v9000:
      #endif
        {
          is_valid_op = true;
          is_kernel_op = true;
          if (cd->callbackSite == CUPTI_API_ENTER) {
            gpu_application_thread_process_activities();
          }
          break;
        }
      default:
        break;
    }

//		PRINT("RUNTIME: is_valid_op = %d \t is_kernel = %d \t cupti_runtime_api_flag = %d \t ompt_runtime_api_flag = %d | callback_site = %d\n",
//					 is_valid_op, is_kernel_op, cupti_runtime_api_flag, ompt_runtime_status_get(), cd->callbackSite);

    if (is_valid_op) {
      if (cd->callbackSite == CUPTI_API_ENTER) {
        // Enter a CUDA runtime api
        cupti_runtime_api_flag_set();

        uint64_t correlation_id = gpu_correlation_id();
        cupti_correlation_id_push(correlation_id);

        // We should make notification records in the api enter callback.
        // A runtime API must be implemented by driver APIs.
        // Though unlikely in most cases,
        // it is still possible that a cupti buffer is full and returned to the host
        // in the interval of a runtime api.
        cct_node_t *api_node = cupti_correlation_callback();

        gpu_op_ccts_t gpu_op_ccts;

        hpcrun_safe_enter();

        gpu_op_ccts_insert(api_node, &gpu_op_ccts, gpu_op_placeholder_flags_all);

        hpcrun_safe_exit();

        cupti_kernel_ph_set(gpu_op_ccts_get(&gpu_op_ccts, gpu_placeholder_type_kernel));
        cupti_trace_ph_set(gpu_op_ccts_get(&gpu_op_ccts, gpu_placeholder_type_trace));

        // Generate notification entry
        uint64_t cpu_submit_time = hpcrun_nanotime();

	gpu_correlation_channel_produce(correlation_id, &gpu_op_ccts, cpu_submit_time);

        TMSG(CUPTI_TRACE, "Runtime push externalId %lu (cb_id = %u)", correlation_id, cb_id);
      } else if (cd->callbackSite == CUPTI_API_EXIT) {
        // Exit an CUDA runtime api
        cupti_runtime_api_flag_unset();

        uint64_t correlation_id;
        correlation_id = cupti_correlation_id_pop();

        TMSG(CUPTI_TRACE, "Runtime pop externalId %lu (cb_id = %u)", correlation_id, cb_id);

        cupti_kernel_ph_set(NULL);
        cupti_trace_ph_set(NULL);
      }
    } else {
      TMSG(CUPTI_TRACE, "Go through runtime with kernel_op %d, valid_op %d, "
        "cuda_runtime %d", is_kernel_op, is_valid_op, cupti_runtime_api_flag);
    }
  }
}

#endif

//******************************************************************************
// interface  operations
//******************************************************************************

void
cupti_device_timestamp_get
(
 CUcontext context,
 uint64_t *time
)
{
  HPCRUN_CUPTI_CALL(cuptiGetTimestamp, (time));
}


void
cupti_activity_timestamp_get
(
 uint64_t *time
)
{
  HPCRUN_CUPTI_CALL(cuptiGetTimestamp, (time));
}


void
cupti_device_buffer_config
(
 size_t buf_size,
 size_t sem_size
)
{
  size_t value_size = sizeof(size_t);
  HPCRUN_CUPTI_CALL(cuptiActivitySetAttribute,
                   (CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &value_size, &buf_size));
  HPCRUN_CUPTI_CALL(cuptiActivitySetAttribute,
                   (CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_POOL_SIZE, &value_size, &sem_size));
}


void
cupti_buffer_alloc
(
 uint8_t **buffer,
 size_t *buffer_size,
 size_t *maxNumRecords
)
{
  // cupti client call this function
  int retval = posix_memalign((void **) buffer,
    (size_t) HPCRUN_CUPTI_ACTIVITY_BUFFER_ALIGNMENT,
    (size_t) HPCRUN_CUPTI_ACTIVITY_BUFFER_SIZE);

  if (retval != 0) {
    cupti_error_callback("CUPTI", "cupti_buffer_alloc", "out of memory");
  }

  *buffer_size = HPCRUN_CUPTI_ACTIVITY_BUFFER_SIZE;

  *maxNumRecords = 0;
}


bool
cupti_buffer_cursor_advance
(
  uint8_t *buffer,
  size_t size,
  CUpti_Activity **current
)
{
  return (CUPTI_FN_NAME(cuptiActivityGetNextRecord)(buffer, size, current) == CUPTI_SUCCESS);
}


void
cupti_buffer_completion_callback
(
 CUcontext ctx,
 uint32_t streamId,
 uint8_t *buffer,
 size_t size,
 size_t validSize
)
{
  TMSG(CUPTI, "Enter CUPTI_buffer_completion");

  hpcrun_thread_init_mem_pool_once(0, NULL, false, true);

  // handle notifications
  cupti_buffer_completion_notify();

  if (validSize > 0) {
    // Signal advance to return pointer to first record
    CUpti_Activity *cupti_activity = NULL;
    bool status = false;
    size_t processed = 0;
    do {
      status = cupti_buffer_cursor_advance(buffer, validSize, &cupti_activity);
      if (status) {
				cupti_activity_process(cupti_activity);
        ++processed;
      }
    } while (status);
    hpcrun_stats_acc_trace_records_add(processed);

    size_t dropped;
    cupti_num_dropped_records_get(ctx, streamId, &dropped);
    if (dropped != 0) {
      hpcrun_stats_acc_trace_records_dropped_add(dropped);
    }
  }

  free(buffer);

  TMSG(CUPTI, "Exit cupti_buffer_completion");
}


//-------------------------------------------------------------
// event specification
//-------------------------------------------------------------

cupti_set_status_t
cupti_monitoring_set
(
 const CUpti_ActivityKind activity_kinds[],
 bool enable
)
{
  TMSG(CUPTI, "Enter cupti_set_monitoring");
  int failed = 0;
  int succeeded = 0;

  cupti_activity_enable_t action =
    (enable ?
     CUPTI_FN_NAME(cuptiActivityEnable):
     CUPTI_FN_NAME(cuptiActivityDisable));

  int i = 0;
  for (;;) {
    CUpti_ActivityKind activity_kind = activity_kinds[i++];
    if (activity_kind == CUPTI_ACTIVITY_KIND_INVALID) break;
    bool succ = action(activity_kind) == CUPTI_SUCCESS;
    if (succ) {
      if (enable) {
        TMSG(CUPTI, "activity %d enable succeeded", activity_kind);
      } else {
        TMSG(CUPTI, "activity %d disable succeeded", activity_kind);
      }
      succeeded++;
    } else {
      if (enable) {
        TMSG(CUPTI, "activity %d enable failed", activity_kind);
      } else {
        TMSG(CUPTI, "activity %d disable failed", activity_kind);
      }
      failed++;
    }
  }
  if (succeeded > 0) {
    if (failed == 0) return cupti_set_all;
    else return cupti_set_some;
  }
  TMSG(CUPTI, "Exit cupti_set_monitoring");
  return cupti_set_none;
}


//-------------------------------------------------------------
// control apis
//-------------------------------------------------------------

void
cupti_init
(
 void
)
{
  FLUSH_ALARM_SIGALLOC();
  
  cupti_activity_enabled.buffer_request = cupti_buffer_alloc;
  cupti_activity_enabled.buffer_complete = cupti_buffer_completion_callback;
}


void
cupti_start
(
 void
)
{
  HPCRUN_CUPTI_CALL(cuptiActivityRegisterCallbacks,
                   (cupti_activity_enabled.buffer_request,
                    cupti_activity_enabled.buffer_complete));
}


void
cupti_finalize
(
 void
)
{
  HPCRUN_CUPTI_CALL(cuptiFinalize, ());
}


void
cupti_num_dropped_records_get
(
 CUcontext context,
 uint32_t streamId,
 size_t* dropped
)
{
  HPCRUN_CUPTI_CALL(cuptiActivityGetNumDroppedRecords,
                   (context, streamId, dropped));
}


//-------------------------------------------------------------
// correlation callback control
//-------------------------------------------------------------


void
cupti_callback_enable
(
 CUpti_SubscriberHandle subscriber,
 CUpti_CallbackId cbid,
 CUpti_CallbackDomain domain
)
{
  HPCRUN_CUPTI_CALL(cuptiEnableCallback, (1, subscriber, domain, cbid));
}


void
cupti_callback_disable
(
 CUpti_SubscriberHandle subscriber,
 CUpti_CallbackId cbid,
 CUpti_CallbackDomain domain
)
{
  HPCRUN_CUPTI_CALL(cuptiEnableCallback, (0, subscriber, domain, cbid));
}

#ifdef NEW_CUPTI

void
cupti_callbacks_subscribe
(
 void
)
{
  cupti_load_callback = cupti_load_callback_cuda;
  cupti_unload_callback = cupti_unload_callback_cuda;
  cupti_correlation_callback = gpu_application_thread_correlation_callback;

  HPCRUN_CUPTI_CALL(cuptiSubscribe, (&cupti_subscriber,
                   (CUpti_CallbackFunc) cupti_subscriber_callback_cuda,
                   (void *) NULL));

  cupti_subscribers_driver_kernel_callbacks_subscribe(1, cupti_subscriber);
  cupti_subscribers_driver_memcpy_htod_callbacks_subscribe(1, cupti_subscriber);
  cupti_subscribers_driver_memcpy_dtoh_callbacks_subscribe(1, cupti_subscriber);
  cupti_subscribers_driver_memcpy_callbacks_subscribe(1, cupti_subscriber);
  cupti_subscribers_runtime_kernel_callbacks_subscribe(1, cupti_subscriber);
  cupti_subscribers_runtime_memcpy_callbacks_subscribe(1, cupti_subscriber);
  cupti_subscribers_resource_module_subscribe(1, cupti_subscriber);
  cupti_subscribers_resource_context_subscribe(1, cupti_subscriber);

  // XXX(Keren): timestamps for sync are captured on CPU
  //cupti_subscribers_driver_sync_callbacks_subscribe(1, cupti_subscriber);
  //cupti_subscribers_runtime_sync_callbacks_subscribe(1, cupti_subscriber);
}


void
cupti_callbacks_unsubscribe
(
)
{
  cupti_load_callback = 0;
  cupti_unload_callback = 0;
  cupti_correlation_callback = 0;

  HPCRUN_CUPTI_CALL(cuptiUnsubscribe, (cupti_subscriber));

  cupti_subscribers_driver_memcpy_htod_callbacks_subscribe(0, cupti_subscriber);
  cupti_subscribers_driver_memcpy_dtoh_callbacks_subscribe(0, cupti_subscriber);
  cupti_subscribers_driver_memcpy_callbacks_subscribe(0, cupti_subscriber);
  cupti_subscribers_driver_sync_callbacks_subscribe(0, cupti_subscriber);
  cupti_subscribers_driver_kernel_callbacks_subscribe(0, cupti_subscriber);
  cupti_subscribers_runtime_memcpy_callbacks_subscribe(0, cupti_subscriber);
  cupti_subscribers_runtime_sync_callbacks_subscribe(0, cupti_subscriber);
  cupti_subscribers_runtime_kernel_callbacks_subscribe(0, cupti_subscriber);
  cupti_subscribers_resource_module_subscribe(0, cupti_subscriber);
  cupti_subscribers_resource_context_subscribe(0, cupti_subscriber);
}

#else

void
cupti_callbacks_subscribe
(
 void
)
{
  cupti_load_callback = cupti_load_callback_cuda;
  cupti_unload_callback = cupti_unload_callback_cuda;
  cupti_correlation_callback = gpu_application_thread_correlation_callback;

  HPCRUN_CUPTI_CALL(cuptiSubscribe, (&cupti_subscriber,
                   (CUpti_CallbackFunc) cupti_subscriber_callback_cuda,
                   (void *) NULL));

  HPCRUN_CUPTI_CALL(cuptiEnableDomain,
                   (1, cupti_subscriber, CUPTI_CB_DOMAIN_DRIVER_API));

  HPCRUN_CUPTI_CALL(cuptiEnableDomain,
                   (1, cupti_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));

  HPCRUN_CUPTI_CALL(cuptiEnableDomain,
                   (1, cupti_subscriber, CUPTI_CB_DOMAIN_RESOURCE));
}


void
cupti_callbacks_unsubscribe
(
)
{
  cupti_load_callback = 0;
  cupti_unload_callback = 0;
  cupti_correlation_callback = 0;

  HPCRUN_CUPTI_CALL(cuptiUnsubscribe, (cupti_subscriber));

  HPCRUN_CUPTI_CALL(cuptiEnableDomain,
                   (0, cupti_subscriber, CUPTI_CB_DOMAIN_DRIVER_API));

  HPCRUN_CUPTI_CALL(cuptiEnableDomain,
                   (0, cupti_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));

  HPCRUN_CUPTI_CALL(cuptiEnableDomain,
                   (0, cupti_subscriber, CUPTI_CB_DOMAIN_RESOURCE));
}

#endif

void
cupti_correlation_enable
(
)
{
  TMSG(CUPTI, "Enter cupti_correlation_enable");
  cupti_correlation_enabled = true;

  // For unknown reasons, external correlation ids do not return using
  // cuptiActivityEnableContext
  HPCRUN_CUPTI_CALL(cuptiActivityEnable,
                   (CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));

  TMSG(CUPTI, "Exit cupti_correlation_enable");
}


void
cupti_correlation_disable
(
)
{
  TMSG(CUPTI, "Enter cupti_correlation_disable");

  if (cupti_correlation_enabled) {
    HPCRUN_CUPTI_CALL(cuptiActivityDisable,
                     (CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
    cupti_correlation_enabled = false;
  }

  TMSG(CUPTI, "Exit cupti_correlation_disable");
}


void
cupti_pc_sampling_enable
(
 CUcontext context,
 int frequency
)
{
  TMSG(CUPTI, "Enter cupti_pc_sampling_enable");

  CUpti_ActivityPCSamplingConfig config;
  config.samplingPeriod = 0;
  config.samplingPeriod2 = frequency;
  config.size = sizeof(config);

  int required;
  int retval = cuda_global_pc_sampling_required(&required);

  if (retval == 0) { // only turn something on if success determining mode

    if (!required) {
      HPCRUN_CUPTI_CALL(cuptiActivityConfigurePCSampling, (context, &config));

      HPCRUN_CUPTI_CALL(cuptiActivityEnableContext,
                        (context, CUPTI_ACTIVITY_KIND_PC_SAMPLING));
     } else {
      HPCRUN_CUPTI_CALL(cuptiActivityEnable, (CUPTI_ACTIVITY_KIND_PC_SAMPLING));
     }
  }

  TMSG(CUPTI, "Exit cupti_pc_sampling_enable");
}


void
cupti_pc_sampling_disable
(
 CUcontext context
)
{
  HPCRUN_CUPTI_CALL(cuptiActivityDisableContext,
                   (context, CUPTI_ACTIVITY_KIND_PC_SAMPLING));
}



//******************************************************************************
// finalizer
//******************************************************************************


void
cupti_activity_flush
(
)
{
  if (cupti_thread_activity_flag_get()) {
    cupti_thread_activity_flag_unset();
    FLUSH_ALARM_INIT();
    if (!FLUSH_ALARM_FIRED()) {
      FLUSH_ALARM_SET();
      HPCRUN_CUPTI_CALL_NOERROR(cuptiActivityFlushAll, (CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
      FLUSH_ALARM_TEST();
      FLUSH_ALARM_CLEAR();
    }
    FLUSH_ALARM_FINI();
  }

  TMSG(CUPTI, "Exit cupti_activity_flush");
}


void
cupti_device_flush(void *args, int how)
{
  TMSG(CUPTI, "Enter cupti_device_flush");

  cupti_activity_flush();

  gpu_application_thread_process_activities();

  TMSG(CUPTI, "Exit CUPTI device flush");

#ifdef NEW_CUPTI
  cupti_range_thread_last();

  TMSG(CUPTI_CCT, "CUPTI unwind time: %.2f\n", unwind_time / 1000000000.0);
  TMSG(CUPTI_CCT, "CUPTI Total cct unwinds %lu, correct unwinds %lu, fast unwinds %lu, slow unwinds %lu\n",
    total_unwinds, correct_unwinds, fast_unwinds, slow_unwinds);
  if (debug_flag_get(DBG_PREFIX(CUPTI_CCT))) {
    cupti_cct_map_stats();
  }
#endif
}

bool
cupti_thread_activity_flag_get()
{
  return cupti_thread_activity_flag;
}

void
cupti_thread_activity_flag_set()
{
  cupti_thread_activity_flag = true;
}

void
cupti_thread_activity_flag_unset()
{
  cupti_thread_activity_flag = false;
}


bool
cupti_runtime_api_flag_get()
{
  return cupti_runtime_api_flag;
}


void
cupti_runtime_api_flag_unset()
{
  cupti_runtime_api_flag = false;
}


void
cupti_runtime_api_flag_set()
{
  cupti_runtime_api_flag = true;
}


cct_node_t *
cupti_kernel_ph_get
(
 void
)
{
  return cupti_kernel_ph;
}


void
cupti_kernel_ph_set
(
 cct_node_t *node
)
{
  cupti_kernel_ph = node;
}


cct_node_t *
cupti_trace_ph_get
(
 void
)
{
  return cupti_trace_ph;
}


void
cupti_trace_ph_set
(
 cct_node_t *node
)
{
  cupti_trace_ph = node;
}


uint64_t
cupti_runtime_correlation_id_get
(
 void
)
{
  return cupti_runtime_correlation_id;
}


void
cupti_runtime_correlation_id_set
(
 uint64_t correlation_id
)
{
  cupti_runtime_correlation_id = correlation_id;
}


uint64_t
cupti_driver_correlation_id_get
(
 void
)
{
  return cupti_driver_correlation_id;
}


void
cupti_driver_correlation_id_set
(
 uint64_t correlation_id
)
{
  cupti_driver_correlation_id = correlation_id;
}


void
cupti_fast_unwind_set
(
 bool fast_unwind
)
{
  cupti_fast_unwind = fast_unwind;
}


bool
cupti_fast_unwind_get
(
)
{
  return cupti_fast_unwind;
}


void
cupti_correlation_threshold_set
(
 int32_t correlation_threshold
)
{
  cupti_correlation_threshold = correlation_threshold;
}


int32_t
cupti_correlation_threshold_get
(
)
{
  return cupti_correlation_threshold;
}


void
cupti_backoff_base_set
(
 int32_t backoff_base
)
{
  cupti_backoff_base = backoff_base;
}


int32_t
cupti_backoff_base_get
(
)
{
  return cupti_backoff_base;
}


void
cupti_sync_yield_set
(
 bool sync_yield
)
{
  cupti_sync_yield = sync_yield;
}


bool
cupti_sync_yield_get
(
)
{
  return cupti_sync_yield;
}


void
cupti_correlation_id_push(uint64_t id)
{
  HPCRUN_CUPTI_CALL(cuptiActivityPushExternalCorrelationId,
    (CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, id));
}


uint64_t
cupti_correlation_id_pop()
{
  uint64_t id;
  HPCRUN_CUPTI_CALL(cuptiActivityPopExternalCorrelationId,
    (CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, &id));
  return id;
}


void
cupti_device_init()
{
  cupti_thread_activity_flag = false;
  cupti_runtime_api_flag = false;

  // FIXME: Callback shutdown currently disabled to handle issues with fork()
  // See the comment preceeding sample-sources/nvidia.c:process_event_list for details.

  // cupti_correlation_enabled = false;
  // cupti_pc_sampling_enabled = false;

  // cupti_correlation_callback = cupti_correlation_callback_dummy;

  // cupti_error_callback = cupti_error_callback_dummy;

  // cupti_activity_enabled.buffer_request = 0;
  // cupti_activity_enabled.buffer_complete = 0;

  // cupti_load_callback = 0;

  // cupti_unload_callback = 0;
}


void
cupti_device_shutdown(void *args, int how)
{
  TMSG(CUPTI, "Enter cupti_device_shutdown");

  cupti_callbacks_unsubscribe();
  cupti_device_flush(args, how);

#ifdef NEW_CUPTI
  if (cupti_range_mode_get() != CUPTI_RANGE_MODE_NONE) {
    // Collect pc samples for all contexts in a range
    // XXX(Keren): There might be some problems in some apps,
    // since CUPTI does not support multiple contexts in the same range
    cupti_range_last();

    // Wait until operations are drained
    // Operation channel is FIFO
    atomic_bool wait;
    atomic_store(&wait, true);
    gpu_activity_t gpu_activity;
    memset(&gpu_activity, 0, sizeof(gpu_activity_t));

    gpu_activity.kind = GPU_ACTIVITY_FLUSH;
    gpu_activity.details.flush.wait = &wait;

    if (cupti_pc_sampling_frequency_get() != CUPTI_PC_SAMPLING_PERIOD_NULL) {
      gpu_operation_multiplexer_push(NULL, NULL, &gpu_activity);
    }

    // TODO(Keren): wait for only a few seconds
    // Special case: monitoring an application without any kernel using gpu=nvidia,pc
    while (atomic_load(&wait)) {}
  }
#endif

  if (cupti_pc_sampling_frequency_get() != CUPTI_PC_SAMPLING_PERIOD_NULL) {
    // Terminate monitor thread
    gpu_operation_multiplexer_fini();
  }

  TMSG(CUPTI, "Exit cupti_device_shutdown");
}

