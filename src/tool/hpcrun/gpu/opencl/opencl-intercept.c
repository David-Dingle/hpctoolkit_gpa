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
// Copyright ((c)) 2002-2020, Rice University
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

//******************************************************************************
// system includes
//******************************************************************************

#include <inttypes.h>



//******************************************************************************
// local includes
//******************************************************************************

#include <hpcrun/gpu/instrumentation/opencl-instrumentation.h>
#include <hpcrun/gpu/gpu-metrics.h>
#include <hpcrun/memory/hpcrun-malloc.h>
#include <hpcrun/messages/messages.h>
#include <lib/prof-lean/hpcrun-gotcha.h>
#include <lib/prof-lean/hpcrun-opencl.h>
#include <lib/prof-lean/stdatomic.h>

#include "opencl-api.h"
#include "opencl-intercept.h"
#include "opencl-memory-manager.h"



//******************************************************************************
// local data
//******************************************************************************

#ifndef HPCRUN_STATIC_LINK
static gotcha_wrappee_handle_t clBuildProgram_handle;
static gotcha_wrappee_handle_t clCreateProgramWithSource_handle;
static gotcha_wrappee_handle_t clCreateCommandQueue_handle;
static gotcha_wrappee_handle_t clEnqueueNDRangeKernel_handle;
static gotcha_wrappee_handle_t clEnqueueReadBuffer_handle;
static gotcha_wrappee_handle_t clEnqueueWriteBuffer_handle;
static atomic_long correlation_id;
static char *debugInfoFullFileName;


#define CL_PROGRAM_DEBUG_INFO_SIZES_INTEL 0x4101
#define CL_PROGRAM_DEBUG_INFO_INTEL       0x4100



//******************************************************************************
// private operations
//******************************************************************************

static void
opencl_intercept_initialize
(
  void
)
{
  atomic_store(&correlation_id, 0);
}


static uint64_t
getCorrelationId
(
  void
)
{
  return atomic_fetch_add(&correlation_id, 1);
}


static void
setDebugInfoFullFileName
(
	char *fileName
)
{
	if (debugInfoFullFileName == NULL) {
		debugInfoFullFileName = fileName;	
	}
}


static void
initializeKernelCallBackInfo
(
  cl_kernel_callback_t *kernel_cb,
  uint64_t correlation_id
)
{
  kernel_cb->correlation_id = correlation_id;
  kernel_cb->type = kernel; 
}


static void
initializeMemoryCallBackInfo
(
  cl_memory_callback_t *mem_transfer_cb,
  uint64_t correlation_id,
  size_t size,
  bool fromHostToDevice
)
{
  mem_transfer_cb->correlation_id = correlation_id;
  mem_transfer_cb->type = (fromHostToDevice) ? memcpy_H2D: memcpy_D2H; 
  mem_transfer_cb->size = size;
  mem_transfer_cb->fromHostToDevice = fromHostToDevice;
  mem_transfer_cb->fromDeviceToHost = !fromHostToDevice;
}

static char*
getKernelNameFromSourceCode
(
	const char *kernelSourceCode
)
{
	char *kernelCode_copy = (char*)hpcrun_malloc(sizeof(kernelSourceCode));
	strcpy(kernelCode_copy, kernelSourceCode);
	char *token = strtok(kernelCode_copy, " ");
	while (token != NULL) {
		if (strcmp(token, "void") == 0) { // not searching for kernel because "supported\n#endif\nkernel"
			token = strtok(NULL, " ");
			printf("kernel name: %s", token);
			return token;
		}
		token = strtok(NULL, " ");
	}
	return NULL;
}


static cl_program
clCreateProgramWithSource_wrapper
(
 cl_context context,
 cl_uint count,
 const char** strings,
 const size_t* lengths,
 cl_int* errcode_ret
)
{
	ETMSG(OPENCL, "inside clCreateProgramWithSource_wrapper");

	FILE *f_ptr;
	for (int i = 0; i < (int)count; i++) {
		// what if a single file has multiple kernels?
		// we need to add logic to get filenames by reading the strings contents
		char fileno = '0' + (i + 1); // right now we are naming the files as index numbers
		// using malloc instead of hpcrun_malloc gives extra garbage characters in file name
		char *filename = (char*)hpcrun_malloc(sizeof(fileno) + 1);
		*filename = fileno + '\0';
		f_ptr = fopen(filename, "w");
		fwrite(strings[i], lengths[i], 1, f_ptr);
	}
	fclose(f_ptr);
	
	clcreateprogramwithsource_t clCreateProgramWithSource_wrappee =
		GOTCHA_GET_TYPED_WRAPPEE(clCreateProgramWithSource_handle, clcreateprogramwithsource_t);
	return clCreateProgramWithSource_wrappee(context, count, strings, lengths, errcode_ret);
}


// we are dumping the debuginfo temporarily since the binary does not have debugsection
// poorly written code: FIXME
static char*
dumpIntelGPUBinary(cl_program program) {
	int device_count = 1;
	cl_int status = CL_SUCCESS;
	size_t *binary_size = (size_t*)hpcrun_malloc(sizeof(size_t) * device_count);

	status = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,	sizeof(size_t), binary_size, NULL);
	assert(status == CL_SUCCESS);
	uint8_t **binary = (uint8_t**)hpcrun_malloc(device_count * sizeof(uint8_t*));
	for (size_t i = 0; i < device_count; ++i) {
		binary[i] = (uint8_t*)hpcrun_malloc(binary_size[i] * sizeof(uint8_t));
	}

	status = clGetProgramInfo(program, CL_PROGRAM_BINARIES, device_count * sizeof(uint8_t*), binary, NULL);
	assert(status == CL_SUCCESS);

	FILE *bin_ptr;
	bin_ptr = fopen("opencl_main.gpubin", "wb");
	fwrite(binary[0], binary_size[0], 1, bin_ptr);

  // SECOND
	size_t *debug_info_size = (size_t*)hpcrun_malloc(sizeof(size_t) * device_count);

	status = clGetProgramInfo(program, CL_PROGRAM_DEBUG_INFO_SIZES_INTEL,	sizeof(size_t), debug_info_size, NULL);
	assert(status == CL_SUCCESS);
	uint8_t **debug_info = (uint8_t**)hpcrun_malloc(device_count * sizeof(uint8_t*));
	for (size_t i = 0; i < device_count; ++i) {
		debug_info[i] = (uint8_t*)hpcrun_malloc(debug_info_size[i] * sizeof(uint8_t));
	}

	status = clGetProgramInfo(program, CL_PROGRAM_DEBUG_INFO_INTEL, device_count * sizeof(uint8_t*), debug_info, NULL);
	assert(status == CL_SUCCESS);

	char *debuginfoFileName = "opencl_main.debuginfo";
	bin_ptr = fopen(debuginfoFileName, "wb");
	fwrite(debug_info[0], debug_info_size[0], 1, bin_ptr);
	fclose(bin_ptr);
  ETMSG(OPENCL, "Intel GPU files dumped successfully");
	return realpath(debuginfoFileName, NULL);
}


static void
clBuildProgramCallback
(
	cl_program program,
	void* user_data
)
{
	char* debugInfoFullFileName = dumpIntelGPUBinary(program);
	setDebugInfoFullFileName(debugInfoFullFileName);
}


// one downside of this appproach is that we may override the callback provided by user
static cl_int
clBuildProgram_wrapper
(
 cl_program program,
 cl_uint num_devices,
 const cl_device_id* device_list,
 const char* options,
 void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
 void* user_data
)
{
  ETMSG(OPENCL, "inside clBuildProgram_wrapper");
  clbuildprogram_t clBuildProgram_wrappee = 
    GOTCHA_GET_TYPED_WRAPPEE(clBuildProgram_handle, clbuildprogram_t);

	char optionsWithDebugFlag[] = " -gline-tables-only ";
	if (options != NULL) {
		strcat(optionsWithDebugFlag, options);
	}
  return clBuildProgram_wrappee(program, num_devices, device_list, (const char*)optionsWithDebugFlag, clBuildProgramCallback, user_data);
}


static cl_command_queue
clCreateCommandQueue_wrapper
(
  cl_context context,
  cl_device_id device,
  cl_command_queue_properties properties,
  cl_int *errcode_ret
)
{
  // enabling profiling
  properties |= (cl_command_queue_properties)CL_QUEUE_PROFILING_ENABLE; 

  clqueue_t clCreateCommandQueue_wrappee = 
    GOTCHA_GET_TYPED_WRAPPEE(clCreateCommandQueue_handle, clqueue_t);

  return clCreateCommandQueue_wrappee(context, device, properties, errcode_ret);
}


static cl_int
clEnqueueNDRangeKernel_wrapper
(
  cl_command_queue command_queue,
  cl_kernel ocl_kernel,
  cl_uint work_dim,
  const size_t *global_work_offset, 
  const size_t *global_work_size,
  const size_t *local_work_size,
  cl_uint num_events_in_wait_list,
  const cl_event *event_wait_list,
  cl_event *event
)
{
  uint64_t correlation_id = getCorrelationId();
  opencl_object_t *kernel_info = opencl_malloc();
  kernel_info->kind = OPENCL_KERNEL_CALLBACK;
  cl_kernel_callback_t *kernel_cb = &(kernel_info->details.ker_cb);
  initializeKernelCallBackInfo(kernel_cb, correlation_id);
  cl_event my_event;
  cl_event *eventp;
  if (!event) {
    kernel_info->isInternalClEvent = true;
    eventp = &my_event;
  } else {
    eventp = event;
    kernel_info->isInternalClEvent = false;
  }
  clkernel_t clEnqueueNDRangeKernel_wrappee = 
    GOTCHA_GET_TYPED_WRAPPEE(clEnqueueNDRangeKernel_handle, clkernel_t);
  cl_int return_status = 
    clEnqueueNDRangeKernel_wrappee(command_queue, ocl_kernel, work_dim, 
				   global_work_offset, global_work_size, 
				   local_work_size, num_events_in_wait_list, 
				   event_wait_list, eventp);

  ETMSG(OPENCL, "registering callback for type: kernel. " 
	"Correlation id: %"PRIu64 "", correlation_id);

  opencl_subscriber_callback(kernel_cb->type, kernel_cb->correlation_id);
  clSetEventCallback_wrapper(*eventp, CL_COMPLETE, 
			     &opencl_activity_completion_callback, kernel_info);
  return return_status;
}


static cl_int
clEnqueueReadBuffer_wrapper
(
  cl_command_queue command_queue,
  cl_mem buffer,
  cl_bool blocking_read,
  size_t offset,
  size_t cb,
  void *ptr,
  cl_uint num_events_in_wait_list,
  const cl_event *event_wait_list,
  cl_event *event
)
{
  uint64_t correlation_id = getCorrelationId();
  opencl_object_t *mem_info = opencl_malloc();
  mem_info->kind = OPENCL_MEMORY_CALLBACK;
  cl_memory_callback_t *mem_transfer_cb = &(mem_info->details.mem_cb);
  initializeMemoryCallBackInfo(mem_transfer_cb, correlation_id, cb, false);
  cl_event my_event;
  cl_event *eventp;
  if (!event) {
    mem_info->isInternalClEvent = true;
    eventp = &my_event;
  } else {
    eventp = event;
    mem_info->isInternalClEvent = false;
  }
  clreadbuffer_t clEnqueueReadBuffer_wrappee = 
    GOTCHA_GET_TYPED_WRAPPEE(clEnqueueReadBuffer_handle, clreadbuffer_t);
  cl_int return_status = 
    clEnqueueReadBuffer_wrappee(command_queue, buffer, blocking_read, offset, 
				cb, ptr, num_events_in_wait_list, 
				event_wait_list, eventp);

  ETMSG(OPENCL, "registering callback for type: D2H. " 
	"Correlation id: %"PRIu64 "", correlation_id);
  ETMSG(OPENCL, "%d(bytes) of data being transferred from device to host", 
	(long)cb);

  opencl_subscriber_callback(mem_transfer_cb->type, 
			     mem_transfer_cb->correlation_id);

  clSetEventCallback_wrapper(*eventp, CL_COMPLETE, 
			     &opencl_activity_completion_callback, mem_info);

  return return_status;
}


static cl_int
clEnqueueWriteBuffer_wrapper
(
  cl_command_queue command_queue,
  cl_mem buffer,
  cl_bool blocking_write,
  size_t offset,
  size_t cb,
  const void *ptr,
  cl_uint num_events_in_wait_list,
  const cl_event *event_wait_list,
  cl_event *event
)
{
  uint64_t correlation_id = getCorrelationId();
  opencl_object_t *mem_info = opencl_malloc();
  mem_info->kind = OPENCL_MEMORY_CALLBACK;
  cl_memory_callback_t *mem_transfer_cb = &(mem_info->details.mem_cb);
  initializeMemoryCallBackInfo(mem_transfer_cb, correlation_id, cb, true);
  cl_event my_event;
  cl_event *eventp;
  if (!event) {
    mem_info->isInternalClEvent = true;
    eventp = &my_event;
  } else {
    eventp = event;
    mem_info->isInternalClEvent = false;
  }
  clwritebuffer_t clEnqueueWriteBuffer_wrappee = 
    GOTCHA_GET_TYPED_WRAPPEE(clEnqueueWriteBuffer_handle, clwritebuffer_t);
  cl_int return_status = 
    clEnqueueWriteBuffer_wrappee(command_queue, buffer, blocking_write, offset,
				 cb, ptr, num_events_in_wait_list, 
				 event_wait_list, eventp);

  ETMSG(OPENCL, "registering callback for type: H2D. " 
	"Correlation id: %"PRIu64 "", correlation_id);

  ETMSG(OPENCL, "%d(bytes) of data being transferred from host to device", 
	(long)cb);

  opencl_subscriber_callback(mem_transfer_cb->type, 
			     mem_transfer_cb->correlation_id);

  clSetEventCallback_wrapper(*eventp, CL_COMPLETE, 
			     &opencl_activity_completion_callback, 
			     (void*) mem_info);

  return return_status;
}

#endif



//******************************************************************************
// gotcha variables
//******************************************************************************

#ifndef HPCRUN_STATIC_LINK
static gotcha_binding_t opencl_bindings[] = {
  {
    "clBuildProgram",
    (void*) clBuildProgram_wrapper,
    &clBuildProgram_handle
  },
  {
    "clCreateProgramWithSource",
    (void*) clCreateProgramWithSource_wrapper,
    &clCreateProgramWithSource_handle
  },
  {
    "clCreateCommandQueue",
    (void*) clCreateCommandQueue_wrapper,
    &clCreateCommandQueue_handle
  },
  {
    "clEnqueueNDRangeKernel",
    (void*)clEnqueueNDRangeKernel_wrapper,
    &clEnqueueNDRangeKernel_handle
  },
  {
    "clEnqueueReadBuffer",
    (void*) clEnqueueReadBuffer_wrapper,
    &clEnqueueReadBuffer_handle
  },
  {
    "clEnqueueWriteBuffer",
    (void*) clEnqueueWriteBuffer_wrapper,
    &clEnqueueWriteBuffer_handle
  }
};
#endif



//******************************************************************************
// interface operations
//******************************************************************************

char*
getDebugInfoFullFileName
(
	void
)
{
	return debugInfoFullFileName;
}


void
opencl_intercept_setup
(
  void
)
{
#ifndef HPCRUN_STATIC_LINK
  ETMSG(OPENCL, "setting up opencl intercepts");
	gpu_metrics_KER_BLKINFO_enable();
  enableProfiling();
  gotcha_wrap(opencl_bindings, 4, "opencl_bindings");
  opencl_intercept_initialize();
#endif
}


void
opencl_intercept_teardown
(
  void
)
{
#ifndef HPCRUN_STATIC_LINK
  // not sure if this works
  gotcha_set_priority("opencl_bindings", -1);
#endif
}
