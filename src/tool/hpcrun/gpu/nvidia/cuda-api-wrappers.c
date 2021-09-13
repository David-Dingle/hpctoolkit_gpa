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
// Copyright ((c)) 2002-2021, Rice University
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
//   cuda-api-wrappers.c
//
// Purpose:
//   intercept cuda api calls
//
//***************************************************************************

#include "cuda-api.h"
#include <stdio.h>

CUresult
cuLaunchKernel
(
 CUfunction f,
 unsigned int gridDimX,
 unsigned int gridDimY,
 unsigned int gridDimZ,
 unsigned int blockDimX,
 unsigned int blockDimY,
 unsigned int blockDimZ,
 unsigned int sharedMemBytes,
 CUstream hStream,
 void **kernelParams,
 void **extra
)
{
  cuda_api_enter_callback();
  fprintf(stderr, "here\n");
  CUresult result = hpcrun_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
    blockDimX, blockDimY, blockDimZ,
    sharedMemBytes, hStream, kernelParams, extra);
  cuda_api_exit_callback();
  return result;
}

CUresult
cuMemcpy
(
 CUdeviceptr dst,
 CUdeviceptr src,
 size_t ByteCount
)
{
  cuda_api_enter_callback();
  CUresult result = hpcrun_cuMemcpy(dst, src, ByteCount);
  cuda_api_exit_callback();
}


CUresult
cuMemcpyHtoD_v2
(
 CUdeviceptr dstDevice,
 const void *srcHost,
 size_t ByteCount
)
{
  cuda_api_enter_callback();
  cudaError_t cuda_error = hpcrun_cuMemcpyHtoD_v2(dstDevice, srcHost, ByteCount);
  cuda_api_exit_callback();
  return cuda_error;
}


CUresult
cuMemcpyDtoH_v2
(
 void *dstHost,
 CUdeviceptr srcDevice,
 size_t ByteCount
)
{
  cuda_api_enter_callback();
  cudaError_t cuda_error = hpcrun_cuMemcpyDtoH_v2(dstHost, srcDevice, ByteCount);
  cuda_api_exit_callback();
  return cuda_error;
}


cudaError_t
cudaLaunchKernel
(
 const void *func,
 dim3 gridDim,
 dim3 blockDim,
 void **args,
 size_t sharedMem,
 cudaStream_t stream
)
{
  cuda_api_enter_callback();
  cudaError_t cuda_error = hpcrun_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
  cuda_api_exit_callback();
  return cuda_error;
}


cudaError_t
cudaMemcpy
(
 void *dst,
 const void *src,
 size_t count,
 enum cudaMemcpyKind kind
)
{
  cuda_api_enter_callback();
  cudaError_t cuda_error = hpcrun_cudaMemcpy(dst, src, count, kind);
  cuda_api_exit_callback();
  return cuda_error;
}
