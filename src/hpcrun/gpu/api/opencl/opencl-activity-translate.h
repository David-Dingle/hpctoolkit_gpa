// SPDX-FileCopyrightText: 2002-2024 Rice University
//
// SPDX-License-Identifier: BSD-3-Clause

// -*-Mode: C++;-*- // technically C99

#ifndef opencl_activity_translate_h
#define opencl_activity_translate_h



//******************************************************************************
// local includes
//******************************************************************************

#include "../../activity/gpu-activity.h"
#include "../../../../common/lean/hpcrun-opencl.h"



//*************************** Forward Declarations **************************

typedef struct opencl_object_t opencl_object_t;

//******************************************************************************
// interface operations
//******************************************************************************

void
opencl_activity_translate
(
  gpu_activity_t *ga,
  opencl_object_t *cb_data,
  gpu_interval_t interval
);

#endif  //_OPENCL_ACTIVITY_TRANSLATE_H_
