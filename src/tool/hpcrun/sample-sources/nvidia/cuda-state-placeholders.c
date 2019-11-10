#include "cuda-state-placeholders.h"

#include <lib/prof-lean/placeholders.h>
#include <hpcrun/fnbounds/fnbounds_interface.h>
#include <hpcrun/safe-sampling.h>
#include <hpcrun/hpcrun-initializers.h>

cuda_placeholders_t cuda_placeholders;

void 
cuda_copy
(
  void
)
{
  //TODO(Keren): extend state to copyin and copyout
}


void 
cuda_alloc
(
  void
)
{
}


void 
cuda_delete
(
  void
)
{
}


void 
cuda_kernel
(
  void
)
{
}


void
cuda_sync
(
 void
)
{
}



void
cuda_init_placeholders
(
)
{
  init_placeholder(&cuda_placeholders.cuda_copy_state, &cuda_copy);
  init_placeholder(&cuda_placeholders.cuda_alloc_state, &cuda_alloc);
  init_placeholder(&cuda_placeholders.cuda_delete_state, &cuda_delete);
  init_placeholder(&cuda_placeholders.cuda_kernel_state, &cuda_kernel);
  init_placeholder(&cuda_placeholders.cuda_sync_state, &cuda_sync);
}
