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

#include <assert.h>
#include <string.h>
#include <math.h>

//******************************************************************************
// local includes
//******************************************************************************

#include <lib/prof-lean/splay-uint64.h>
#include <lib/prof-lean/spinlock.h>

#include <hpcrun/messages/messages.h>
#include <hpcrun/memory/hpcrun-malloc.h>

#include "cupti-cct-map.h"
#include "../gpu-splay-allocator.h"

//******************************************************************************
// macros
//******************************************************************************

#define st_insert				\
  typed_splay_insert(cct)

#define st_lookup				\
  typed_splay_lookup(cct)

#define st_forall				\
  typed_splay_forall(cct)

#define st_delete				\
  typed_splay_delete(cct)

#define st_count				\
  typed_splay_count(cct)

#define st_alloc(free_list)			\
  typed_splay_alloc(free_list, typed_splay_node(cct))

#define st_free(free_list, node)		\
  typed_splay_free(free_list, node)

#undef typed_splay_node
#define typed_splay_node(cct) cupti_cct_map_entry_t 

//******************************************************************************
// type declarations
//******************************************************************************

struct cupti_cct_map_entry_s {
  struct cupti_cct_map_entry_s *left;
  struct cupti_cct_map_entry_s *right;
  uint64_t cct;

  uint64_t sampled_count;
  uint64_t count;
  uint32_t range_id;
}; 


//******************************************************************************
// local data
//******************************************************************************

static cupti_cct_map_entry_t *map_root = NULL;
static cupti_cct_map_entry_t *free_list = NULL;

typed_splay_impl(cct)


static cupti_cct_map_entry_t *
cupti_cct_map_entry_new
(
 cct_node_t *cct,
 uint32_t range_id
)
{
  cupti_cct_map_entry_t *e;
  e = st_alloc(&free_list);

  memset(e, 0, sizeof(cupti_cct_map_entry_t));

  e->cct = (uint64_t)cct;
  e->range_id = range_id;
  e->sampled_count = 1;
  e->count = 1;

  return e;
}


static void
clear_fn_helper
(
 cupti_cct_map_entry_t *entry,
 splay_visit_t visit_type,
 void *args
)
{
  if (visit_type == splay_postorder_visit) {
    st_free(&free_list, entry);
  }
}


static __thread double count_total = 0;
static __thread double depth_total = 0;
static __thread double entry_num = 0;


static void
stats_fn_helper
(
 cupti_cct_map_entry_t *entry,
 splay_visit_t visit_type,
 void *args
)
{
  if (visit_type == splay_postorder_visit) {
    double count = entry->count;
    count_total += count;
    cct_node_t *node = (cct_node_t *)entry->cct;
    int depth = 0;
    while (node != NULL) {
      node = hpcrun_cct_parent(node);
      depth += 1;
    }
    depth_total += depth;
    entry_num += 1;
  }
}

//******************************************************************************
// interface operations
//******************************************************************************

cupti_cct_map_entry_t *
cupti_cct_map_lookup
(
 cct_node_t *cct
)
{
  cupti_cct_map_entry_t *result = st_lookup(&map_root, (uint64_t)cct);

  return result;
}


void
cupti_cct_map_insert
(
 cct_node_t *cct,
 uint32_t range_id
)
{
  cupti_cct_map_entry_t *entry = st_lookup(&map_root, (uint64_t)cct);

  if (entry == NULL) {
    entry = cupti_cct_map_entry_new(cct, range_id);
    st_insert(&map_root, entry);
  }
}


void
cupti_cct_map_clear()
{
  st_forall(map_root, splay_allorder, clear_fn_helper, NULL);
  map_root = NULL;
}


void
cupti_cct_map_stats()
{
  st_forall(map_root, splay_allorder, stats_fn_helper, NULL);
  double depth_mean = depth_total / entry_num;
  TMSG(CUPTI_CCT, "CUPTI Stats count total %f unique call path %f depth mean %f",
    count_total, entry_num, depth_mean);
}


uint32_t
cupti_cct_map_entry_range_id_get
(
 cupti_cct_map_entry_t *entry
)
{
  return entry->range_id;
}


uint64_t
cupti_cct_map_entry_count_get
(
 cupti_cct_map_entry_t *entry
)
{
  return entry->count;
}


uint64_t
cupti_cct_map_entry_sampled_count_get
(
 cupti_cct_map_entry_t *entry
)
{
  return entry->sampled_count;
}


void
cupti_cct_map_entry_range_id_update(cupti_cct_map_entry_t *entry, uint32_t range_id)
{
  entry->range_id = range_id;
}


void
cupti_cct_map_entry_count_increase(cupti_cct_map_entry_t *entry, uint64_t sampled_count, uint64_t count)
{
  entry->sampled_count += sampled_count;
  entry->count += count;
}
