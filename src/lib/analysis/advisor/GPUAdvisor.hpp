// -*-Mode: C++;-*-

// * BeginRiceCopyright *****************************************************
//
// $HeadURL$
// $Id$
//
// --------------------------------------------------------------------------
// Part of HPCToolkit (hpctoolkit.org)
//
// Information about sources of support for research and development of
// HPCToolkit is at 'hpctoolkit.org' and in 'README.Acknowledgments'.
// --------------------------------------------------------------------------
//
// Copyright ((c)) 2002-2018, Rice University
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
//   $HeadURL$
//
// Purpose:
//   [The purpose of this file]
//
// Description:
//   [The set of functions, macros, etc. defined in the file]
//
//***************************************************************************

#ifndef Analysis_Advisor_GPUAdvisor_hpp
#define Analysis_Advisor_GPUAdvisor_hpp

//************************* System Include Files ****************************

#include <iostream>
#include <map>
#include <queue>
#include <stack>
#include <string>
#include <tuple>
#include <vector>

//*************************** User Include Files ****************************

#include <include/uint.h>

#include <lib/binutils/LM.hpp>
#include <lib/cuda/AnalyzeInstruction.hpp>
#include <lib/cuda/DotCFG.hpp>
#include <lib/prof/CallPath-Profile.hpp>
#include <lib/prof/LoadMap.hpp>
#include <lib/prof/Struct-Tree.hpp>

#include "../CCTGraph.hpp"
#include "../MetricNameProfMap.hpp"
#include "GPUArchitecture.hpp"
#include "GPUEstimator.hpp"
#include "GPUOptimizer.hpp"

//*************************** Forward Declarations ***************************

//****************************************************************************

namespace Analysis {

class GPUAdvisor {
 public:
  typedef std::tuple<double, Prof::CCT::ADynNode *, std::string> AdviceTuple;

 public:
  explicit GPUAdvisor(Prof::CallPath::Profile *prof, MetricNameProfMap *metric_name_prof_map)
      : _prof(prof),
        _metric_name_prof_map(metric_name_prof_map),
        _gpu_root(NULL),
        _gpu_kernel(NULL),
        _arch(NULL) {}

  MetricNameProfMap *metric_name_prof_map() { return this->_metric_name_prof_map; }

  void init(const std::string &gpu_arch);

  void configInst(const std::string &lm_name, const std::vector<CudaParse::Function *> &functions);

  void configGPURoot(Prof::CCT::ADynNode *gpu_root, Prof::CCT::ADynNode *gpu_kernel);

  void blame(CCTBlames &cct_blames, std::map<uint, std::vector<std::pair<VMA, VMA>>>* blames);

  void advise(const CCTBlames &cct_blames);

  std::vector<AdviceTuple> get_advice();

  ~GPUAdvisor() {
    for (auto *optimizer : _code_optimizers) {
      delete optimizer;
    }
    for (auto *optimizer : _parallel_optimizers) {
      delete optimizer;
    }
    for (auto *optimizer : _binary_optimizers) {
      delete optimizer;
    }
    for (auto *estimator : _estimators) {
      delete estimator;
    }
    if (_arch) {
      delete _arch;
    }
  }

 private:
  typedef std::map<double, std::vector<GPUOptimizer *>, std::greater<double>> OptimizerRank;

  typedef std::map<int, std::map<int, std::vector<std::vector<CudaParse::Block *>>>> CCTEdgePathMap;

  struct VMAProperty {
    VMA vma;
    Prof::CCT::ADynNode *prof_node;
    CudaParse::InstructionStat *inst;
    CudaParse::Function *function;
    CudaParse::Block *block;
    int latency_lower;
    int latency_upper;
    int latency_issue;

    VMAProperty()
        : vma(0),
          prof_node(NULL),
          inst(NULL),
          function(NULL),
          block(NULL),
          latency_lower(0),
          latency_upper(0),
          latency_issue(0) {}
  };

  typedef std::map<VMA, VMAProperty> VMAPropertyMap;

  typedef std::map<VMA, Prof::Struct::Stmt *> VMAStructureMap;

  enum TrackType {
    TRACK_REG = 0,
    TRACK_PRED_REG = 1,
    TRACK_PREDICATE = 2,
    TRACK_UNIFORM = 3,
    TRACK_BARRIER = 4
  };

 private:
  void attributeBlameMetric(int mpi_rank, int thread_id, Prof::CCT::ANode *node,
                            const std::string &blame_name, double blame);

  void initCCTDepGraph(int mpi_rank, int thread_id, CCTGraph<Prof::CCT::ADynNode *> &cct_dep_graph);

  void pruneCCTDepGraphOpcode(int mpi_rank, int thread_id,
                              CCTGraph<Prof::CCT::ADynNode *> &cct_dep_graph);

  void pruneCCTDepGraphBarrier(int mpi_rank, int thread_id,
                               CCTGraph<Prof::CCT::ADynNode *> &cct_dep_graph);

  void pruneCCTDepGraphLatency(int mpi_rank, int thread_id,
                               CCTGraph<Prof::CCT::ADynNode *> &cct_dep_graph,
                               CCTEdgePathMap &cct_edge_path_map);

  void pruneCCTDepGraphExecution(int mpi_rank, int thread_id,
                                 CCTGraph<Prof::CCT::ADynNode *> &cct_dep_graph,
                                 CCTEdgePathMap &cct_edge_path_map);

  void pruneCCTDepGraphBranch(int mpi_rank, int thread_id,
                              CCTGraph<Prof::CCT::ADynNode *> &cct_dep_graph,
                              CCTEdgePathMap &cct_edge_path_map);

  void trackDep(int to_vma, int from_vma, int reg, CudaParse::Block *to_block,
                CudaParse::Block *from_block, int latency_issue, int latency,
                std::set<CudaParse::Block *> &visited_blocks, std::vector<CudaParse::Block *> &path,
                std::vector<std::vector<CudaParse::Block *>> &paths, TrackType track_type,
                bool fixed, int barrier_threshold);

  void trackDepInit(int to_vma, int from_vma, int dst, CCTEdgePathMap &cct_edge_path_map,
                    TrackType track_type, bool fixed, int barrier_threshold);

  double computePathInsts(int mpi_rank, int thread_id, int from_vma, int to_vma,
                          std::vector<CudaParse::Block *> &path);

  void reverseRatio(std::map<Prof::CCT::ADynNode *, double> &distance,
                    std::map<Prof::CCT::ADynNode *, double> &insts);

  double computeEfficiency(int mpi_rank, int thread_id, CudaParse::InstructionStat *inst,
                           Prof::CCT::ADynNode *node);

  double computePredTrue(int mpi_rank, int thread_id, CudaParse::InstructionStat *inst,
                         Prof::CCT::ADynNode *node);

  std::pair<std::string, std::string> detailizeExecBlame(CudaParse::InstructionStat *from_inst,
                                                         CudaParse::InstructionStat *to_inst);

  std::pair<std::string, std::string> detailizeMemBlame(CudaParse::InstructionStat *from_inst);

  void blameCCTDepGraph(int mpi_rank, int thread_id, CCTGraph<Prof::CCT::ADynNode *> &cct_dep_graph,
                        CCTEdgePathMap &cct_edge_path_map, InstBlames &inst_blames,
                        std::map<uint, std::vector<std::pair<VMA, VMA>>>* blames);

  void detailizeInstBlames(InstBlames &inst_blames);

  void overlayInstBlames(InstBlames &inst_blames, KernelBlame &kernel_blame);

  KernelStats readKernelStats(int mpi_rank, int thread_id);

  void concatAdvice(const OptimizerRank &optimizer_rank);

  // Helper functions
  int demandNodeMetric(int mpi_rank, int thread_id, Prof::CCT::ADynNode *node);

  std::string debugInstOffset(int vma);

  void debugInstDepGraph();

  void debugCCTDepPaths(CCTEdgePathMap &cct_edge_path_map);

  void debugCCTDepGraphSummary(int mpi_rank, int thread_id,
                               CCTGraph<Prof::CCT::ADynNode *> &cct_dep_graph);

  void debugCCTDepGraph(int mpi_rank, int thread_id,
                        CCTGraph<Prof::CCT::ADynNode *> &cct_dep_graph);

  void debugCCTDepGraphNoPath(int mpi_rank, int thread_id,
                              CCTGraph<Prof::CCT::ADynNode *> &cct_dep_graph);

  void debugCCTDepGraphStallExec(int mpi_rank, int thread_id,
                                 CCTGraph<Prof::CCT::ADynNode *> &cct_dep_graph);

  void debugCCTDepGraphSinglePath(CCTGraph<Prof::CCT::ADynNode *> &cct_dep_graph);

  void debugInstBlames(InstBlames &inst_blames);

 private:
  std::string _inst_metric;
  std::string _stall_metric;
  std::string _issue_metric;

  // stl
  std::string _invalid_stall_metric;
  std::string _tex_stall_metric;
  std::string _ifetch_stall_metric;
  std::string _pipe_bsy_stall_metric;
  std::string _mem_thr_stall_metric;
  std::string _nosel_stall_metric;
  std::string _other_stall_metric;
  std::string _sleep_stall_metric;
  std::string _cmem_stall_metric;
  std::string _none_stall_metric;

  // lat
  std::string _invalid_lat_metric;
  std::string _tex_lat_metric;
  std::string _ifetch_lat_metric;
  std::string _pipe_bsy_lat_metric;
  std::string _mem_thr_lat_metric;
  std::string _nosel_lat_metric;
  std::string _other_lat_metric;
  std::string _sleep_lat_metric;
  std::string _cmem_lat_metric;
  std::string _none_lat_metric;

  // stl
  std::string _exec_dep_stall_metric;
  std::string _exec_dep_dep_stall_metric;
  std::string _exec_dep_sche_stall_metric;
  std::string _exec_dep_smem_stall_metric;
  std::string _exec_dep_cmem_stall_metric;
  std::string _exec_dep_war_stall_metric;
  std::string _exec_dep_ind_stall_metric;
  std::string _mem_dep_stall_metric;
  std::string _mem_dep_gmem_stall_metric;
  std::string _mem_dep_cmem_stall_metric;
  std::string _mem_dep_tmem_stall_metric;
  std::string _mem_dep_lmem_stall_metric;
  std::string _sync_stall_metric;

  // lat
  std::string _exec_dep_lat_metric;
  std::string _exec_dep_dep_lat_metric;
  std::string _exec_dep_sche_lat_metric;
  std::string _exec_dep_smem_lat_metric;
  std::string _exec_dep_cmem_lat_metric;
  std::string _exec_dep_war_lat_metric;
  std::string _exec_dep_ind_lat_metric;
  std::string _mem_dep_lat_metric;
  std::string _mem_dep_gmem_lat_metric;
  std::string _mem_dep_cmem_lat_metric;
  std::string _mem_dep_tmem_lat_metric;
  std::string _mem_dep_lmem_lat_metric;
  std::string _sync_lat_metric;

  // [<stl, lat>]
  std::vector<std::pair<std::string, std::string>> _inst_metrics;
  std::vector<std::pair<std::string, std::string>> _dep_metrics;

  // inst
  std::string _inst_exe_metric;
  std::string _inst_exe_pred_metric;

  // branch
  std::string _branch_div_metric;
  std::string _branch_exe_metric;

  // global
  std::string _gmem_cache_load_trans_metric;
  std::string _gmem_uncache_load_trans_metric;
  std::string _gmem_cache_store_trans_metric;

  std::string _gmem_cache_load_trans_theor_metric;
  std::string _gmem_uncache_load_trans_theor_metric;
  std::string _gmem_cache_store_trans_theor_metric;

  // shared
  std::string _smem_load_trans_metric;
  std::string _smem_store_trans_metric;
  std::string _smem_load_trans_theor_metric;
  std::string _smem_store_trans_theor_metric;

  Prof::CallPath::Profile *_prof;
  MetricNameProfMap *_metric_name_prof_map;

  Prof::CCT::ADynNode *_gpu_root;
  Prof::CCT::ADynNode *_gpu_kernel;

  std::map<int, int> _function_offset;

  CCTGraph<CudaParse::InstructionStat *> _inst_dep_graph;
  VMAPropertyMap _vma_prop_map;
  VMAStructureMap _vma_struct_map;

  std::vector<GPUOptimizer *> _code_optimizers;
  std::vector<GPUOptimizer *> _parallel_optimizers;
  std::vector<GPUOptimizer *> _binary_optimizers;
  std::vector<GPUEstimator *> _estimators;

  GPUArchitecture *_arch;

  KernelStats _kernel_stats;

  std::vector<AdviceTuple> _advice;
  std::stringstream _output;

 private:
  const size_t _top_kernels = 10;
  const size_t _top_optimizers = 5;
};

}  // namespace Analysis

#endif  // Analysis_Advisor_GPUAdvisor_hpp
