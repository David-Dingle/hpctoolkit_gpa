
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

//************************* System Include Files ****************************

#include <iostream>
#include <fstream>

#include <string>
#include <climits>
#include <cstring>

#include <typeinfo>
#include <unordered_map>

#include <sys/stat.h>

//*************************** User Include Files ****************************

#include <include/uint.h>
#include <include/gcc-attr.h>
#include <include/gpu-metric-names.h>

#include "CallPath-TorchView.hpp"

using std::string;

#include <lib/prof/CCT-Tree.hpp>
#include <lib/prof/CallPath-Profile.hpp>
#include <lib/prof/Metric-Mgr.hpp>
#include <lib/prof/Metric-ADesc.hpp>

#include <lib/profxml/XercesUtil.hpp>
#include <lib/profxml/PGMReader.hpp>

#include <lib/prof-lean/hpcrun-metric.h>

#include <lib/binutils/LM.hpp>
#include <lib/binutils/VMAInterval.hpp>

#include <lib/xml/xml.hpp>

#include <lib/support/diagnostics.h>
#include <lib/support/Logic.hpp>
#include <lib/support/IOUtil.hpp>
#include <lib/support/StrUtil.hpp>


#include <vector>
#include <queue>
#include <iostream>

#include <memory>

namespace Analysis {

  namespace CallPath {

    typedef struct TV_CTX_NODE{
      int32_t ctx_id;
      std::string context;

      TV_CTX_NODE(){
        this->ctx_id = 0;
        this->context = "";
      }

      TV_CTX_NODE(int32_t cid) : ctx_id(cid), context("") {}

      TV_CTX_NODE(const TV_CTX_NODE& rhs){
        this->ctx_id = rhs.ctx_id;
        this->context = std::string(rhs.context);
      }

      void operator=(const TV_CTX_NODE& rhs){
        this->ctx_id = rhs.ctx_id;
        this->context = std::string(rhs.context);
      }
    } ctx_node_t;

    typedef struct PythonContex{
      std::string file_name;
      std::string function_name;
      int function_first_lineno;
      int lineno;

      PythonContex(){
        this->file_name = "";
        this->function_name = "";
        this->function_first_lineno = 0;
        this->lineno = 0;
      }

      PythonContex(const PythonContex& rhs){
        this->file_name = std::string(rhs.file_name);
        this->function_name = std::string(rhs.function_name);
        this->function_first_lineno = rhs.function_first_lineno;
        this->lineno = rhs.lineno;
      }

      void operator=(const PythonContex& rhs){
        this->file_name = std::string(rhs.file_name);
        this->function_name = std::string(rhs.function_name);
        this->function_first_lineno = rhs.function_first_lineno;
        this->lineno = rhs.lineno;
      }
    } python_context_t;

    typedef struct Torch_View_Call_Path{
      uint64_t global_id;
      ctx_node_t ctx_node;
      int num_states;
      std::size_t hash;
      uint16_t lm_id = 0;
      std::vector<uint64_t> function_offsets;
      std::vector<std::pair<uintptr_t, uintptr_t>> lm_ips;
      std::map<uint64_t, std::map<uintptr_t, double>> ip_weights;
      std::vector<python_context_t> python_contexts;

      Torch_View_Call_Path() = default;

      Torch_View_Call_Path(const Torch_View_Call_Path& rhs){
        this->global_id = rhs.global_id;
        this->ctx_node = rhs.ctx_node;
        this->num_states = rhs.num_states;
        this->hash = rhs.hash;
        this->lm_id = rhs.lm_id;
        this->function_offsets = std::vector<uint64_t>(rhs.function_offsets);
        this->lm_ips = std::vector<std::pair<uintptr_t, uintptr_t>>(rhs.lm_ips);
        this->python_contexts = std::vector<python_context_t>(rhs.python_contexts);
      }

      void operator=(const Torch_View_Call_Path& rhs){
        this->global_id = rhs.global_id;
        this->ctx_node = rhs.ctx_node;
        this->num_states = rhs.num_states;
        this->hash = rhs.hash;
        this->lm_id = rhs.lm_id;
        this->function_offsets = std::vector<uint64_t>(rhs.function_offsets);
        this->lm_ips = std::vector<std::pair<uintptr_t, uintptr_t>>(rhs.lm_ips);
        this->python_contexts = std::vector<python_context_t>(rhs.python_contexts);
      }

    } torch_view_call_path_t;

    typedef std::vector<torch_view_call_path_t> VIEW_CTX_MAP;



    static void read_memory_node(const std::string &file_name, VIEW_CTX_MAP &view_ctx_map,
                                blamed_pc_pairs_t* blames) {
      std::ifstream fileread(file_name);
      std::string word;
      bool is_id = false;
      bool is_num_states = false;
      bool is_file_name = false;
      bool is_function_name = false;
      bool is_function_first_lineno = false;
      bool is_lineno = false;
      bool is_pystates_hash = false;
      bool is_cct_node_persistent_id = false;
      bool is_lm_id = false;
      bool is_function_offsets = false;
      bool is_lm_ip = false;
      bool is_ctx_id = false;

      int32_t current_ctx_persistent_id = 0;

      while (fileread >> word) {

        if (word == "id"){
          is_id = true;
          is_num_states = false;
          is_file_name = false;
          is_function_name = false;
          is_function_first_lineno = false;
          is_lineno = false;
          is_pystates_hash = false;
          is_cct_node_persistent_id = false;
          is_lm_id = false;
          is_function_offsets = false;
          is_lm_ip = false;
          is_ctx_id = false;

          view_ctx_map.emplace_back();
          continue;
        }

        if (word == "num_states"){
          is_id = false;
          is_num_states = true;
          is_file_name = false;
          is_function_name = false;
          is_function_first_lineno = false;
          is_lineno = false;
          is_pystates_hash = false;
          is_cct_node_persistent_id = false;
          is_lm_id = false;
          is_function_offsets = false;
          is_lm_ip = false;
          is_ctx_id = false;

          continue;
        }

        if (word == "file_name"){
          is_id = false;
          is_num_states = false;
          is_file_name = true;
          is_function_name = false;
          is_function_first_lineno = false;
          is_lineno = false;
          is_pystates_hash = false;
          is_cct_node_persistent_id = false;
          is_lm_id = false;
          is_function_offsets = false;
          is_lm_ip = false;
          is_ctx_id = false;

          view_ctx_map.back().python_contexts.emplace_back();
          continue;
        }

        if (word == "function_name"){
          is_id = false;
          is_num_states = false;
          is_file_name = false;
          is_function_name = true;
          is_function_first_lineno = false;
          is_lineno = false;
          is_pystates_hash = false;
          is_cct_node_persistent_id = false;
          is_lm_id = false;
          is_function_offsets = false;
          is_lm_ip = false;
          is_ctx_id = false;

          continue;
        }

        if (word == "function_first_lineno"){
          is_id = false;
          is_num_states = false;
          is_file_name = false;
          is_function_name = false;
          is_function_first_lineno = true;
          is_lineno = false;
          is_pystates_hash = false;
          is_cct_node_persistent_id = false;
          is_lm_id = false;
          is_function_offsets = false;
          is_lm_ip = false;
          is_ctx_id = false;

          continue;
        }

        if (word == "lineno"){
          is_id = false;
          is_num_states = false;
          is_file_name = false;
          is_function_name = false;
          is_function_first_lineno = false;
          is_lineno = true;
          is_pystates_hash = false;
          is_cct_node_persistent_id = false;
          is_lm_id = false;
          is_function_offsets = false;
          is_lm_ip = false;
          is_ctx_id = false;

          continue;
        }

        if (word == "pystates_hash"){
          is_id = false;
          is_num_states = false;
          is_file_name = false;
          is_function_name = false;
          is_function_first_lineno = false;
          is_lineno = false;
          is_pystates_hash = true;
          is_cct_node_persistent_id = false;
          is_lm_id = false;
          is_function_offsets = false;
          is_lm_ip = false;
          is_ctx_id = false;

          continue;
        }

        if (word == "cct_node_persistent_id"){
          is_id = false;
          is_num_states = false;
          is_file_name = false;
          is_function_name = false;
          is_function_first_lineno = false;
          is_lineno = false;
          is_pystates_hash = false;
          is_cct_node_persistent_id = true;
          is_lm_id = false;
          is_function_offsets = false;
          is_lm_ip = false;
          is_ctx_id = false;

          continue;
        }

        if (word == "lm_id"){
          is_id = false;
          is_num_states = false;
          is_file_name = false;
          is_function_name = false;
          is_function_first_lineno = false;
          is_lineno = false;
          is_pystates_hash = false;
          is_cct_node_persistent_id = false;
          is_lm_id = true;
          is_function_offsets = false;
          is_lm_ip = false;
          is_ctx_id = false;

          continue;
        }

        if (word == "function_offset"){
          is_id = false;
          is_num_states = false;
          is_file_name = false;
          is_function_name = false;
          is_function_first_lineno = false;
          is_lineno = false;
          is_pystates_hash = false;
          is_cct_node_persistent_id = false;
          is_lm_id = false;
          is_function_offsets = true;
          is_lm_ip = false;
          is_ctx_id = false;

          continue;
        }

        if (word == "lm_ip"){
          is_id = false;
          is_num_states = false;
          is_file_name = false;
          is_function_name = false;
          is_function_first_lineno = false;
          is_lineno = false;
          is_pystates_hash = false;
          is_cct_node_persistent_id = false;
          is_lm_id = false;
          is_function_offsets = false;
          is_lm_ip = true;
          is_ctx_id = false;

          continue;
        }

        if (word == "ctx_id"){
          is_id = false;
          is_num_states = false;
          is_file_name = false;
          is_function_name = false;
          is_function_first_lineno = false;
          is_lineno = false;
          is_pystates_hash = false;
          is_cct_node_persistent_id = false;
          is_lm_id = false;
          is_function_offsets = false;
          is_lm_ip = false;
          is_ctx_id = true;

          continue;
        }

        if(is_id) {
          // std::size_t pos{};
          uint64_t id = (uint64_t)std::stoul(word);
          view_ctx_map.back().global_id = id;

          continue;
        }

        if(is_num_states){
          try{
            view_ctx_map.back().num_states = (int)std::stoi(word);
          }catch(...){
            // std::cerr << "num_states" << std::endl;
          }

          continue;
        }

        if(is_file_name) {
          view_ctx_map.back().python_contexts.back().file_name = word;

          continue;
        }

        if(is_function_name) {
          view_ctx_map.back().python_contexts.back().function_name = word;

          continue;
        }

        if(is_function_first_lineno) {
          try{
            view_ctx_map.back().python_contexts.back().function_first_lineno = (int)std::stoi(word);
          }catch(...){
            std::cerr << "first_lineno" << std::endl;
          }

          continue;
        }

        if(is_lineno) {
          try{
            view_ctx_map.back().python_contexts.back().lineno = (int)std::stoi(word);
          }catch(...){
            // std::cerr << "ERROR: lineon" << std::endl;
          }

          continue;
        }

        if(is_pystates_hash) {
          view_ctx_map.back().hash = (std::size_t)std::hash<std::string>{}(word);
          // std::cout << "I SAW: " << word << std::endl;
          continue;
        }

        if(is_cct_node_persistent_id) {
          current_ctx_persistent_id = (int32_t)std::stol(word);

          continue;
        }

        if(is_lm_id) {
          for(VIEW_CTX_MAP::iterator it = view_ctx_map.end() - 1; it >= view_ctx_map.begin(); it--) {
            if(it->ctx_node.ctx_id == current_ctx_persistent_id) {
              it->lm_id = (uint16_t)std::stol(word);

              goto flag_outer;  // Be Careful !
            }
          }
        }

        if(is_function_offsets) {
          for(VIEW_CTX_MAP::iterator it = view_ctx_map.end() - 1; it >= view_ctx_map.begin(); it--) {
            if(it->ctx_node.ctx_id == current_ctx_persistent_id) {
              uintptr_t function_offset = (uintptr_t)std::stol(word);
              it->function_offsets.emplace_back(function_offset);

              goto flag_outer;
            }
          }
        }

        if(is_lm_ip) {
          uintptr_t sample_pc = (uintptr_t)std::stol(word);
          for(VIEW_CTX_MAP::iterator it = view_ctx_map.end() - 1; it >= view_ctx_map.begin(); it--) {
            if(it->ctx_node.ctx_id == current_ctx_persistent_id) {
              // uintptr_t sample_pc = (uintptr_t)std::stol(word);
              bool flag = false;
              for(auto& [func_addr, pc_vector] : (*blames)[it->lm_id]) {
                if(func_addr == it->function_offsets.back()){
                  for(pc_pair_t& piter : pc_vector){
                    if(((uintptr_t)piter.second == sample_pc)) {
                      if(flag) {
                        it->function_offsets.emplace_back(it->function_offsets.back());
                      }
                      it->lm_ips.emplace_back(std::pair<uintptr_t, uintptr_t>{(uintptr_t)piter.first, (uintptr_t)piter.second});
                      flag = true;
                    }
                  }
                }
              }
              if(!flag) {
                it->lm_ips.emplace_back(std::pair<uintptr_t, uintptr_t>{sample_pc /*0*/, sample_pc});
              }
              goto flag_outer;
            }
          }
        }
        if(is_ctx_id) {
          view_ctx_map.back().ctx_node.ctx_id = (int32_t)std::stol(word);

          continue;
        }
        flag_outer: ;

      }
      fileread.close();
    }


#define MAX_STR_LEN 128

    static std::string
    trunc(const std::string &raw_str) {
      std::string str = raw_str;
      if (str.size() > MAX_STR_LEN) {
        str.erase(str.begin() + MAX_STR_LEN, str.end());
      }
      return str;
    }

    static std::vector<std::string>
    getInlineStack(Prof::Struct::ACodeNode *stmt) {
      std::vector<std::string> st;
      Prof::Struct::Alien *alien = stmt->ancestorAlien();
      if (alien) {
        auto func_name = trunc(alien->name());
        auto *stmt = alien->parent();
        if (stmt) {
          if (alien->name() == "<inline>") {
            // Inline macro
          } else if (stmt->type() == Prof::Struct::ANode::TyAlien) {
            // inline function
            alien = dynamic_cast<Prof::Struct::Alien *>(stmt);
          } else {
            return st;
          }
          auto file_name = alien->fileName();
          auto line = std::to_string(alien->begLine());
          auto name = file_name + ":" + line + "\t" + func_name;
          st.push_back(name);

          while (true) {
            stmt = alien->parent();
            if (stmt) {
              alien = stmt->ancestorAlien();
              if (alien) {
                func_name = trunc(alien->name());
                stmt = alien->parent();
                if (stmt) {
                  if (alien->name() == "<inline>") {
                    // Inline macro
                  } else if (stmt->type() == Prof::Struct::ANode::TyAlien) {
                    // inline function
                    alien = dynamic_cast<Prof::Struct::Alien *>(stmt);
                  } else {
                    break;
                  }
                  file_name = alien->fileName();
                  line = std::to_string(alien->begLine());
                  name = file_name + ":" + line + "\t" + func_name;
                  st.push_back(name);
                } else {
                  break;
                }
              } else {
                break;
              }
            } else {
              break;
            }
          }
        }
      }

      std::reverse(st.begin(), st.end());
      return st;
    }

#define MAX_FRAMES 20

    static void matchCCTNode(Prof::CallPath::CCTIdToCCTNodeMap &cctNodeMap, VIEW_CTX_MAP &ctx_node_map) { 
      // match nodes
      for (auto &iter : ctx_node_map) {
        auto &node = iter.ctx_node;
        Prof::CCT::ANode *cct = NULL;

        if (cctNodeMap.find(node.ctx_id) != cctNodeMap.end()) {
          cct = cctNodeMap.at(node.ctx_id);
        } else {
          auto node_id = (uint32_t)(-node.ctx_id);
          if (cctNodeMap.find(node_id) != cctNodeMap.end()) {
            cct = cctNodeMap.at(node_id);
          }
        }

        if (cct) {
          std::stack<Prof::CCT::ProcFrm *> st;
          Prof::CCT::ProcFrm *proc_frm = NULL;
          std::string cct_context;

          if (cct->type() != Prof::CCT::ANode::TyProcFrm &&
            cct->type() != Prof::CCT::ANode::TyRoot) {
            proc_frm = cct->ancestorProcFrm(); 

            if (proc_frm != NULL) {
              auto *strct = cct->structure();
              if (strct->ancestorAlien()) {
                auto alien_st = getInlineStack(strct);
                for (auto &name : alien_st) {
                  // Get inline call stack
                  cct_context.append(name);
                  cct_context.append("#\n");
                }
              }
              auto *file_struct = strct->ancestorFile();
              auto file_name = file_struct->name();
              auto line = std::to_string(strct->begLine());
              auto name = file_name + ":" + line + "\t <op>";
              cct_context.append(name);
              cct_context.append("#\n");
            }
          } else {
            proc_frm = dynamic_cast<Prof::CCT::ProcFrm *>(cct);
          }

          while (proc_frm) {
            if (st.size() > MAX_FRAMES) {
              break;
            }
            st.push(proc_frm);
            auto *stmt = proc_frm->parent();
            if (stmt) {
              proc_frm = stmt->ancestorProcFrm();
            } else {
              break;
            }
          };

          while (!st.empty()) {
            proc_frm = st.top();
            st.pop();
            if (proc_frm->structure()) {
              if (proc_frm->ancestorCall()) {
                auto func_name = trunc(proc_frm->structure()->name());
                auto *call = proc_frm->ancestorCall();
                auto *call_strct = call->structure();
                auto line = std::to_string(call_strct->begLine());
                std::string file_name = "Unknown";
                if (call_strct->ancestorAlien()) {
                  auto alien_st = getInlineStack(call_strct);
                  for (auto &name : alien_st) {
                    // Get inline call stack
                    node.context.append(name);
                    node.context.append("#\n");
                  }

                  auto fname = call_strct->ancestorAlien()->fileName();
                  if (fname.find("<unknown file>") == std::string::npos) {
                    file_name = fname;
                  }
                  auto name = file_name + ":" + line + "\t" + func_name;
                  node.context.append(name);
                  node.context.append("#\n");
                } else if (call_strct->ancestorFile()) {
                  auto fname = call_strct->ancestorFile()->name();
                  if (fname.find("<unknown file>") == std::string::npos) {
                    file_name = fname;
                  }
                  auto name = file_name + ":" + line + "\t" + func_name;
                  node.context.append(name);
                  node.context.append("#\n");
                }
              }
            }
          }

          if (cct_context.size() != 0) {
            std::cout << cct_context << std::endl; // DEBUG
            node.context.append(cct_context);
          }
        }
      }
    }


    static void outputContext(const std::string &file_name, VIEW_CTX_MAP &ctx_node_map,
                              blamed_pc_pairs_t* blames) {

      // start find hot blames
      std::map<Prof::LoadMap::LMId_t, std::vector<std::pair<VMA, uint32_t>>> hot = std::map<Prof::LoadMap::LMId_t, std::vector<std::pair<VMA, uint32_t>>>{};
      for(auto & iter : ctx_node_map) {
        if(hot.find(iter.lm_id) != hot.end()) {
          hot[iter.lm_id] = std::vector<std::pair<VMA, uint32_t>> {};
        }
      }  // populate all the lm(s) we collected
      for(auto& [lm_id, func_addrs]: (*blames) ){
        for(auto& fiter : func_addrs) {
          for(auto& piter : fiter.second) {
            bool find = false;
            for(auto& hiter : hot[lm_id]) {
              if(hiter.first == piter.first) {
                hiter.second += 1;
                find = true;
                // break;
              }
            }
            if(!find) {
              hot[lm_id].emplace_back(piter.first, 1);
            }
          }
        }
      }  // count all the blamed pc(s)
      for(auto& [lm_id, b_c] : hot){
        std::sort(b_c.begin(), b_c.end(),
                  [](const std::pair<VMA, uint32_t>& a, const std::pair<VMA, uint32_t>& b){
                    return a.second > b.second;
                  } );
      }  // sort blamed instruction by its counter value

      // end find hot blames

      // start calculating blamed PC weight
      uint64_t num_blames = 0;
      for(auto& niter : ctx_node_map) {
        num_blames += niter.lm_ips.size();
      }
      std::cout << "num_blames: " << num_blames << std::endl;
      for(auto& niter : ctx_node_map) {
        std::cout << "Size of Func: " << niter.function_offsets.size() << std::endl;
        std::cout << "Size of blamed pc: " << niter.lm_ips.size() << std::endl;
        for(size_t i = 0; i < niter.function_offsets.size(); i++){
          if(niter.lm_ips.at(i).first == 0){
            continue;
          }
          std::cout << "PRINT Blamed PC: " << std::hex << niter.lm_ips.at(i).first << std::dec << std::endl;
          if(niter.ip_weights[niter.function_offsets.at(i)].find(niter.lm_ips.at(i).first) == niter.ip_weights[niter.function_offsets.at(i)].end()) {
            niter.ip_weights[niter.function_offsets.at(i)][niter.lm_ips.at(i).first] = 1.0;
            std::cout << std::hex << niter.lm_ips.at(i).first << std::dec << " inserted" << std::endl;
          } else{
            niter.ip_weights[niter.function_offsets.at(i)][niter.lm_ips.at(i).first] += 1.0;
            std::cout << std::hex << niter.lm_ips.at(i).first << std::dec << " added." << std::endl;
          }
        }
        for(auto& [f, wmap] : niter.ip_weights){
          for(auto& [b, w] : wmap){
            std::cout << "pc: " << std::hex << b << std::dec << " count: " << w << std::endl;
            w /= num_blames;
            w *= 100;
            std::cout << "pc: " << std::hex << b << std::dec << " weight: " << w << std::endl;
          }
        }
      }
      // end calculating blamed PC weight

      std::ofstream out(file_name + ".context");
      // start print hot blames
      // for(auto& [lm, b_c] : hot){
      //   out << "Load Module: " << lm << std::endl ;
      //   for(auto& [b, c] : b_c){
      //     out << std::hex << b << std::dec << ":" << std::hex << c << std::dec << " ";
      //   }
      //   out << std::endl;
      // }

      for(auto& niter : ctx_node_map){
        if(!niter.lm_id){
          continue;
        }
        out << "ctx_id: " << niter.ctx_node.ctx_id << " LM: " << niter.lm_id << std::endl;
        for(auto& [f, wmap] : niter.ip_weights){
          out << "  Func_Addr: " << f << std::endl;
          for(auto& [b, w] : wmap){
            out << "    " << std::hex << b << std::dec << ": " << w << "%  ";
          } out << std::endl;
        } out << std::endl;
      }

      out << "\n------------------------------------------------------------\n" << std::endl;
      // end print hot blames

      for (auto& iter : ctx_node_map) {
        if(iter.lm_id == 0) {
          continue;
        }
        out << iter.global_id << ": " << std::endl;
        for (auto& piter : iter.python_contexts) {
          // out << piter.file_name << std::endl;
          // out << piter.function_name << std::endl;
          // out << piter.function_first_lineno << std::endl;
          // out << piter.lineno << std::endl;
          out << "  " << piter.file_name << ":" << piter.function_name << ":" << piter.function_first_lineno << ":" << piter.lineno << std::endl;
        }
        out << "pystates_hash: " << iter.hash << std::endl;
        out << "ctx_id: " << iter.ctx_node.ctx_id << std::endl;
        if(iter.lm_id != 0) {  // if PC info is available
          out << "leaf lm_id: " << iter.lm_id << std::endl;
        }
        if(!iter.lm_ips.empty()) {
          out << "lm_ip:"; 
          std::sort(iter.lm_ips.begin(),
                    iter.lm_ips.end(),
                    [] (const std::pair<uintptr_t, uintptr_t> a, const std::pair<uintptr_t, uintptr_t> b)
                       { return a.first < b.first; });
          for(auto v : iter.lm_ips) {
            if(v.first != 0) {
              out << " " << std::hex << v.first << std::dec << "/" << std::hex << v.second << std::dec;
            } else{
              out << " " << std::hex << v.first << std::dec << "/" << std::hex << v.second << std::dec;
            }
          }
          out << std::endl;
        }
        out << iter.ctx_node.context << std::endl;;
      }

      out.close();
    }

    static void finish(VIEW_CTX_MAP& ctx_node_map) {
      for(auto& iter : ctx_node_map) {
        iter.python_contexts.clear();
      }
    }

// VIEW_CTX_MAP view_ctx_map;

    void analyzeTorchViewMain(Prof::CallPath::Profile &prof,
                              const std::vector<std::string> &torchViewFiles,
                              blamed_pc_pairs_t* blames) {
      Prof::CallPath::CCTIdToCCTNodeMap cctNodeMap;

      Prof::CCT::ANodeIterator prof_it(prof.cct()->root(), NULL/*filter*/, false/*leavesOnly*/,
                                       IteratorStack::PreOrder); // PreOrder
      for (Prof::CCT::ANode *n = NULL; (n = prof_it.current()); ++prof_it) {
        Prof::CCT::ADynNode* n_dyn = dynamic_cast<Prof::CCT::ADynNode*>(n);
        if (n_dyn) {
          cctNodeMap.insert(std::make_pair(n_dyn->cpId(), n));
        }
      }

      for (auto &file : torchViewFiles) {

        VIEW_CTX_MAP view_ctx_map;

        read_memory_node(file, view_ctx_map, blames);

        matchCCTNode(cctNodeMap, view_ctx_map);

        outputContext(file, view_ctx_map, blames);

        finish(view_ctx_map);

      }
    }
  } // namespace CallPath
} // namespace Analysis
