
//************************* System Include Files ****************************

#include <iostream>
#include <fstream>

#include <string>
#include <climits>
#include <cstring>
#include <cstdio>

#include <typeinfo>
#include <unordered_map>
#include <algorithm>
#include <stack>

#include <sys/stat.h>

//*************************** User Include Files ****************************

#include <include/uint.h>
#include <include/gcc-attr.h>
#include <include/gpu-metric-names.h>

#include "GPUOptimizer.hpp"
#include "Inspection.hpp"

using std::string;

#include <lib/prof/CCT-Tree.hpp>
#include <lib/prof/Struct-Tree.hpp>
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
#include <iostream>

namespace Analysis {

std::stack<Prof::Struct::Alien *> InspectionFormatter::getInlineStack(Prof::Struct::ACodeNode *stmt) {
  std::stack<Prof::Struct::Alien *> st;
  Prof::Struct::Alien *alien = stmt->ancestorAlien();

  while (alien) {
    st.push(alien);
    auto *stmt = alien->parent();
    if (stmt) {
      alien = stmt->ancestorAlien(); 
    } else {
      break;
    }
  };

  return st;
}


std::string SimpleInspectionFormatter::formatInlineStack(std::stack<Prof::Struct::Alien *> &inline_stack) {
  std::stringstream ss;

  ss << "Inline stack: " << std::endl;
  while (inline_stack.empty() == false) {
    auto *inline_struct = inline_stack.top();
    inline_stack.pop();
    ss << "Line " << inline_struct->begLine() << " in " << inline_struct->fileName() << std::endl;
  }

  return ss.str();
}


std::string SimpleInspectionFormatter::format(const Inspection &inspection) {
  std::stringstream ss;

  // Overview
  ss << "Apply " << inspection.optimization << " optimization,";
  
  if (inspection.ratio != -1.0) {
    ss << " ratio " << inspection.ratio * 100 << "%,";
  } 

  if (inspection.speedup != -1.0) {
    ss << " estimate speedup " << inspection.speedup << "x";
  }

  ss << std::endl;

  // Specific suggestion
  if (inspection.warp_count.first != -1) {
    ss << "Adjust #warps from " << inspection.warp_count.first << " to "
      << inspection.warp_count.second << std::endl;
  }

  if (inspection.block_count.first != -1) {
    ss << "Adjust #blocks from " << inspection.block_count.first << " to "
      << inspection.block_count.second << std::endl;
  }

  if (inspection.reg_count.first != -1) {
    ss << "Adjust #regs from " << inspection.reg_count.first << " to "
      << inspection.reg_count.second << std::endl;
  }

  // Hot regions
  for (auto &inst_blame : inspection.top_regions) {
    ss << "Hot " << inst_blame.blame_name << " code (" << inst_blame.stall_blame << "):" << std::endl;

    auto *src_struct = inst_blame.src_struct;
    auto *dst_struct = inst_blame.dst_struct;
    auto *src_func = src_struct->ancestorProc();
    auto *dst_func = dst_struct->ancestorProc();
    auto src_vma = (inst_blame.src)->pc - src_func->vmaSet().begin()->beg();
    auto dst_vma = (inst_blame.dst)->pc - dst_func->vmaSet().begin()->beg();

    // Print inline call stack
    std::stack<Prof::Struct::Alien *> src_inline_stack = getInlineStack(src_struct);
    std::stack<Prof::Struct::Alien *> dst_inline_stack = getInlineStack(dst_struct);

    ss << "----------------------------------------------------------" << std::endl;
    ss << "From" << std::endl;
    if (src_inline_stack.empty() == false) {
      ss << formatInlineStack(src_inline_stack);
      ss << std::hex << "0x" << src_vma << std::dec << " at Line " <<
        src_struct->begLine() << std::endl;
    } else {
      auto src_file = src_struct->ancestorFile();
      ss << std::hex << "0x" << src_vma << std::dec << " at Line " <<
        src_struct->begLine() << " in " <<  src_file->name() << std::endl;
    }

    ss << "To" << std::endl;
    if (dst_inline_stack.empty() == false) {
      ss << formatInlineStack(dst_inline_stack);
      ss << std::hex << "0x" << dst_vma << std::dec << " at Line " <<
        dst_struct->begLine() << std::endl;
    } else {
      auto dst_file = dst_struct->ancestorFile();
      ss << std::hex << "0x" << dst_vma << std::dec << " at Line " <<
        dst_struct->begLine() << " in " <<  dst_file->name() << std::endl;
    }
    ss << "----------------------------------------------------------" << std::endl;
  }

  ss << std::endl;

  return ss.str();
};

}  // namespace analysis
