
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

#ifndef Analysis_CallPath_CallPath_TorchView_hpp
#define Analysis_CallPath_CallPath_TorchView_hpp

//************************* System Include Files ****************************

#include <iostream>
#include <vector>
#include <stack>
#include <string>
//#include <boost/archive/binary_oarchive.hpp>
//#include <boost/archive/binary_iarchive.hpp>
//#include <boost/serialization/map.hpp>
//#include <boost/serialization/string.hpp>
//#include <boost/serialization/list.hpp>
//*************************** User Include Files ****************************

#include <include/uint.h>

#include <lib/analysis/advisor/GPUInstruction.hpp>
#include <lib/binutils/LM.hpp>
#include <lib/binutils/VMAInterval.hpp>
#include <lib/prof/CallPath-Profile.hpp>
#include <lib/prof/Struct-Tree.hpp>

// typedef std::pair<VMA, VMA> pc_pair_t;
// typedef std::map<uint64_t, std::vector<pc_pair_t>> function_pcs_t;
// typedef std::map<uint64_t, function_pcs_t> blamed_pc_pairs_t;

namespace Analysis {

  namespace CallPath {

    void analyzeTorchViewMain(Prof::CallPath::Profile &prof,
                              const std::vector<std::string> &torchViewFiles,
                              blamed_pc_pairs_t* blames);

  } // namespace CallPath

} // namespace Analysis

#endif
