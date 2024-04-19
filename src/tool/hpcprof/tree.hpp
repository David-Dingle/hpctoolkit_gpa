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
// Copyright ((c)) 2002-2024, Rice University
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

#ifndef HPCPROF_MPI_TREE_HPP
#define HPCPROF_MPI_TREE_HPP

#include "sinks/packed.hpp"
#include "sources/packed.hpp"
#include "pipeline.hpp"

#include <vector>

/// Representative structure for an n-ary rank-based reduction tree
class RankTree final {
public:
  RankTree(std::size_t arity);
  ~RankTree() = default;

  const std::size_t arity;
  const std::size_t parent;
  const std::size_t min;
  const std::size_t max;
};

/// Sink for sending the initial CCT up the tree. Can be constructed with an
/// additional vector argument to be passed to MetricReciever.
class Sender : public hpctoolkit::sinks::Packed {
public:
  Sender(RankTree&);

  hpctoolkit::ExtensionClass requirements() const noexcept override { return {}; }
  hpctoolkit::DataClass accepts() const noexcept override {
    using namespace hpctoolkit::literals;
    return data::attributes + data::references + data::contexts;
  }
  void notifyPipeline() noexcept override;
  void write() override;

private:
  RankTree& tree;
};

/// Source for receiving the data from Sender
class Receiver : public hpctoolkit::sources::Packed {
public:
  Receiver(std::size_t, std::vector<std::uint8_t>&);
  Receiver(std::vector<std::uint8_t>&);
  ~Receiver() = default;

  hpctoolkit::DataClass provides() const noexcept override {
    using namespace hpctoolkit::literals;
    return data::attributes + data::references + data::contexts;
  }
  hpctoolkit::DataClass finalizeRequest(const hpctoolkit::DataClass& d) const noexcept override {
    return d;
  }
  void read(const hpctoolkit::DataClass&) override;

  static void append(hpctoolkit::ProfilePipeline::Settings&, RankTree&,
      std::deque<std::vector<std::uint8_t>>&);

private:
  std::size_t peer;
  bool readBlock = false;
  std::vector<std::uint8_t>& block;
  bool parsedBlock = false;
};

/// Sink for sending Statistic data up the tree.
class MetricSender : public hpctoolkit::sinks::ParallelPacked {
public:
  MetricSender(RankTree&);
  ~MetricSender() = default;

  hpctoolkit::DataClass accepts() const noexcept override {
    using namespace hpctoolkit::literals;
    return data::attributes + data::contexts + data::metrics + data::ctxTimepoints;
  }
  void notifyPipeline() noexcept override;
  void write() override;
  hpctoolkit::util::WorkshareResult help() override;

private:
  RankTree& tree;
  std::once_flag waveOnce;
};

/// Source for receiving the data from MetricSender. Can also unpack the data
/// generated by Sender.
class MetricReceiver : public hpctoolkit::sources::Packed {
public:
  MetricReceiver(std::size_t, hpctoolkit::sources::Packed::IdTracker&);
  ~MetricReceiver() = default;

  hpctoolkit::DataClass provides() const noexcept override {
    using namespace hpctoolkit::literals;
    return data::metrics + data::ctxTimepoints;
  }
  hpctoolkit::DataClass finalizeRequest(const hpctoolkit::DataClass& d) const noexcept override {
    return d;
  }
  void read(const hpctoolkit::DataClass& d) override;

  static void append(hpctoolkit::ProfilePipeline::Settings&, RankTree&,
      hpctoolkit::sources::Packed::IdTracker&);

private:
  std::size_t peer;
  bool readBlock = false;
  std::vector<uint8_t> block;
  bool parsedBlock = false;
};

#endif  // HPCPROF_MPI_TREE_HPP
