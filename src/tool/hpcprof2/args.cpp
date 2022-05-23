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
// Copyright ((c)) 2019-2022, Rice University
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

#include "lib/profile/util/vgannotations.hpp"

#include "args.hpp"

#include "lib/profile/source.hpp"
#include "lib/profile/finalizers/kernelsyms.hpp"
#include "lib/profile/finalizers/struct.hpp"
#include "include/hpctoolkit-config.h"
#include "lib/profile/mpi/all.hpp"

#include "lib/prof-lean/cpuset_hwthreads.h"

#include <cassert>
#include <getopt.h>
#include <iostream>
#include <iomanip>
#include <omp.h>
#include <random>

using namespace hpctoolkit;
namespace fs = stdshim::filesystem;

static const std::string version = HPCTOOLKIT_VERSION;
static const std::string summary =
"[options]... <measurement files/directories>...";
static const std::string header = R"EOF(
Attribute measurements made by `hpcrun' back to the original source using maps
generated by `hpcstruct', packaging the result as a monolithic database
viewable in `hpc[trace]viewer'.
)EOF";
static const std::string footer = R"EOF(
For best results, compile your application with debug information, generate
structure data (`-S'), and provide prefix replacements (`-R') as needed.
)EOF";

static const std::string options = R"EOF(
General Options:
  -h, --help                  Display this help and exit.
      --version               Print version information and exit.
  -v, --verbose               Enable additional information output.
  -q, --quiet
                              Disable non-error messages. Overrides --verbose.
                              If repeated will disable all output.
  -o FILE                     Output to the given filename.
      --force                 Overwrite the output if it exists already.
  -O FILE                     Shorthand for `--force -o FILE'.
  -Q, --dry-run               Disable output. Useful for performance testing.
  -jN                         Use N threads to accelerate processing.

Input Options:
  -S, --structure=FILE        Read binary structure information from FILE.
  -R, --replace-prefix=FROM=TO
                              Replace path prefixes when searching for source
                              files and binaries. Use `\=' to escape `=', use
                              `\\' to escape `\'.

Output Options:
  -n, --title=NAME            Specify a title for the output database.
  -f, --format=FORMAT
                              Specify the database output format.
                              Default is `exmldb` for the usual format,
                              using `sparse` outputs in the new sparse format.
  -M (none|STAT[,STAT...])
                              Disable or enable generation of global
                              statistics. STAT is one of the following:
                                    sum: Linear sum (over threads)
                                 normal: Linear mean and standard deviation
                                extrema: Minimum and maximum
                                  stats: All of the above
                              `none' disables all global statistics.
      --no-thread-local       Disable generation of thread-local statistics.
      --no-traces             Disable generation of traces.
      --no-source             Disable embedded source output.

Processing options:
      --dwarf-max-size=<limit>[<unit>]
                              Specify a limit on the binary size to parse DWARF
                              data from. Units are K,M,G,T (powers of 1024)
                              If limit is "unlimited," always parses DWARF.
                              Default limit is 100M.

Compatibility Options:
      --name=NAME             Equivalent to `-n NAME'
      --metric-db (yes|no)    `no' is equivalent to --no-thread-local.

Current Obsolete Options:
  -I, --include=DIR           Unsupported, use `-R' instead.
      --debug                 Depreciated, use `-v' or `-q' instead.
      --force-metric          Unsupported.
      --remove-redundancy     Unsupported (effect is always enabled).
      --struct-id             Unsupported.
)EOF";

const bool string_starts_with(const std::string& a, const std::string& n) {
  auto it_n = n.begin();
  for(auto it = a.begin(); it != a.end() && it_n != n.end(); ++it, ++it_n) {
    if(*it != *it_n) return false;
  }
  return it_n == n.end();
}
const bool string_ends_with(const std::string& a, const std::string& n) {
  auto it_n = n.rbegin();
  for(auto it = a.rbegin(); it != a.rend() && it_n != n.rend(); ++it, ++it_n) {
    if(*it != *it_n) return false;
  }
  return it_n == n.rend();
}

ProfArgs::ProfArgs(int argc, char* const argv[])
  : title(), threads(0), output(),
    include_sources(true), include_traces(true), include_thread_local(true),
    format(Format::sparse), dwarfMaxSize(100*1024*1024), valgrindUnclean(false) {
  int arg_includeSources = include_sources;
  int arg_includeTraces = include_traces;
  int arg_overwriteOutput = 0;
  int arg_valgrindUnclean = valgrindUnclean;
  struct option longopts[] = {
    // These first ones are more special and must be in this order.
    {"version", no_argument, NULL, 0},
    {"metric-db", required_argument, NULL, 0},
    {"no-thread-local", no_argument, NULL, 0},
    {"dwarf-max-size", required_argument, NULL, 0},
    // The rest can be in any order
    {"help", no_argument, NULL, 'h'},
    {"verbose", no_argument, NULL, 'v'},
    {"quiet", no_argument, NULL, 'q'},
    {"dry-run", no_argument, NULL, 'Q'},
    {"replace-prefix", required_argument, NULL, 'R'},
    {"title", required_argument, NULL, 'n'},
    {"format", required_argument, NULL, 'f'},
    {"no-traces", no_argument, &arg_includeTraces, 0},
    {"no-source", no_argument, &arg_includeSources, 0},
    {"name", required_argument, NULL, 'n'},
    {"force", no_argument, &arg_overwriteOutput, 1},
    {"valgrind-unclean", no_argument, &arg_valgrindUnclean, 1},
    {0, 0, 0, 0}
  };

  bool seenNoThreadLocal = false;
  bool seenMetricDB = false;
  bool dryRun = false;

  int quiet = 0;
  util::log::Settings logSettings(true, true, false);

  int opt;
  int longopt;
  while((opt = getopt_long(argc, argv, "hvqQO:o:j:S:R:n:f:M:", longopts, &longopt)) >= 0) {
    switch(opt) {
    case 'h':
      std::cout << "Usage: " << fs::path(argv[0]).filename().string()
                             << " " << summary  // header begins with a '\n'
                << header << options << footer;
      std::exit(0);
    case 'v':
      logSettings.info() = true;
      break;
    case 'q':
      quiet++;
      break;
    case 'O':
      arg_overwriteOutput = 1;
      // fallthrough
    case 'o':
      output = fs::path(optarg);
      if(!output.has_filename()) output = output.parent_path();
      break;
    case 'Q':
      dryRun = true;
      break;
    case 'j': {
      std::size_t pos;
      int in_threads = std::stoi(optarg, &pos, 10);
      if(pos == 0 || optarg[pos] != '\0' || in_threads < 0) {
        std::cerr << "Invalid thread number '" << optarg << "'!\n";
        std::exit(2);
      }
      threads = in_threads;
      break;
    }
    case 'S': {
      fs::path path(optarg);
      std::unique_ptr<finalizers::StructFile> c;
      try {
        c.reset(new finalizers::StructFile(path));
      } catch(...) {
        std::cerr << "Invalid structure file '" << optarg << "'!\n";
        std::exit(2);
      }
      for(const auto& p : c->forPaths())
        structheads[p.filename()].emplace_back(p.parent_path());
      structs.emplace_back(std::move(c), path);
      break;
    }
    case 'R': {
      std::string from;
      std::string to;
      std::string* current = &from;
      for(std::size_t idx = 0; optarg[idx] != '\0'; idx++) {
        if(optarg[idx] == '=' && current == &from)
          current = &to;
        else if(optarg[idx] == '\\' && optarg[idx+1] == '=') {
          *current += '=';
          idx++;  // Skip over escaped equals
        } else if(optarg[idx] == '\\' && optarg[idx+1] == '\\') {
          *current += '\\';
          idx++;  // Skip over escaped backslash
        } else
          *current += optarg[idx];
      }
      if(from.empty()) {
        std::cerr << "Missing source prefix in -R '" << optarg << "'!\n";
        std::exit(2);
      }
      if(to.empty()) {
        std::cerr << "Missing destination prefix in -R '" << optarg << "'!\n";
        std::exit(2);
      }
      if(!prefixes.emplace(fs::path(from), fs::path(to)).second) {
        std::cerr << "Duplicate replacement for prefix '" << from << "'!\n";
        std::exit(2);
      }
      break;
    }
    case 'n':
      title = optarg;
      break;
    case 'f': {
      std::string form(optarg);
      if(form == "sparse") format = Format::sparse;
      else {
        std::cerr << "Unrecognized output format '" << form << "'!\n";
        std::exit(2);
      }
      break;
    }
    case 'M': {
      const char* const options[] = {"none", "sum", "normal", "extrema",
                                     "stats", nullptr};
      char* value;
      while(optarg[0] != '\0') {
        switch(getsubopt(&optarg, const_cast<char*const*>(options), &value)) {
        case 0:  // none
          stats.sum = stats.mean = stats.min = stats.max = stats.stddev
                    = stats.cfvar = false;
          break;
        case 1:  // sum
          stats.sum = true; break;
        case 2:  // normal
          stats.mean = stats.stddev = stats.cfvar = true; break;
        case 3:  // extrema
          stats.min = stats.max = true; break;
        case 4:  // stats
          stats.sum = stats.mean = stats.min = stats.max = stats.stddev
                    = stats.cfvar = true;
          break;
        default:
          std::cout << "Unrecognized argument to -M: " << value << "\n"
                       "Usage: " << fs::path(argv[0]).filename().string()
                                 << " " << summary << "\n";
          std::exit(2);
        }
      }
      break;
    }
    case 0:
      switch(longopt) {
      case 0:  // --version
        std::cout << fs::path(argv[0]).filename().string()
                  << " of HPCToolkit " << version << "\n";
        std::exit(0);
      case 1: {  // --metric-db (mutually exclusive with --no-thread-local)
        if(seenNoThreadLocal) {
          std::cerr << "Error: --metric-db and --no-thread-local cannot be used together!\n";
          std::exit(2);
        }
        std::string arg(optarg);
        if(arg == "yes") include_thread_local = true;
        else if(arg == "no") include_thread_local = false;
        else {
          std::cerr << "Error: --metric-db argument must be `yes' or `no'!\n";
          std::exit(2);
        }
        seenMetricDB = true;
        break;
      }
      case 2:  // --no-thread-local (mutually exclusive with --metric-db)
        if(seenMetricDB) {
          std::cerr << "Error: --metric-db and --no-thread-local cannot be used together!\n";
          std::exit(2);
        }
        include_thread_local = false;
        seenNoThreadLocal = true;
        break;
      case 3: {  // --dwarf-max-size
        char* end;
        double limit = std::strtod(optarg, &end);
        if(end == optarg) {  // Failed conversion
          std::string s(optarg);
          size_t start;
          for(start = 0; start < s.size() && std::isspace(s[start]); start++);
          s = std::move(s).substr(start);

          if(s == "unlimited") dwarfMaxSize = std::numeric_limits<uintmax_t>::max();
          else {
            std::cerr << "Error: invalid limit for --dwarf-max-size: `"
                      << s << "'\n";
            std::exit(2);
          }
        } else {
          uintmax_t factor = 1024;
          if(end[0] != '\0') {
            switch(end[0]) {
            case 'k': case 'K': factor = 1024; break;
            case 'm': case 'M': factor = 1024 * 1024; break;
            case 'g': case 'G': factor = 1024 * 1024 * 1024; break;
            case 't': case 'T': factor = 1024UL * 1024 * 1024 * 1024; break;
            }
            if(end[1] != '\0') {
              std::cerr << "Error: invalid suffix for --dwarf-max-size: `"
                        << optarg << "'\n";
              std::exit(2);
            }
          }
          dwarfMaxSize = std::floor(limit * factor);
        }
      }
      }
      break;
    default:
      std::cout << "Usage: " << fs::path(argv[0]).filename().string()
                             << " " << summary << "\n";
      std::exit(2);
    }
  }

  include_sources = arg_includeSources;
  include_traces = arg_includeTraces;
  valgrindUnclean = arg_valgrindUnclean;

  if(threads == 0) {
    threads = cpuset_hwthreads();
  }

  if(quiet > 0) {
    logSettings = util::log::Settings::none;
    logSettings.error() = quiet < 2;
  }
  util::log::Settings::set(std::move(logSettings));
  util::log::info{} << "Maximum verbosity enabled";

  if(dryRun) {
    output = fs::path();
    util::log::info{} << "Dry run enabled, final output will be skipped.";
  } else {
    if(mpi::World::rank() == 0) {
      if(output.empty()) {
        // Default to something semi-reasonable.
        output = "hpctoolkit-database";
        if(argc - optind == 1) {  // == only one input file argument
          stdshim::filesystem::path input = argv[optind];
          if(input.filename().empty()) input = input.parent_path();

          auto fn = input.filename().string();
          if(string_starts_with(fn, "hpctoolkit-")) {
            if(string_ends_with(fn, "-measurements"))
              fn = fn.substr(0, fn.size() - 13);
            output = input.parent_path() / (fn + "-database");
          }
        }
        util::log::warning{} << "Output database argument not given, defaulting"
          " to `" << output.string() << "'";
      }
      if(stdshim::filesystem::exists(output)) {
        if(arg_overwriteOutput == 0) {
          // The output must not exist beforehand, otherwise we will munge the
          // path until it doesn't exist anymore.
          // There's a potential for races here, which we don't attempt to fix;
          // the user should be explicit about their outputs.
          auto fbase = output.filename().string();
          output = output.parent_path();

          std::minstd_rand gen(std::random_device{}());
          std::uniform_int_distribution<uint32_t> rand;
          std::ostringstream ss;
          do {
            ss.str("");
            ss << fbase << "-" << std::hex << std::setfill('0') << std::setw(8)
               << rand(gen);
          } while(stdshim::filesystem::exists(output / ss.str()));
          util::log::warning{} << "Output database `" << (output/fbase).string()
            << "' exists, outputting to `" << (output/ss.str()).string() << "'";
          output /= ss.str();
        } else {
          // The output should be overwritten, so we first remove it.
          stdshim::filesystem::remove_all(output);
        }
      }
    }
    output = mpi::bcast(output.string(), 0);
  }

  // Gather up all the potential inputs, and distribute them across the ranks
  std::vector<std::pair<stdshim::filesystem::path, std::size_t>> files;
  {
    std::vector<std::string> files_s;
    if(mpi::World::rank() == 0) {
      std::vector<std::vector<std::string>> allfiles(mpi::World::size());
      std::size_t peer = 0;
      for(int idx = optind; idx < argc; idx++) {
        fs::path p(argv[idx]);
        if(fs::is_directory(p)) {
          for(const auto& de: fs::directory_iterator(p)) {
            allfiles[peer].emplace_back(de.path().string());
            peer = (peer + 1) % allfiles.size();
          }
          // Also check for a kernel_symbols/ directory for ksymsfiles.
          fs::path sp = p / "kernel_symbols";
          if(fs::is_directory(sp))
            ProfArgs::ksyms.emplace_back(std::make_unique<finalizers::KernelSymbols>(std::move(sp)));
          // Also check for a structs/ directory for extra structfiles.
          sp = p / "structs";
          if(fs::exists(sp)) {
            for(const auto& de: fs::directory_iterator(sp)) {
              std::unique_ptr<ProfileFinalizer> c;
              if(de.path().extension() != ".hpcstruct") continue;
              try {
                c.reset(new finalizers::StructFile(de));
              } catch(...) { continue; }
              ProfArgs::structs.emplace_back(std::move(c), de);
            }
          }
        } else {
          allfiles[peer].emplace_back(p.string());
          peer = (peer + 1) % allfiles.size();
        }
        // We use an empty string to mark the boundaries between argument "groups"
        for(auto& fs: allfiles) fs.emplace_back("");
      }
      files_s = mpi::scatter(std::move(allfiles), 0);
    } else {
      files_s = mpi::scatter<std::vector<std::string>>(0);
    }
    std::size_t arg = 0;
    files.reserve(files_s.size());
    for(auto& p: files_s) {
      if(p.empty()) arg++;
      else files.emplace_back(std::move(p), arg);
    }
  }

  // Every rank tests its allocated set of inputs, and the total number of
  // successes per group is summed.
  std::vector<std::uint32_t> cnts(argc - optind, 0);
  {
    std::vector<std::atomic<std::uint32_t>> cnts_a(cnts.size());
    for(auto& a: cnts_a) a.store(0, std::memory_order_relaxed);

  #ifdef ENABLE_VG_ANNOTATIONS
    char start_arc;
    char end_arc;
  #endif

    const fs::path profileext = std::string(".")+HPCRUN_ProfileFnmSfx;

    ANNOTATE_HAPPENS_BEFORE(&start_arc);
    #pragma omp parallel num_threads(threads)
    {
      ANNOTATE_HAPPENS_AFTER(&start_arc);
      decltype(sources) my_sources;
      #pragma omp for schedule(dynamic) nowait
      for(std::size_t i = 0; i < files.size(); i++) {
        auto pg = std::move(files[i]);
        auto s = ProfileSource::create_for(pg.first);
        if(s) {
          my_sources.emplace_back(std::move(s), std::move(pg.first));
          cnts_a[pg.second].fetch_add(1, std::memory_order_relaxed);
        } else if(pg.first.extension() == profileext) {
          util::log::warning{} << pg.first.filename().string() << " is named "
              "as a measurement profile but does not appear to be one";
        }
      }
      #pragma omp critical
      for(auto& sp: my_sources) sources.emplace_back(std::move(sp));
      ANNOTATE_HAPPENS_BEFORE(&end_arc);
    }
    ANNOTATE_HAPPENS_AFTER(&end_arc);
    for(std::size_t i = 0; i < cnts.size(); i++)
      cnts[i] = cnts_a[i].load(std::memory_order_relaxed);
  }
  cnts = mpi::allreduce(std::move(cnts), mpi::Op::sum());
  std::size_t totalcnt = 0;
  for(auto c: cnts) totalcnt += c;

  // If there are any arguments missing successes, rank 0 exits early
  if(mpi::World::rank() == 0) {
    if(totalcnt == 0) {
      std::cerr << "No input files given!\n"
                   "Usage: " << fs::path(argv[0]).filename().string()
                              << " " << summary << "\n";
      std::exit(2);
    }
    for(std::size_t g = 0; g < cnts.size(); g++) {
      if(cnts[g] == 0) {
        std::cerr << "Argument does not contain any profiles: " << argv[optind+g] << "\n";
        std::exit(2);
      }
    }
  }

  // Every rank over the average workload ships its extra paths up to rank 0,
  // every rank under the limit gives a report as to how many it can take.
  std::size_t limit = totalcnt / mpi::World::size();
  std::uint32_t avail = sources.size() < limit+1 ? limit+1 - sources.size() : 0;
  std::vector<std::string> extra;
  if(sources.size() > 0) {
    for(std::size_t i = sources.size()-1; i > limit; i--) {
      assert(i == sources.size()-1);
      extra.emplace_back(std::move(sources.back().second).string());
      sources.pop_back();
    }
  }
  auto avails = mpi::gather(avail, 0);
  auto extras = mpi::gather(std::move(extra), 0);

  // Allocate extra strings to ranks with available slots.
  if(avails && extras) {
    std::size_t numextra = 0;
    std::size_t numnearavail = 0;
    std::size_t numavail = 0;
    for(std::size_t r = 0; r < mpi::World::size(); r++) {
      numextra += extras.value()[r].size();
      if(avails.value()[r] > 1) numnearavail += avails.value()[r]-1;
      numavail += avails.value()[r];
    }

    std::vector<std::vector<std::string>> allocations(mpi::World::size());
    std::size_t next = 0;
    bool nearfull = false;
    for(auto& ps: *extras) {
      for(auto& p: ps) {
        while(1) {
          while((*avails)[next] <= (nearfull ? 0 : 1) && next < avails->size()) {
            next++;
          }
          if(next < avails->size()) break;
          assert(!nearfull && "Ran out of slots trying to allocate inputs to ranks!");
          // Try again, but allocate more aggressively
          nearfull = true;
          next = 0;
        }
        allocations[next].emplace_back(std::move(p));
        (*avails)[next] -= 1;
      }
    }
    extra = mpi::scatter(std::move(allocations), 0);
  } else extra = mpi::scatter<std::vector<std::string>>(0);

  // Add the inputs newly allocated to us to our set
  for(auto& p_s: extra) {
    stdshim::filesystem::path p = std::move(p_s);
    auto s = ProfileSource::create_for(p);
    if(!s) util::log::fatal{} << "Input " << p << " has changed on disk, please let it stablize before continuing!";
    sources.emplace_back(std::move(s), std::move(p));
  }
}

static std::pair<bool, fs::path> remove_prefix(const fs::path& path, const fs::path& pre) {
  if(pre.root_path() != path.root_path()) return {false, fs::path()};
  auto rpath = path.relative_path();
  auto rpathit = rpath.begin();
  auto rpathend = rpath.end();
  for(const auto& e: pre.relative_path()) {
    if(rpathit == rpathend) return {false, fs::path()};  // Missing a component
    if(e == "") break;  // Directory ending component
    if(e == "*") {  // Glob-esque match
      ++rpathit;
      continue;
    }
    if(*rpathit != e) return {false, fs::path()};  // Wrong component
    ++rpathit;
  }
  fs::path rem;
  for(; rpathit != rpathend; ++rpathit) rem /= *rpathit;
  return {true, rem};
}

static std::optional<fs::path> search(const std::unordered_map<fs::path, fs::path>& prefixes,
                                      const fs::path& p) noexcept {
  std::error_code ec;
  for(const auto& ft: prefixes) {
    auto xp = remove_prefix(p, ft.first);
    if(xp.first) {
      fs::path rp = ft.second / xp.second;
      if(fs::is_regular_file(rp, ec)) return rp;
    }
  }
  if(fs::is_regular_file(p, ec)) return p;  // If all else fails;
  return std::nullopt;
}

void ProfArgs::StatisticsExtender::appendStatistics(const Metric& m, Metric::StatsAccess mas) noexcept {
  if(m.visibility() == Metric::Settings::visibility_t::invisible) return;
  Metric::Statistics s;
  s.sum = args.stats.sum;
  s.mean = args.stats.mean;
  s.min = args.stats.min;
  s.max = args.stats.max;
  s.stddev = args.stats.stddev;
  s.cfvar = args.stats.cfvar;
  mas.requestStatistics(std::move(s));
}

std::optional<fs::path> ProfArgs::Prefixer::resolvePath(const File& f) noexcept {
  return search(args.prefixes, f.path());
}

std::optional<fs::path> ProfArgs::Prefixer::resolvePath(const Module& m) noexcept {
  return search(args.prefixes, m.path());
}

std::optional<std::pair<util::optional_ref<Context>, Context&>>
ProfArgs::StructWarner::classify(Context& c, NestedScope& ns) noexcept {
  if(ns.flat().type() == Scope::Type::point) {
    // Check if there any Structfiles might match this Module
    const auto& m = ns.flat().point_data().first;
    const auto it = args.structheads.find(m.path().filename());
    if(it != args.structheads.end()) {
      std::cerr << "WARNING: Struct file partial match on "
                << m.path().filename().string() << ", did you forget a -R?\n"
                   "Suggestions:\n";
      for(const auto& pre: it->second) {
        std::cerr << "  -R '" << m.path().parent_path().string()
                              << "'='" << pre.string() << "'\n";
      }
    }
  }
  return std::nullopt;
}
