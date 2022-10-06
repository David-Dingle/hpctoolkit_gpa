import collections
import collections.abc
import itertools
import os
import platform
import re
import shlex
import shutil
import textwrap
import typing as T
from pathlib import Path

from .logs import FgColor, colorize
from .util import flatten


class Unsatisfiable(Exception):
    "Exception raised when the given variant is unsatisfiable"

    def __init__(self, missing):
        super().__init__(f"missing definition for argument {missing}")
        self.missing = missing


class Impossible(Exception):
    "Exception raised when the given variant is impossible"

    def __init__(self, a, b):
        super().__init__(f'conflict between "{a}" and "{b}"')
        self.a, self.b = a, b


def _multiwhich(*cmds: str) -> T.Optional[str]:
    for cmd in cmds:
        res = shutil.which(cmd)
        if res:
            return res
    return None


class DependencyConfiguration:
    """State needed to derive configure-time arguments, based on simple configuration files"""

    def __init__(self):
        self.configs = []

    def load(self, fn: Path, ctx: T.Optional[Path] = None):
        with open(fn, encoding="utf-8") as f:
            for line in f:
                self.configs.append((ctx, line))

    def get(self, argument):
        "Fetch the full form of the given argument, by searching the configs"
        argument = argument.lstrip()
        for ctx, line in self.configs:
            if line.startswith(argument):
                if argument[-1].isspace():
                    line = line[len(argument) :]
                line = line.strip()

                result = []
                for word in shlex.split(line):
                    envvars = {
                        "CC": os.environ.get("CC", _multiwhich("gcc", "icc", "cc")),
                        "CXX": os.environ.get("CXX", _multiwhich("g++", "icpc", "CC", "c++")),
                    }
                    if ctx is not None:
                        envvars["CTX"] = ctx.absolute().as_posix()
                    for var, val in envvars.items():
                        var = "${" + var + "}"
                        if var not in word:
                            continue
                        if val is None:
                            raise Unsatisfiable(var)
                        word = word.replace(var, val)
                    result.append(word)
                return result
        raise Unsatisfiable(argument)


class _ManifestFile:
    def __init__(self, path):
        self.path = Path(path)

    def check(self, installdir):
        if (Path(installdir) / self.path).is_file():
            return {self.path}, set()
        return set(), {self.path}


class _ManifestLib(_ManifestFile):
    def __init__(self, path, target, *aliases):
        super().__init__(path)
        self.target = str(target)
        self.aliases = [str(a) for a in aliases]

    def check(self, installdir):
        installdir = Path(installdir)
        found, missing = set(), set()

        target = installdir / self.path
        target = target.with_name(target.name + self.target)
        if not target.is_file():
            missing.add(target.relative_to(installdir))
        if target.is_symlink():
            missing.add(
                (target.relative_to(installdir), f"Unexpected symlink to {os.readlink(target)}")
            )

        for a in self.aliases:
            alias = installdir / self.path
            alias = alias.with_name(alias.name + a)
            if not alias.is_file():
                missing.add(alias.relative_to(installdir))
                continue
            if not alias.is_absolute():
                missing.add((alias.relative_to(installdir), "Not a symlink"))
                continue

            targ = Path(os.readlink(alias))
            if len(targ.parts) > 1:
                missing.add(
                    (alias.relative_to(installdir), "Invalid symlink, must point to sibling file")
                )
                continue
            if targ.name != target.name:
                missing.add(
                    (alias.relative_to(installdir), f"Invalid symlink, must point to {target.name}")
                )
                continue

            found.add(alias.relative_to(installdir))

        return found, missing


class _ManifestExtLib(_ManifestFile):
    def __init__(self, path, main_suffix, *suffixes):
        super().__init__(path)
        self.main_suffix = str(main_suffix)
        self.suffixes = [str(s) for s in suffixes]

    def check(self, installdir):
        installdir = Path(installdir)
        common = installdir / self.path
        found = set()

        main_path = common.with_name(common.name + self.main_suffix)
        if not main_path.is_file():
            return set(), {main_path.relative_to(installdir)}
        found.add(main_path.relative_to(installdir))

        for path in common.parent.iterdir():
            if path.name.startswith(common.name) and path != main_path:
                name = path.name[len(common.name) :]
                if any(re.match(s, name) for s in self.suffixes):
                    found.add(path.relative_to(installdir))

        return found, set()


class Manifest:
    """Representation of an install manifest"""

    def __init__(self, *, mpi: bool):
        """Given a set of variant-keywords, determine the install manifest as a list of ManifestFiles"""

        def so(n):
            return r"\.so" + r"\.\d+" * n.__index__()

        self.files = [
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libboost_atomic-mt", ".so", so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libboost_atomic", ".so", so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libboost_chrono", ".so", so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libboost_date_time-mt", ".so", so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libboost_date_time", ".so", so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libboost_filesystem-mt", ".so", so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libboost_filesystem", ".so", so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libboost_graph-mt", ".so", so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libboost_graph", ".so", so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libboost_regex-mt", ".so", so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libboost_regex", ".so", so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libboost_system-mt", ".so", so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libboost_system", ".so", so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libboost_thread-mt", ".so", so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libboost_thread", ".so", so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libboost_timer-mt", ".so", so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libboost_timer", ".so", so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libbz2", ".so", so(1), so(2), so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libcommon", ".so", so(2), so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libdw", ".so", so(1), r"-\d+\.\d+\.so"),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libdynDwarf", ".so", so(2), so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libdynElf", ".so", so(1), r"-\d+\.\d+\.so"),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libelf", ".so", so(1), r"-\d+\.\d+\.so"),
            _ManifestExtLib(
                "lib/hpctoolkit/ext-libs/libinstructionAPI", ".so", so(1), r"-\d+\.\d+\.so"
            ),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/liblzma", ".so", so(1), so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libmonitor_wrap", ".a"),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libmonitor", ".so", ".so.0", ".so.0.0.0"),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libparseAPI", ".so", so(2), so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libpfm", ".so", so(1), so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libsymtabAPI", ".so", so(2), so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libtbb", ".so", so(1)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libtbbmalloc_proxy", ".so", so(1)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libtbbmalloc", ".so", so(1)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libxerces-c", ".a"),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libxerces-c", ".so", r"-\d+.\d+\.so"),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libz", ".a"),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libz", ".so", so(1), so(3)),
            _ManifestFile("bin/hpclink"),
            _ManifestFile("bin/hpcprof"),
            _ManifestFile("bin/hpcrun"),
            _ManifestFile("bin/hpcstruct"),
            _ManifestFile("include/hpctoolkit.h"),
            _ManifestFile("lib/hpctoolkit/hash-file"),
            _ManifestFile("lib/hpctoolkit/libhpcrun_audit.a"),
            _ManifestFile("lib/hpctoolkit/libhpcrun_audit.la"),
            _ManifestFile("lib/hpctoolkit/libhpcrun_dlmopen.a"),
            _ManifestFile("lib/hpctoolkit/libhpcrun_dlmopen.la"),
            _ManifestFile("lib/hpctoolkit/libhpcrun_fake_audit.a"),
            _ManifestFile("lib/hpctoolkit/libhpcrun_fake_audit.la"),
            _ManifestFile("lib/hpctoolkit/libhpcrun_ga.a"),
            _ManifestFile("lib/hpctoolkit/libhpcrun_ga.la"),
            _ManifestFile("lib/hpctoolkit/libhpcrun_gprof.a"),
            _ManifestFile("lib/hpctoolkit/libhpcrun_gprof.la"),
            _ManifestFile("lib/hpctoolkit/libhpcrun_io.a"),
            _ManifestFile("lib/hpctoolkit/libhpcrun_io.la"),
            _ManifestFile("lib/hpctoolkit/libhpcrun_memleak.a"),
            _ManifestFile("lib/hpctoolkit/libhpcrun_memleak.la"),
            _ManifestFile("lib/hpctoolkit/libhpcrun_pthread.a"),
            _ManifestFile("lib/hpctoolkit/libhpcrun_pthread.la"),
            _ManifestFile("lib/hpctoolkit/libhpcrun_wrap.a"),
            _ManifestFile("lib/hpctoolkit/libhpcrun.o"),
            _ManifestFile("lib/hpctoolkit/libhpcrun.so"),
            # XXX: Why is there no libhpcrun.so.0.0.0 and libhpcrun.la?
            _ManifestFile("lib/hpctoolkit/libhpctoolkit.a"),
            _ManifestFile("lib/hpctoolkit/libhpctoolkit.la"),
            _ManifestFile("lib/hpctoolkit/plugins/ga"),
            _ManifestFile("lib/hpctoolkit/plugins/io"),
            _ManifestFile("lib/hpctoolkit/plugins/memleak"),
            _ManifestFile("lib/hpctoolkit/plugins/pthread"),
            # XXX: Why is there no gprof?
            _ManifestFile("libexec/hpctoolkit/config.guess"),
            _ManifestFile("libexec/hpctoolkit/dotgraph-bin"),
            _ManifestFile("libexec/hpctoolkit/dotgraph"),
            _ManifestFile("libexec/hpctoolkit/hpcfnbounds"),
            _ManifestFile("libexec/hpctoolkit/hpcguess"),
            _ManifestFile("libexec/hpctoolkit/hpclog"),
            _ManifestFile("libexec/hpctoolkit/hpcplatform"),
            _ManifestFile("libexec/hpctoolkit/hpcproftt-bin"),
            _ManifestFile("libexec/hpctoolkit/hpcproftt"),
            _ManifestFile("libexec/hpctoolkit/hpcsummary"),
            _ManifestFile("libexec/hpctoolkit/hpctracedump"),
            _ManifestFile("libexec/hpctoolkit/renamestruct.sh"),
            _ManifestFile("share/doc/hpctoolkit/documentation.html"),
            _ManifestFile("share/doc/hpctoolkit/download.html"),
            _ManifestFile("share/doc/hpctoolkit/examples.html"),
            _ManifestFile("share/doc/hpctoolkit/fig/hpctoolkit-workflow.png"),
            _ManifestFile("share/doc/hpctoolkit/fig/hpcviewer-annotated-screenshot.jpg"),
            _ManifestFile("share/doc/hpctoolkit/fig/index.html"),
            _ManifestFile("share/doc/hpctoolkit/fig/spacer.gif"),
            _ManifestFile("share/doc/hpctoolkit/FORMATS.md"),
            _ManifestFile("share/doc/hpctoolkit/googleeeb6a75d4102e1ef.html"),
            _ManifestFile("share/doc/hpctoolkit/hpctoolkit.org.sitemap.txt"),
            _ManifestFile("share/doc/hpctoolkit/index.html"),
            _ManifestFile("share/doc/hpctoolkit/info-acks.html"),
            _ManifestFile("share/doc/hpctoolkit/info-people.html"),
            _ManifestFile("share/doc/hpctoolkit/LICENSE"),
            _ManifestFile("share/doc/hpctoolkit/man/hpclink.html"),
            _ManifestFile("share/doc/hpctoolkit/man/hpcprof-mpi.html"),
            _ManifestFile("share/doc/hpctoolkit/man/hpcprof.html"),
            _ManifestFile("share/doc/hpctoolkit/man/hpcproftt.html"),
            _ManifestFile("share/doc/hpctoolkit/man/hpcrun.html"),
            _ManifestFile("share/doc/hpctoolkit/man/hpcstruct.html"),
            _ManifestFile("share/doc/hpctoolkit/man/hpctoolkit.html"),
            _ManifestFile("share/doc/hpctoolkit/man/hpcviewer.html"),
            _ManifestFile("share/doc/hpctoolkit/manual/HPCToolkit-users-manual.pdf"),
            _ManifestFile("share/doc/hpctoolkit/overview.html"),
            _ManifestFile("share/doc/hpctoolkit/publications.html"),
            _ManifestFile("share/doc/hpctoolkit/README.Acknowledgments"),
            _ManifestFile("share/doc/hpctoolkit/README.Install"),
            _ManifestFile("share/doc/hpctoolkit/README.md"),
            _ManifestFile("share/doc/hpctoolkit/README.ReleaseNotes"),
            _ManifestFile("share/doc/hpctoolkit/software-instructions.html"),
            _ManifestFile("share/doc/hpctoolkit/software.html"),
            _ManifestFile("share/doc/hpctoolkit/spack-issues.html"),
            _ManifestFile("share/doc/hpctoolkit/style/footer-hpctoolkit.js"),
            _ManifestFile("share/doc/hpctoolkit/style/header-hpctoolkit.js"),
            _ManifestFile("share/doc/hpctoolkit/style/header.gif"),
            _ManifestFile("share/doc/hpctoolkit/style/index.html"),
            _ManifestFile("share/doc/hpctoolkit/style/style.css"),
            _ManifestFile("share/doc/hpctoolkit/training.html"),
            _ManifestFile("share/hpctoolkit/dtd/hpc-experiment.dtd"),
            _ManifestFile("share/hpctoolkit/dtd/hpc-structure.dtd"),
            _ManifestFile("share/hpctoolkit/dtd/hpcprof-config.dtd"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/isoamsa.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/isoamsb.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/isoamsc.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/isoamsn.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/isoamso.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/isoamsr.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/isobox.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/isocyr1.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/isocyr2.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/isodia.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/isogrk3.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/isolat1.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/isolat2.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/isomfrk.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/isomopf.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/isomscr.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/isonum.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/isopub.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/isotech.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/mathml.dtd"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/mmlalias.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/mmlextra.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/xhtml1-transitional-mathml.dtd"),
            _ManifestFile("share/man/man1/hpclink.1hpctoolkit"),
            _ManifestFile("share/man/man1/hpcprof-mpi.1hpctoolkit"),
            _ManifestFile("share/man/man1/hpcprof.1hpctoolkit"),
            _ManifestFile("share/man/man1/hpcproftt.1hpctoolkit"),
            _ManifestFile("share/man/man1/hpcrun.1hpctoolkit"),
            _ManifestFile("share/man/man1/hpcstruct.1hpctoolkit"),
            _ManifestFile("share/man/man1/hpctoolkit.1hpctoolkit"),
            _ManifestFile("share/man/man1/hpcviewer.1hpctoolkit"),
            _ManifestLib("lib/hpctoolkit/libhpcrun_audit.so", ".0.0.0", ".0", ""),
            _ManifestLib("lib/hpctoolkit/libhpcrun_dlmopen.so", ".0.0.0", ".0", ""),
            _ManifestLib("lib/hpctoolkit/libhpcrun_fake_audit.so", ".0.0.0", ".0", ""),
            _ManifestLib("lib/hpctoolkit/libhpcrun_ga.so", ".0.0.0", ".0", ""),
            _ManifestLib("lib/hpctoolkit/libhpcrun_gprof.so", ".0.0.0", ".0", ""),
            _ManifestLib("lib/hpctoolkit/libhpcrun_io.so", ".0.0.0", ".0", ""),
            _ManifestLib("lib/hpctoolkit/libhpcrun_memleak.so", ".0.0.0", ".0", ""),
            _ManifestLib("lib/hpctoolkit/libhpcrun_pthread.so", ".0.0.0", ".0", ""),
            _ManifestLib("lib/hpctoolkit/libhpctoolkit.so", ".0.0.0", ".0", ""),
        ]

        if mpi:
            self.files += [
                _ManifestFile("bin/hpcprof-mpi"),
            ]

    def check(self, installdir: Path) -> tuple[int, int]:
        """Scan an install directory and compare against the expected manifest. Prints the results
        of the checks to the log. Return the counts of missing and unexpected files."""

        # First derive the full listing of actually installed files
        listing = set()
        for root, _, files in os.walk(installdir):
            for fn in files:
                listing.add((Path(root) / fn).relative_to(installdir))

        # Then match these files up with the results we found
        n_unexpected = 0
        n_uninstalled = 0
        warnings = []
        errors = []
        for f in self.files:
            found, not_found = f.check(installdir)
            warnings.extend(f"+ {fn.as_posix()}" for fn in found - listing)
            n_unexpected += len(found - listing)
            listing -= found
            for fn in not_found:
                if isinstance(fn, tuple):
                    fn, msg = fn
                    errors.append(f"! {fn.as_posix()}\n  ^ {textwrap.indent(msg, '    ')}")
                else:
                    errors.append(f"- {fn.as_posix()}")
            n_uninstalled += len(not_found)

        # Print out the warnings and then the errors, with colors
        with colorize(FgColor.warning):
            for hunk in warnings:
                print(hunk)
        with colorize(FgColor.error):
            for hunk in errors:
                print(hunk)

        return n_uninstalled, n_unexpected


class Configuration:
    """Representation of a possible build configuration of HPCToolkit"""

    def __init__(self, depcfg: DependencyConfiguration, variant: dict[str, bool]):
        """Derive the full Configuration from the given DependencyConfiguration and variant-keywords."""
        make = shutil.which("make")
        if make is None:
            raise RuntimeError("Unable to find make!")
        self.make: str = make

        self.manifest: Manifest = Manifest(mpi=variant["mpi"])

        fragments: list[str] = self.__class__._collect_fragments(depcfg, variant)

        # Parse the now-together fragments to derive the environment and configure args
        self.args: list[str] = []
        self.env: T.Any = collections.ChainMap({}, os.environ)
        for arg in flatten(fragments):
            m = re.fullmatch(r"ENV\{(\w+)\}=(.*)", arg)
            if m is None:
                self.args.append(arg)
            else:
                self.env[m.group(1)] = m.group(2)

    @staticmethod
    def _collect_fragments(depcfg: DependencyConfiguration, variant: dict[str, bool]) -> list[str]:
        fragments = [
            depcfg.get("--with-boost="),
            depcfg.get("--with-bzip="),
            depcfg.get("--with-dyninst="),
            depcfg.get("--with-elfutils="),
            depcfg.get("--with-tbb="),
            depcfg.get("--with-libmonitor="),
            depcfg.get("--with-libunwind="),
            depcfg.get("--with-xerces="),
            depcfg.get("--with-lzma="),
            depcfg.get("--with-zlib="),
            depcfg.get("--with-libiberty="),
            depcfg.get("--with-memkind="),
            depcfg.get("--with-yaml-cpp="),
        ]

        if platform.machine() == "x86_64":
            fragments.append(depcfg.get("--with-xed="))

        if variant["papi"]:
            fragments.append(depcfg.get("--with-papi="))
        else:
            fragments.append(depcfg.get("--with-perfmon="))

        if variant["cuda"]:
            fragments.append(depcfg.get("--with-cuda="))

        if variant["level0"]:
            fragments.append(depcfg.get("--with-level0="))

        if variant["opencl"]:
            fragments.append(depcfg.get("--with-opencl="))

        # if False:  # TODO: GTPin
        #     fragments.extend(
        #         [
        #             depcfg.get("--with-gtpin="),
        #             depcfg.get("--with-igc="),
        #         ]
        #     )

        if variant["rocm"]:
            try:
                fragments.append(depcfg.get("--with-rocm="))
            except Unsatisfiable:
                # Try the split-form arguments instead
                fragments.extend(
                    [
                        depcfg.get("--with-rocm-hip="),
                        depcfg.get("--with-rocm-hsa="),
                        depcfg.get("--with-rocm-tracer="),
                        depcfg.get("--with-rocm-profiler="),
                    ]
                )

        # if False:  # TODO: all-static (do we really want to support this?)
        #     fragments.append("--enable-all-static")

        fragments.extend([f"MPI{cc}=" for cc in ("CC", "F77")])
        if variant["mpi"]:
            fragments.append(depcfg.get("MPICXX="))
            fragments.append("--enable-force-hpcprof-mpi")
        else:
            fragments.append("MPICXX=")

        if variant["debug"]:
            fragments.append("--enable-develop")

        return fragments

    @classmethod
    def all_variants(cls):
        """Generate a list of all possible variants as dictionaries of variant-keywords"""

        def vbool(x, first=False):
            return [(x, first), (x, not first)]

        return map(
            dict,
            itertools.product(
                *reversed(
                    [
                        vbool("mpi"),
                        vbool("debug", True),
                        vbool("papi", True),
                        vbool("opencl"),
                        vbool("cuda"),
                        vbool("rocm"),
                        vbool("level0"),
                    ]
                )
            ),
        )

    @staticmethod
    def to_string(
        variant: dict[str, bool],
        separator: str = " ",
    ) -> str:
        """Generate the string form for a variant set"""

        def vbool(n):
            return f"+{n}" if variant[n] else f"~{n}"

        return separator.join(
            [
                vbool("mpi"),
                vbool("debug"),
                vbool("papi"),
                vbool("opencl"),
                vbool("cuda"),
                vbool("rocm"),
                vbool("level0"),
            ]
        )

    @staticmethod
    def parse(arg: str) -> dict[str, bool]:
        """Parse a variant-spec (vaugely Spack format) into a variant (dict)"""
        result = {}
        for wmatch in re.finditer(r"[+~\w]+", arg):
            word = wmatch.group(0)
            if word[0] not in "+~":
                raise ValueError("Variants must have a value indicator (+~): " + word)
            for match in re.finditer(r"([+~])(\w*)", word):
                value, variant = match.group(1), match.group(2)
                result[variant] = value == "+"
        for k in result:
            if k not in ("mpi", "debug", "papi", "opencl", "cuda", "rocm", "level0"):
                raise ValueError(f"Invalid variant name {k}")
        return result

    @staticmethod
    def satisfies(specific: dict[str, bool], general: dict[str, bool]) -> bool:
        """Test if the `specific` variant is a subset of a `general` variant"""
        for k, v in general.items():
            if k in specific and specific[k] != v:
                return False
        return True
