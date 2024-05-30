/* src/include/hpctoolkit-config.h.  Generated from hpctoolkit-config.h.in by configure.  */
/* src/include/hpctoolkit-config.h.in.  Generated from configure.ac by autoheader.  */

/* binutils version 2.34 or later */
#define BINUTILS_234 1

/* "Cuda install prefix" */
#define CUDA_INSTALL_PREFIX "/usr/local/cuda/"

/* "CUPTI install prefix" */
#define CUPTI_INSTALL_PREFIX "/usr/local/cuda//extras/CUPTI/"

/* Data-centric tracing */
/* #undef DATACENTRIC_TRACE */

/* dyninst uses Instruction::Ptr */
/* #undef DYNINST_INSTRUCTION_PTR */

/* dyninst supports cuda */
#define DYNINST_USE_CUDA 1

/* dyninst built with libdw */
#define DYNINST_USE_LIBDW 1

/* Support for CLOCK_THREAD_CPUTIME_ID */
#define ENABLE_CLOCK_CPUTIME 1

/* Support for CLOCK_REALTIME and SIGEV_THREAD_ID */
#define ENABLE_CLOCK_REALTIME 1

/* ParseAPI supports openmp (for hpcstruct) */
#define ENABLE_OPENMP 1

/* Symtab supports openmp (for fnbounds) */
#define ENABLE_OPENMP_SYMTAB 1

/* Add extra annotations for debugging with Valgrind */
/* #undef ENABLE_VG_ANNOTATIONS */

/* Support for AMD XOP instructions */
/* #undef ENABLE_XOP */

/* The <cupti.h> header file is available. */
#define HAVE_CUPTI_H 1

/* Define to 1 if you have the <cxxabi.h> header file. */
#define HAVE_CXXABI_H 1

/* Define to 1 if you have the <dlfcn.h> header file. */
#define HAVE_DLFCN_H 1

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define to 1 if you have the <memory.h> header file. */
#define HAVE_MEMORY_H 1

/* HOST OS: 32 and 64 bit OS libraries */
#define HAVE_OS_MULTILIB 1

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if the system has the type `uint'. */
#define HAVE_UINT 1

/* C compiler supports type "uint" */
#define HAVE_UINT_LANG_C 1

/* Define to 1 if the system has the type `ulong'. */
#define HAVE_ULONG 1

/* C compiler supports type "ulong" */
#define HAVE_ULONG_LANG_C 1

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* Define to 1 if the system has the type `ushort'. */
#define HAVE_USHORT 1

/* C compiler supports type "ushort" */
#define HAVE_USHORT_LANG_C 1

/* C compiler supports type "voidp" */
/* #undef HAVE_VOIDP_LANG_C */

/* Host is big endian. */
/* #undef HOST_BIG_ENDIAN */

/* HOST CPU: ARM 64 (aarch64 */
/* #undef HOST_CPU_ARM64 */

/* HOST CPU: ia64 (itanium) */
/* #undef HOST_CPU_IA64 */

/* HOST CPU: PowerPC (ppc) */
/* #undef HOST_CPU_PPC */

/* HOST CPU: x86 (32-bit) */
/* #undef HOST_CPU_x86 */

/* HOST CPU: x86-64 */
#define HOST_CPU_x86_64 1

/* Host is little endian. */
#define HOST_LITTLE_ENDIAN 1

/* HOST OS: IRIX */
/* #undef HOST_OS_IRIX */

/* HOST OS: Linux */
#define HOST_OS_LINUX 1

/* HOST OS: MacOS */
/* #undef HOST_OS_MACOS */

/* HOST OS: Solaris */
/* #undef HOST_OS_SOLARIS */

/* HOST OS: Tru64 */
/* #undef HOST_OS_TRU64 */

/* HOST platform: MIPS64LE_LINUX */
/* #undef HOST_PLATFORM_MIPS64LE_LINUX */

/* IBM Blue Gene support */
/* #undef HOST_SYSTEM_IBM_BLUEGENE */

/* Git branch and commit hash, if known. */
#define HPCTOOLKIT_GIT_VERSION "unknown (not a git repo)"

/* HPCToolkit install prefix */
#define HPCTOOLKIT_INSTALL_PREFIX "/home/xjding/Projects/GPA/GPA/gpa/hpctoolkit"

/* Spack version and variants, if known. */
#define HPCTOOLKIT_SPACK_SPEC "unknown (not a spack build)"

/* HPCToolkit version */
#define HPCTOOLKIT_VERSION "2020.07-develop"

/* HPCToolkit version string */
#define HPCTOOLKIT_VERSION_STRING "A member of HPCToolkit, version 2020.07-develop"

/* Define to the sub-directory where libtool stores uninstalled libraries. */
#define LT_OBJDIR ".libs/"

/* Standard C headers */
/* #undef NO_STD_CHEADERS */

/* Have support for cuda */
#define OPT_HAVE_CUDA 1

/* Name of package */
#define PACKAGE "hpctoolkit"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "hpctoolkit-forum@rice.edu"

/* Define to the full name of this package. */
#define PACKAGE_NAME "hpctoolkit"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "hpctoolkit 2020.07-develop"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "hpctoolkit"

/* Define to the home page for this package. */
#define PACKAGE_URL "http://hpctoolkit.org/"

/* Define to the version of this package. */
#define PACKAGE_VERSION "2020.07-develop"

/* The size of `void*', as computed by sizeof. */
#define SIZEOF_VOIDP 8

/* Define to 1 if you have the ANSI C header files. */
#define STDC_HEADERS 1

/* libunwind is the specified unwinder */
/* #undef USE_LIBUNW */

/* use ROCM */
/* #undef USE_ROCM */

/* Use system byteswap.h */
#define USE_SYSTEM_BYTESWAP 1

/* Version number of package */
#define VERSION "2020.07-develop"

/* Define WORDS_BIGENDIAN to 1 if your processor stores words with the most
   significant byte first (like Motorola and SPARC, unlike Intel). */
#if defined AC_APPLE_UNIVERSAL_BUILD
# if defined __BIG_ENDIAN__
#  define WORDS_BIGENDIAN 1
# endif
#else
# ifndef WORDS_BIGENDIAN
/* #  undef WORDS_BIGENDIAN */
# endif
#endif

/* Fix pthread.h */
/* #undef __thread */
