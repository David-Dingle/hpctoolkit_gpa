// SPDX-FileCopyrightText: 2002-2024 Rice University
//
// SPDX-License-Identifier: BSD-3-Clause

// -*-Mode: C++;-*- // technically C99

#define _GNU_SOURCE

#include <unistd.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "executable-path.h"

/*
 *******************************************************************************
 * forward declarations
 *******************************************************************************
 */

static int assemble_fullpath(const char *prefix, const int terminator,
                             const char *suffix, char *result,
                             char *result_end);



/*
 *******************************************************************************
 * interface operations
 *******************************************************************************
 */

/*
 * NOTES:
 * 1. path_list is colon separated
 * 2. pathresult must point to a buffer of length PATH_MAX
 * 2. this routine can't allocate any memory, which
 *    makes the implementation a bit more complicated
 *    because it can't handle cases as uniformly
 *    using in-place modification.
 */
char *
executable_path(const char *filename, const char *path_list,
                char *executable_name)
{
  if (!access(filename, F_OK)) return realpath(filename, executable_name);
  else {
    char path[PATH_MAX];
    int failure;
    const char *path_prefix;
    const char *colon;

    /* check for bad absolute path (lookup already failed above) */
    if (*filename == '/') return NULL;

    /* look for the file using each path in path_list */
    path_prefix = path_list;
    colon = path_list;
    while (path_prefix) {
      colon = strchr(path_prefix,':'); /* find the end of path_prefix */

      /*
       * assemble a new path using the current path_prefix. the character
       * marking the end of path_prefix is either a ':' if there is another
       * path in path_list or it is a 0 if path_prefix points to the last
       * prefix
       */
      failure = assemble_fullpath(path_prefix, (colon ? ':' : 0), filename,
                                  path, &path[PATH_MAX - 1]);
      if (failure) return NULL;

      /* if the file is present at path, return its real path in
       * executable_name
       */
      if (!access(path, X_OK)) return realpath(path, executable_name);

      /* move path_prefix to the next path in path_list, if any */
      path_prefix = (colon ? colon + 1 : NULL);
    }
  }

  return NULL;
}



/*
 *******************************************************************************
 * private operations
 *******************************************************************************
 */

/*
 * function: strtcpy
 *
 * purpose:
 *   copy string from src to dest until either the terminating character t
 *   is encountered in the src string, or dest passes dest_end
 *
 * arguments: t is the terminating character
 */
static char *
strtcpy(char *dest, const char *src, const int t, char *dest_end)
{
  for (;;) {
    char c = *src++;
    if (c == t) break;
    if (dest > dest_end) return 0;
    *dest++ = c;
  }
  return dest;
}


static int
assemble_fullpath(const char *prefix, const int terminator,
                  const char *suffix, char *result, char *result_end)
{
  char *end;

  /* copy prefix */
  end = strtcpy(result, prefix, terminator, result_end);
  if (!end) return -1; /* string too long */

  /* ensure path prefix is terminated by a slash */
  if (*(end - 1) != '/') {
    if (end > result_end) return -1;
    else *end++ = '/';
  }

  /* append suffix */
  end = strtcpy(end, suffix, 0, result_end);
  if (!end) return -1;

  /* null terminate */
  if (end > result_end) return -1;
  *end = 0;

  return 0;
}
