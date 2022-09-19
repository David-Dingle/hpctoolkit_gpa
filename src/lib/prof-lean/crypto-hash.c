// -*-Mode: C++;-*- // technically C99

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
// Copyright ((c)) 2002-2022, Rice University
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
//   crypto-hash.c
//
// Purpose:
//   compute a cryptographic hash of a sequence of bytes. this is used
//   to name information presented to hpcrun in memory (e.g. a GPU binary) 
//   that needs to be saved for post-mortem analysis.
//
//***************************************************************************


//*****************************************************************************
// system includes
//*****************************************************************************

#include <assert.h>
#include <string.h>

#include "md5.h"
#include "crypto-hash.h"



//*****************************************************************************
// macros
//*****************************************************************************

 

#define LOWER_NIBBLE_MASK 	(0x0f)
#define UPPER_NIBBLE(c) 	((c >> 4) & LOWER_NIBBLE_MASK)
#define LOWER_NIBBLE(c) 	(c  & LOWER_NIBBLE_MASK)

#define HEX_TO_ASCII(c) ((c > 9) ?  'a' + (c - 10) : ('0' + c))


//*****************************************************************************
// interface operations
//*****************************************************************************

//-----------------------------------------------------------------------------
// function: 
//   crypto_hash_compute
//
// arguments:
//   input:        
//     pointer to a vector of bytes that will be crytographically hashed
//   input_length:        
//     length in bytes of the input
//   hash:        
//     pointer to a vector of bytes of length >= CRYPTO_HASH_LENGTH
//
// return value:
//   0: success
//   non-zero: failure
//-----------------------------------------------------------------------------
int
crypto_hash_compute
(
  const unsigned char *input, 
  size_t input_length,
  unsigned char *hash,
  unsigned int hash_length
)
{
  struct md5_context context;
  struct md5_digest digest;

  if (hash_length < CRYPTO_HASH_LENGTH) {
    // failure: caller not prepared to accept a hash of at least the length 
    // that we will provide
    return -1;
  }

  // zero the hash result
  memset(hash, 0, hash_length); 

  // compute an MD5 hash of input
  md5_init(&context);
  md5_update(&context, input, (unsigned int) input_length);
  md5_finalize(&context, &digest);

  memcpy(hash, &digest, CRYPTO_HASH_LENGTH);

  return 0;
}


//-----------------------------------------------------------------------------
// function: 
//   crypto_hash_to_hexstring
//
// arguments:
//   hash:        
//     pointer to crytographic hash computed by cryto_hash_compute
//   hash_string: 
//     pointer to character buffer where string equivalent of the hash code 
//     will be written
//   hash_string_length: 
//     length of the hash string must be >= CRYPTO_HASH_STRING_LENGTH
//
// return value:
//   0: success
//   non-zero: failure
//-----------------------------------------------------------------------------
int
crypto_hash_to_hexstring
(
  const unsigned char *hash,
  char *hash_string,
  unsigned int hash_string_length
)
{
  if (hash_string_length < CRYPTO_HASH_STRING_LENGTH) {
    return -1;
  }

  int chars = CRYPTO_HASH_LENGTH;
  while (chars-- > 0) {
    unsigned char val_u = UPPER_NIBBLE(*hash); 
    *hash_string++ = HEX_TO_ASCII(val_u);

    unsigned char val_l = LOWER_NIBBLE(*hash);
    *hash_string++ = HEX_TO_ASCII(val_l);
    hash++;
  }
  *hash_string++ = 0;

  return 0;
}


//-----------------------------------------------------------------------------
// function:
//   crypto_compute_hash_string
//
// arguments:
//   data:
//     pointer to data to hash
//   data_size:
//     length of data in bytes
//   hash_string:
//     pointer to result string from hashing data bytes
//   hash_string_length:
//     length of the hash string must be >= CRYPTO_HASH_STRING_LENGTH
//
// return value:
//   0: success
//   non-zero: failure
//-----------------------------------------------------------------------------
int
crypto_compute_hash_string
(
 const unsigned char *data,
 size_t data_size,
 char *hash_string,
 unsigned int hash_string_length
)
{
  if (hash_string_length < CRYPTO_HASH_STRING_LENGTH) {
    return -1;
  }

  // Compute hash for data
  unsigned char hash[CRYPTO_HASH_LENGTH];
  crypto_hash_compute(data, data_size, hash, CRYPTO_HASH_LENGTH);

  // Turn hash into string
  crypto_hash_to_hexstring(hash, hash_string, CRYPTO_HASH_STRING_LENGTH);

  return 0;
}



//******************************************************************************
// unit test
//******************************************************************************

// To run the crypto/md5 unit test, change #if to 1 and compile as:
//
//   gcc -g -O crypto-hash.c md5.c -o hash
//
// Run as:  ./hash filename
//
// Results should match the output of /usr/bin/md5sum.
//

#ifdef UNIT_TEST

#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int
main(int argc, char **argv)
{
  const char *filename = "/bin/ls";

  // check that the specified file is present
  struct stat statbuf;
  int stat_result = stat(filename, &statbuf);
  if (stat_result != 0) {
    printf("stat(%s) failed = %d\n", filename, stat_result);
    exit(-1);
  }
  size_t filesize = statbuf.st_size;

  char *filebuf; 
  filebuf = (char *) malloc(filesize);

  // read a file to hash
  FILE *file = fopen(filename, "r");
  if (file == 0) {
    printf("fopen(%s) failed.\n", filename);
    exit(-1);
  }

  size_t read_result = fread(filebuf, 1, filesize, file);
  if (read_result != filesize) {
    printf("read(%s) failed. expected %ld bytes, got %ld bytes\n", 
	   filename, filesize, read_result);
    exit(-1);
  }

  fclose(file);

  // allocate space for the hash
  unsigned char hash[CRYPTO_HASH_LENGTH];
  int crypto_result;

  // short buffer test
  crypto_result = crypto_hash_compute(filebuf, filesize, hash, CRYPTO_HASH_LENGTH - 1);
  if (crypto_result) {
    printf("crypto_hash_compute - success - detected a short hash buffer\n");
  } else {
    printf("crypto_hash_compute - fail - didn't detect short hash buffer\n");
    exit(-1);
  }

  // compute hash
  crypto_result = crypto_hash_compute(filebuf, filesize, hash, CRYPTO_HASH_LENGTH);
  if (crypto_result == 0) {
    printf("crypto_hash_compute - success - with right-sized hash buffer\n");
  } else {
    printf("crypto_hash_compute - fail - with right-sized hash buffer\n");
    exit(-1);
  }

  unsigned char long_hash[CRYPTO_HASH_LENGTH];

  // long buffer test
  crypto_result = crypto_hash_compute(filebuf, filesize, long_hash, sizeof(long_hash));
  if (crypto_result == 0) {
    printf("crypto_hash_compute - success - with long hash buffer\n");
  } else {
    printf("crypto_hash_compute - fail - with long hash buffer\n");
    exit(-1);
  }

  // allocate space for ASCII version of hash 
  char buffer[CRYPTO_HASH_STRING_LENGTH];

  // initialize buffer with recognizable output
  memset(buffer, '+', sizeof(buffer));

  // compute ASCII version of hash with short buffer
  crypto_result = crypto_hash_to_hexstring(hash, buffer, sizeof(buffer) - 1);
  if (crypto_result) {
    printf("crypto_hash_to_hexstring - success - detected short string buffer\n");
  } else {
    printf("crypto_hash_to_hexstring - fail - didn't short string buffer\n");
    exit(-1);
  }

  // initialize buffer with recognizable output
  memset(buffer, '+', sizeof(buffer));

  // compute ASCII version of hash with right-sized buffer
  crypto_result = crypto_hash_to_hexstring(hash, buffer, sizeof(buffer));
   if (crypto_result == 0) {
     printf("crypto_hash_to_hexstring - success - with right-sized buffer\n");
  } else {
    printf("crypto_hash_to_hexstring - fail - with right-sized buffer\n");
    exit(-1);
  }

  char long_buffer[CRYPTO_HASH_STRING_LENGTH+10];
  memset(long_buffer, '-', sizeof(long_buffer));

  // compute ASCII version of hash with long buffer
  crypto_result = crypto_hash_to_hexstring(hash, long_buffer, sizeof(long_buffer));
  if (crypto_result == 0) {
    printf("crypto_hash_to_hexstring - success - with long string buffer\n");
  } else {
    printf("crypto_hash_to_hexstring - fail - with long string buffer\n");
    exit(-1);
  }

   // compare string from right-sized buffer vs. long buffer
  if (strcmp(buffer, long_buffer) == 0) {
    printf("crypto_hash_to_hexstring - success - consistent output with long buffer\n");
  } else {
    printf("crypto_hash_to_hexstring - fail - inconsistent output with long buffer: %s vs. %s\n", 
	    buffer, long_buffer);
    exit(-1);
  }

  {
    char short_buffer[CRYPTO_HASH_STRING_LENGTH-1];
    char perfect_buffer[CRYPTO_HASH_STRING_LENGTH];
    char long_buffer[CRYPTO_HASH_STRING_LENGTH+10];
    memset(short_buffer, '-', sizeof(short_buffer));
    memset(perfect_buffer, '-', sizeof(perfect_buffer));
    memset(long_buffer, '=', sizeof(long_buffer));

    int s_len = crypto_compute_hash_string(filebuf, filesize, short_buffer, sizeof(short_buffer));
    int p_len = crypto_compute_hash_string(filebuf, filesize, perfect_buffer, sizeof(perfect_buffer));
    int l_len = crypto_compute_hash_string(filebuf, filesize, long_buffer, sizeof(long_buffer));

    if (s_len == 0) {
      printf("crypto_compute_hash_string - fail - didn't detect short string buffer\n");
      exit(-1);
    } else {
      printf("crypto_compute_hash_string - success - detected short string buffer\n");
    }

    if (p_len) {
      printf("crypto_compute_hash_string - fail - with right-sized string buffer\n");
      exit(-1);
    } else {
      printf("crypto_compute_hash_string - success - with right-sized string buffer\n");
    }

    if (l_len) {
      printf("crypto_compute_hash_string - fail - with long string buffer\n");
      exit(-1);
    } else {
      printf("crypto_compute_hash_string - success - with long string buffer\n");
    }

    if (strcmp(perfect_buffer, long_buffer) == 0) {
      printf("crypto_compute_hash_string - success - consistent output with long buffer\n");
    } else {
      printf("crypto_compute_hash_string - fail - inconsistent output "
        "with long buffer: %s vs. %s\n", perfect_buffer, long_buffer);
      exit(-1);
    }
  }

  free(filebuf);

  printf("hash string: %s %s\n", buffer, filename);

  return 0;
}

#endif  // end unit test
