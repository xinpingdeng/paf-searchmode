#ifndef _CUH
#define _CUH

#include "paf_diskdb.cuh"

#include "dada_hdu.h"
#include "dada_def.h"
#include "ipcio.h"
#include "multilog.h"
#include "ascii_header.h"
#include "daemon.h"
#include "futils.h"

#include <cuda_runtime.h>
#include <cuda.h>

#define DADA_HDR_SIZE   4096

typedef struct conf_t
{
  key_t key;
  int sod;
  char fname[MSTR_LEN], hfname[MSTR_LEN];
  FILE *fp;
  dada_hdu_t *hdu;
  multilog_t *log;
  size_t hdrsz;
  size_t rbufsz;
  int device_id;
}conf_t;

int init_diskdb(conf_t *conf);
int destroy_diskdb(conf_t conf);
int do_diskdb(conf_t conf);
#endif
