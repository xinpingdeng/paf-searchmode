#ifndef _PROCESS_CUH
#define _PROCESS_CUH

#include <cuda_runtime.h>
#include <cuda.h>
#include <cufft.h>
#include <stdio.h>

#include "dada_cuda.h"
#include "dada_hdu.h"
#include "dada_def.h"
#include "ipcio.h"
#include "ascii_header.h"
#include "daemon.h"
#include "futils.h"
#include "paf_process.cuh"

#define CUFFT_NX32

#define DADA_HDR_SIZE         4096
#define NCHK_NIC              48   // How many frequency chunks we will receive, we should read the number from metadata
#define NCHAN_CHK             7
#define NSAMP_DF              128
#define NPOL_SAMP             2
#define NDIM_POL              2

#define NBYTE_RT              8    // cudaComplex
#define NBYTE_IN              2    // int16_t
#define NBYTE_OUT             1    // int8_t
//#define NBYTE_OUT             4    // float

#define OSAMP_RATEI           0.84375
#define CUFFT_RANK1           1
#define CUFFT_RANK2           1

#ifdef CUFFT_NX32
#define CUFFT_NX1             32
#define CUFFT_MOD1            13             // Set to remove oversampled data
#define NCHAN_KEEP1           27             // (OSAMP_RATEI * CUFFT_NX1)
#define CUFFT_NX2             32
#define CUFFT_MOD2            16              // CUFFT_NX2 / 2
#define NCHAN_KEEP2           8192           // (CUFFT_NX2 * NCHAN_FINAL)
#define NCHAN_EDGE            440             // (NCHAN_KEEP1 * NCHK_NIC * NCHAN_CHK - CUFFT_NX2 * NCHAN_FINAL)/2
#define TILE_DIM              32              // CUFFT_NX2
#define NROWBLOCK_TRANS       8               // a good number which can be devided by CUFFT_NX2 (TILE_DIM)
#endif

#ifdef CUFFT_NX64
#define CUFFT_NX1             64
#define CUFFT_MOD1            27             // Set to remove oversampled data
#define NCHAN_KEEP1           54             // (OSAMP_RATEI * CUFFT_NX1)
#define CUFFT_NX2             64
#define CUFFT_MOD2            32              // CUFFT_NX2 / 2
#define NCHAN_KEEP2           16384           // (CUFFT_NX2 * NCHAN_FINAL)
#define NCHAN_EDGE            880             // (NCHAN_KEEP1 * NCHK_NIC * NCHAN_CHK - CUFFT_NX2 * NCHAN_FINAL)/2
#define TILE_DIM              64              // CUFFT_NX2
#define NROWBLOCK_TRANS       16              // a good number which can be devided by CUFFT_NX2 (TILE_DIM)
#endif

#define NCHAN_RATEI           1.107421875     // (NCHAN_KEEP1 * NCHK_NIC * NCHAN_CHK)/(CUFFT_NX2 * NCHAN_FINAL)
#define NCHAN_FINAL           256             // Final number of channels

#define SCL_INT8              127.0f          // For int8_t
#define SCL_NSIG              4.0f            // 4 sigma, 99.993666%

/* 
   The following parameters are for the speedup of transpose_scale_kernel3 and transpose_scale_kernel4;
   Be careful here as these parameters are sort of fixed. 
   For example, if we change the NCHAN_FINAL or CUFFT_NX2, we need to change the parameters here;
   We may not be able to find a suitable parameter if we do the change;
   In short, the current setup here only works with selected configuration.
   The second look into it turns out that the parameters here are more general.
*/
//#define TILE_DIM              32              // CUFFT_NX2
//#define NROWBLOCK_TRANS       8               // a good number which can be devided by CUFFT_NX2 (TILE_DIM)

typedef struct conf_t
{
  int device_id;
  int stream;
  
  char hfname[MSTR_LEN];
  int sod;
  int stream_ndf;
  int nstream;
  float scl_ndim;

  int nrun_blk;
  char dir[MSTR_LEN];
  char utc_start[MSTR_LEN];
  uint64_t picoseconds;
  
  key_t key_out, key_in;
  dada_hdu_t *hdu_out, *hdu_in;
  
  char *hdrbuf_in, *hdrbuf_out;
  int64_t *dbuf_in;
  int8_t *dbuf_out;
  //float *dbuf_out;

  double freq;
  
  double rbufin_ndf;
  size_t bufin_size, bufout_size; // Device buffer size for all streams
  size_t sbufin_size, sbufout_size; // Buffer size for each stream
  size_t bufrt1_size, bufrt2_size;
  size_t sbufrt1_size, sbufrt2_size;
  cufftComplex *buf_rt1, *buf_rt2;
  size_t hbufin_offset, dbufin_offset;
  size_t bufrt1_offset, bufrt2_offset;
  size_t dbufout_offset, hbufout_offset;
  size_t nsamp1, npol1, ndata1;
  size_t nsamp2, npol2, ndata2;
  //size_t nbufin_rbuf;   // How many input GPU memory buffer can be fitted into the input ring buffer;
  
  size_t hdrsz, rbufin_size, rbufout_size; // HDR size for both HDU and ring buffer size of input HDU;
  // Input ring buffer size is different from the size of bufin, which is the size for GPU input memory;
  // Out ring buffer size is the same with the size of bufout, which is the size for GPU output memory;
  
  float *ddat_offs, *dsquare_mean, *ddat_scl;
  float *hdat_offs, *hsquare_mean, *hdat_scl;
  cudaStream_t *streams;
  cufftHandle *fft_plans1, *fft_plans2;
  
  dim3 gridsize_unpack, blocksize_unpack;
  dim3 gridsize_transpose_pad, blocksize_transpose_pad;
  dim3 gridsize_sum1, blocksize_sum1;
  dim3 gridsize_sum2, blocksize_sum2;
  dim3 gridsize_swap_select_transpose, blocksize_swap_select_transpose;
  dim3 gridsize_swap_select_transpose_swap, blocksize_swap_select_transpose_swap;
  dim3 gridsize_swap, blocksize_swap;
  dim3 gridsize_scale, blocksize_scale;
  
  dim3 gridsize_mean, blocksize_mean;
  dim3 gridsize_transpose_scale, blocksize_transpose_scale;
  dim3 gridsize_transpose_scale2, blocksize_transpose_scale2;
  dim3 gridsize_transpose_scale3, blocksize_transpose_scale3;
  dim3 gridsize_transpose_scale4, blocksize_transpose_scale4;
  dim3 gridsize_transpose_float, blocksize_transpose_float;
}conf_t; 

int init_process(conf_t *conf);
int do_process(conf_t conf);
int dat_offs_scl(conf_t conf);
int register_header(conf_t *conf);

int destroy_process(conf_t conf);

#endif