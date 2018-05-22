#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>
#include <math.h>

#include "multilog.h"
#include "process.cuh"
#include "cudautil.cuh"
#include "kernel.cuh"

extern multilog_t *runtime_log;

int init_process(conf_t *conf)
{
  CudaSafeCall(cudaSetDevice(conf->device_id));
  
  int i;
  int iembed1, istride1, idist1, oembed1, ostride1, odist1, batch1, nx1;
  int iembed2, istride2, idist2, oembed2, ostride2, odist2, batch2, nx2;
  ipcbuf_t *db = NULL;
  
  /* Prepare buffer, stream and fft plan for process */
  conf->scl_ndim    = conf->rbufin_ndf * NSAMP_DF * NPOL_SAMP * NDIM_POL;
  conf->nsamp1      = conf->stream_ndf * NCHK_NIC * NCHAN_CHK * NSAMP_DF;
  conf->npol1       = conf->nsamp1 * NPOL_SAMP;
  conf->ndata1      = conf->npol1  * NDIM_POL;

  conf->nsamp2      = conf->nsamp1 * OSAMP_RATEI / NCHAN_RATEI;
  conf->npol2       = conf->nsamp2 * NPOL_SAMP;
  conf->ndata2      = conf->npol2  * NDIM_POL;
  
  nx1        = CUFFT_NX1;
  batch1     = conf->npol1 / CUFFT_NX1;
  
  iembed1    = nx1;
  istride1   = 1;
  idist1     = nx1;
  
  oembed1    = nx1;
  ostride1   = 1;
  odist1     = nx1;
  
  nx2        = CUFFT_NX2;
  batch2     = conf->npol2 / CUFFT_NX2;
  
  iembed2    = nx2;
  istride2   = 1;
  idist2     = nx2;
  
  oembed2    = nx2;
  ostride2   = 1;
  odist2     = nx2;

  conf->streams = (cudaStream_t *)malloc(conf->nstream * sizeof(cudaStream_t));
  conf->fft_plans1 = (cufftHandle *)malloc(conf->nstream * sizeof(cufftHandle));
  conf->fft_plans2 = (cufftHandle *)malloc(conf->nstream * sizeof(cufftHandle));
  for(i = 0; i < conf->nstream; i ++)
    {
      CudaSafeCall(cudaStreamCreate(&conf->streams[i]));
      CufftSafeCall(cufftPlanMany(&conf->fft_plans1[i], CUFFT_RANK1, &nx1, &iembed1, istride1, idist1, &oembed1, ostride1, odist1, CUFFT_C2C, batch1));
      CufftSafeCall(cufftPlanMany(&conf->fft_plans2[i], CUFFT_RANK2, &nx2, &iembed2, istride2, idist2, &oembed2, ostride2, odist2, CUFFT_C2C, batch2));
      
      CufftSafeCall(cufftSetStream(conf->fft_plans1[i], conf->streams[i]));
      CufftSafeCall(cufftSetStream(conf->fft_plans2[i], conf->streams[i]));
    }
  
  conf->sbufin_size  = conf->ndata1 * NBYTE_IN;
  conf->sbufout_size = conf->ndata2 * NBYTE_OUT;
  
  conf->bufin_size   = conf->nstream * conf->sbufin_size;
  conf->bufout_size  = conf->nstream * conf->sbufout_size;
  
  conf->sbufrt1_size = conf->npol1 * NBYTE_RT;
  conf->sbufrt2_size = conf->npol2 * NBYTE_RT;
  conf->bufrt1_size  = conf->nstream * conf->sbufrt1_size;
  conf->bufrt2_size  = conf->nstream * conf->sbufrt2_size;
    
  conf->hbufin_offset = conf->sbufin_size / sizeof(char);
  conf->dbufin_offset = conf->sbufin_size / sizeof(int64_t);
  conf->bufrt1_offset = conf->sbufrt1_size / sizeof(cufftComplex);
  conf->bufrt2_offset = conf->sbufrt2_size / sizeof(cufftComplex);
  
  conf->dbufout_offset = conf->sbufout_size / sizeof(int8_t);
  //conf->dbufout_offset = conf->sbufout_size / sizeof(float);
  conf->hbufout_offset = conf->sbufout_size / sizeof(char);
  	  
  CudaSafeCall(cudaMalloc((void **)&conf->dbuf_in, conf->bufin_size));   
  CudaSafeCall(cudaMalloc((void **)&conf->dbuf_out, conf->bufout_size)); 

  CudaSafeCall(cudaMalloc((void **)&conf->ddat_offs, NCHAN_FINAL * sizeof(float)));
  CudaSafeCall(cudaMalloc((void **)&conf->dsquare_mean, NCHAN_FINAL * sizeof(float)));
  CudaSafeCall(cudaMalloc((void **)&conf->ddat_scl, NCHAN_FINAL * sizeof(float)));
  
  CudaSafeCall(cudaMemset((void *)conf->ddat_offs, 0, NCHAN_FINAL * sizeof(float)));   // We have to clear the memory for this parameter
  CudaSafeCall(cudaMemset((void *)conf->dsquare_mean, 0, NCHAN_FINAL * sizeof(float)));// We have to clear the memory for this parameter
  
  CudaSafeCall(cudaMallocHost((void **)&conf->hdat_scl, NCHAN_FINAL * sizeof(float)));   // Malloc device memory to receive data from host
  CudaSafeCall(cudaMallocHost((void **)&conf->hdat_offs, NCHAN_FINAL * sizeof(float)));   // Malloc device memory to receive data from host
  CudaSafeCall(cudaMallocHost((void **)&conf->hsquare_mean, NCHAN_FINAL * sizeof(float)));   // Malloc device memory to receive data from host

  CudaSafeCall(cudaMalloc((void **)&conf->buf_rt1, conf->bufrt1_size));
  CudaSafeCall(cudaMalloc((void **)&conf->buf_rt2, conf->bufrt2_size)); 

  /* Prepare the setup of kernels */
  conf->gridsize_unpack.x = conf->stream_ndf;
  conf->gridsize_unpack.y = NCHK_NIC;
  conf->gridsize_unpack.z = 1;
  conf->blocksize_unpack.x = NSAMP_DF; 
  conf->blocksize_unpack.y = NCHAN_CHK;
  conf->blocksize_unpack.z = 1;
  
  conf->gridsize_swap_select_transpose.x = NCHK_NIC * NCHAN_CHK;
  conf->gridsize_swap_select_transpose.y = conf->stream_ndf * NSAMP_DF / CUFFT_NX1;
  conf->gridsize_swap_select_transpose.z = 1;  
  conf->blocksize_swap_select_transpose.x = CUFFT_NX1;
  conf->blocksize_swap_select_transpose.y = 1;
  conf->blocksize_swap_select_transpose.z = 1;

  conf->gridsize_swap_select_transpose_swap.x = NCHK_NIC * NCHAN_CHK;
  conf->gridsize_swap_select_transpose_swap.y = conf->stream_ndf * NSAMP_DF / CUFFT_NX1;
  conf->gridsize_swap_select_transpose_swap.z = 1;  
  conf->blocksize_swap_select_transpose_swap.x = CUFFT_NX1;
  conf->blocksize_swap_select_transpose_swap.y = 1;
  conf->blocksize_swap_select_transpose_swap.z = 1;
      
  conf->gridsize_swap.x = conf->stream_ndf * NSAMP_DF / CUFFT_NX1; 
  conf->gridsize_swap.y = NCHAN_FINAL;
  conf->gridsize_swap.z = 1;
  conf->blocksize_swap.x = CUFFT_NX2;
  conf->blocksize_swap.y = 1;
  conf->blocksize_swap.z = 1;
  
  conf->gridsize_mean.x = 1; 
  conf->gridsize_mean.y = 1; 
  conf->gridsize_mean.z = 1;
  conf->blocksize_mean.x = NCHAN_FINAL; 
  conf->blocksize_mean.y = 1;
  conf->blocksize_mean.z = 1;
  
  conf->gridsize_scale.x = 1;
  conf->gridsize_scale.y = 1;
  conf->gridsize_scale.z = 1;
  conf->blocksize_scale.x = NCHAN_FINAL;
  conf->blocksize_scale.y = 1;
  conf->blocksize_scale.z = 1;
  
  conf->gridsize_transpose_scale.x = conf->stream_ndf * NSAMP_DF / CUFFT_NX1; 
  conf->gridsize_transpose_scale.y = NCHAN_FINAL;
  conf->gridsize_transpose_scale.z = 1;
  conf->blocksize_transpose_scale.x = CUFFT_NX2;
  conf->blocksize_transpose_scale.y = 1;
  conf->blocksize_transpose_scale.z = 1;
  
  conf->gridsize_transpose_pad.x = conf->stream_ndf * NSAMP_DF / CUFFT_NX1; 
  conf->gridsize_transpose_pad.y = NCHAN_FINAL;
  conf->gridsize_transpose_pad.z = 1;
  conf->blocksize_transpose_pad.x = CUFFT_NX2;
  conf->blocksize_transpose_pad.y = 1;
  conf->blocksize_transpose_pad.z = 1;

  conf->gridsize_sum1.x = NCHAN_FINAL;
  conf->gridsize_sum1.y = conf->stream_ndf * NPOL_SAMP;
  conf->gridsize_sum1.z = 1;
  conf->blocksize_sum1.x = NSAMP_DF / 2;
  conf->blocksize_sum1.y = 1;
  conf->blocksize_sum1.z = 1;
  
  conf->gridsize_sum2.x = NCHAN_FINAL;
  conf->gridsize_sum2.y = 1;
  conf->gridsize_sum2.z = 1;
  conf->blocksize_sum2.x = conf->stream_ndf * NPOL_SAMP / 2;
  conf->blocksize_sum2.y = 1;
  conf->blocksize_sum2.z = 1;
  
  conf->gridsize_transpose_scale2.x = conf->stream_ndf * NSAMP_DF / CUFFT_NX1; 
  conf->gridsize_transpose_scale2.y = 1;
  conf->gridsize_transpose_scale2.z = 1;
  conf->blocksize_transpose_scale2.x = CUFFT_NX2;
  conf->blocksize_transpose_scale2.y = CUFFT_NX2;
  conf->blocksize_transpose_scale2.z = 1;
  
  conf->gridsize_transpose_scale3.x = conf->stream_ndf * NSAMP_DF / CUFFT_NX1; 
  conf->gridsize_transpose_scale3.y = NCHAN_FINAL / TILE_DIM;
  conf->gridsize_transpose_scale3.z = 1;
  conf->blocksize_transpose_scale3.x = TILE_DIM;
  conf->blocksize_transpose_scale3.y = NROWBLOCK_TRANS;
  conf->blocksize_transpose_scale3.z = 1;
  
  conf->gridsize_transpose_scale4.x = conf->stream_ndf * NSAMP_DF / CUFFT_NX1; 
  conf->gridsize_transpose_scale4.y = NCHAN_FINAL / TILE_DIM;
  conf->gridsize_transpose_scale4.z = 1;
  conf->blocksize_transpose_scale4.x = TILE_DIM;
  conf->blocksize_transpose_scale4.y = NROWBLOCK_TRANS;
  conf->blocksize_transpose_scale4.z = 1;
  
  conf->gridsize_transpose_float.x = conf->stream_ndf * NSAMP_DF / CUFFT_NX1; 
  conf->gridsize_transpose_float.y = NCHAN_FINAL / TILE_DIM;
  conf->gridsize_transpose_float.z = 1;
  conf->blocksize_transpose_float.x = TILE_DIM;
  conf->blocksize_transpose_float.y = NROWBLOCK_TRANS;
  conf->blocksize_transpose_float.z = 1;
  
  /* attach to input ring buffer */
  conf->hdu_in = dada_hdu_create(runtime_log);
  dada_hdu_set_key(conf->hdu_in, conf->key_in);
  if(dada_hdu_connect(conf->hdu_in) < 0)
    {
      multilog(runtime_log, LOG_ERR, "could not connect to hdu\n");
      fprintf(stderr, "Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;    
    }  
  db = (ipcbuf_t *) conf->hdu_in->data_block;
  conf->rbufin_size = ipcbuf_get_bufsz(db);  
  if(conf->rbufin_size % conf->bufin_size != 0)  
    {
      multilog(runtime_log, LOG_ERR, "data buffer size mismatch\n");
      fprintf(stderr, "Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;    
    }
  
  /* registers the existing host memory range for use by CUDA */
  dada_cuda_dbregister(conf->hdu_in);
        
  conf->hdrsz = ipcbuf_get_bufsz(conf->hdu_in->header_block);  
  if(conf->hdrsz != DADA_HDR_SIZE)    // This number should match
    {
      multilog(runtime_log, LOG_ERR, "data buffer size mismatch\n");
      fprintf(stderr, "Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;    
    }
  
  /* make ourselves the read client */
  if(dada_hdu_lock_read(conf->hdu_in) < 0)
    {
      multilog(runtime_log, LOG_ERR, "open_hdu: could not lock write\n");
      fprintf(stderr, "Error locking HDU, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }

  /* Prepare output ring buffer */  
  conf->hdu_out = dada_hdu_create(runtime_log);
  dada_hdu_set_key(conf->hdu_out, conf->key_out);
  if(dada_hdu_connect(conf->hdu_out) < 0)
    {
      multilog(runtime_log, LOG_ERR, "could not connect to hdu\n");
      fprintf(stderr, "Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;    
    }
  db = (ipcbuf_t *) conf->hdu_out->data_block;
  conf->rbufout_size = ipcbuf_get_bufsz(db);
  if(conf->rbufout_size % conf->bufout_size != 0)  
    {
      multilog(runtime_log, LOG_ERR, "data buffer size mismatch\n");
      fprintf(stderr, "Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;    
    }
  
  conf->hdrsz = ipcbuf_get_bufsz(conf->hdu_out->header_block);  
  if(conf->hdrsz != DADA_HDR_SIZE)    // This number should match
    {
      multilog(runtime_log, LOG_ERR, "data buffer size mismatch\n");
      fprintf(stderr, "Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;    
    }  
  /* make ourselves the write client */
  if(dada_hdu_lock_write(conf->hdu_out) < 0)
    {
      multilog(runtime_log, LOG_ERR, "open_hdu: could not lock write\n");
      fprintf(stderr, "Error locking HDU, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }

  //if(conf->sod)
  //  {      
  //    if(ipcbuf_enable_sod(db, 0, 0) < 0)  // We start at the beginning
  //	{
  //multilog(runtime_log, LOG_ERR, "Can not write data before start, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  //	  fprintf(stderr, "Can not write data before start, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  //	  return EXIT_FAILURE;
  //	}
  //  }
  //else
  //  {
  //    if(ipcbuf_disable_sod(db) < 0)
  //	{
  //multilog(runtime_log, LOG_ERR, "Can not write data before start, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  //	  fprintf(stderr, "Can not write data before start, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  //	  return EXIT_FAILURE;
  //	}
  //  }
      
  /* Register header */
  if(register_header(conf))
    {
      multilog(runtime_log, LOG_ERR, "header register failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf(stderr, "header register failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  return EXIT_SUCCESS;
}

int do_process(conf_t conf)
{
  CudaSafeCall(cudaSetDevice(conf.device_id));
  
  /*
    The whole procedure is :
    1. Unpack the data and reorder it from TFTFP to PFT order, prepare for the forward FFT;
    2. Forward FFT the PFT data to get finer channelzation and the data is in PFTF order after FFT;
    3. Swap the FFT output to put the frequency centre on the right place, drop frequency channel edge and band edge and put the data into PTF order;
    4. Swap the previous data PTF data and put the centre frequency at bin 0 for each FFT block, prepare for inverse FFT;
    5. Inverse FFT the data to get PTFT order data;
    6. Swap the data to get TFP data and scale it;
  */
  size_t i, j;
  size_t hbufin_offset, dbufin_offset, bufrt1_offset, bufrt2_offset, hbufout_offset, dbufout_offset;
  dim3 gridsize_unpack, blocksize_unpack;
  dim3 gridsize_swap_select_transpose_swap, blocksize_swap_select_transpose_swap;
  dim3 gridsize_transpose_scale3, blocksize_transpose_scale3;
  dim3 gridsize_transpose_scale4, blocksize_transpose_scale4;
  dim3 gridsize_transpose_float, blocksize_transpose_float;
  uint64_t block_id = 0;
  size_t curbufsz;
  
  gridsize_unpack            = conf.gridsize_unpack;
  blocksize_unpack           = conf.blocksize_unpack;
  gridsize_swap_select_transpose_swap  = conf.gridsize_swap_select_transpose_swap;   
  blocksize_swap_select_transpose_swap = conf.blocksize_swap_select_transpose_swap;  
  gridsize_transpose_scale3       = conf.gridsize_transpose_scale3;
  blocksize_transpose_scale3      = conf.blocksize_transpose_scale3; 
  gridsize_transpose_scale4       = conf.gridsize_transpose_scale4;
  blocksize_transpose_scale4      = conf.blocksize_transpose_scale4;
  gridsize_transpose_float        = conf.gridsize_transpose_float;
  blocksize_transpose_float       = conf.blocksize_transpose_float;
    
  /* Get scale of data */
  dat_offs_scl(conf);
#ifdef DEBUG
  for(i = 0; i < NCHAN_FINAL; i++)
    fprintf(stdout, "DAT_OFFS:\t%E\tDAT_SCL:\t%E\n", conf.hdat_offs[i], conf.hdat_scl[i]);
#endif
  
  /* Do the real job */
#ifdef DEBUG
  double elapsed_time;
  struct timespec start, stop;
  clock_gettime(CLOCK_REALTIME, &start);
#endif

  conf.hdu_out->data_block->curbuf = ipcio_open_block_write(conf.hdu_out->data_block, &block_id);   /* Open buffer to write */
  
  while(conf.hdu_in->data_block->curbufsz == conf.rbufin_size)
    // The first time we open a block at the scale calculation, we need to make sure that the input ring buffer block is bigger than the block needed for scale calculation
    // Otherwise we have to open couple of blocks to calculate scales and these blocks will dropped after that
    {
      //for(i = 0; i < conf.rbufin_size; i += conf.bufin_size)
      for(i = 0; i < conf.nrun_blk; i ++)
	{
#ifdef DEBUG
	  clock_gettime(CLOCK_REALTIME, &stop);
	  elapsed_time = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1000000000.0L;
	  fprintf(stdout, "elapsed time to read %d data frame steps is %f s\n", conf.stream_ndf * conf.nstream, elapsed_time);
#endif
	  
#ifdef DEBUG
	  float elapsed_event;
	  cudaEvent_t start_event, stop_event;
	  CudaSafeCall(cudaEventCreate(&start_event));
	  CudaSafeCall(cudaEventCreate(&stop_event));
	  CudaSafeCall(cudaEventRecord(start_event));
#endif
	  
	  for(j = 0; j < conf.nstream; j++)
	    {
	      hbufin_offset = j * conf.hbufin_offset + i * conf.bufin_size;
	      dbufin_offset = j * conf.dbufin_offset; 
	      bufrt1_offset = j * conf.bufrt1_offset;
	      bufrt2_offset = j * conf.bufrt2_offset;
	      
	      dbufout_offset = j * conf.dbufout_offset;
	      hbufout_offset = j * conf.hbufout_offset + i * conf.bufout_size;
	      
	      /* Copy data into device */
#ifdef DEBUG
	      clock_gettime(CLOCK_REALTIME, &start);
#endif 
	      CudaSafeCall(cudaMemcpyAsync(&conf.dbuf_in[dbufin_offset], &conf.hdu_in->data_block->curbuf[hbufin_offset], conf.sbufin_size, cudaMemcpyHostToDevice, conf.streams[j]));
	      
#ifdef DEBUG
	      clock_gettime(CLOCK_REALTIME, &stop);
	      elapsed_time = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1000000000.0L;
	      fprintf(stdout, "elapsed time to memcpy %d data frame steps is %f s\n", conf.stream_ndf, elapsed_time);
#endif
	      
	      /* Unpack raw data into cufftComplex array */
	      unpack_kernel<<<gridsize_unpack, blocksize_unpack, 0, conf.streams[j]>>>(&conf.dbuf_in[dbufin_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp1);
	      
	      /* Do forward FFT */
	      CufftSafeCall(cufftExecC2C(conf.fft_plans1[j], &conf.buf_rt1[bufrt1_offset], &conf.buf_rt1[bufrt1_offset], CUFFT_FORWARD));
	      
	      /* Prepare for inverse FFT */
	      swap_select_transpose_swap_kernel<<<gridsize_swap_select_transpose_swap, blocksize_swap_select_transpose_swap, 0, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.buf_rt2[bufrt2_offset], conf.nsamp1, conf.nsamp2); // This kernle is the combination of previous kernels	  
	      
	      /* Do inverse FFT */
	      CufftSafeCall(cufftExecC2C(conf.fft_plans2[j], &conf.buf_rt2[bufrt2_offset], &conf.buf_rt2[bufrt2_offset], CUFFT_INVERSE));
	      
	      /* Get final output */
	      //transpose_scale_kernel3<<<gridsize_transpose_scale3, blocksize_transpose_scale3, 0, conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.dbuf_out[dbufout_offset], conf.nsamp2, conf.scale);  
	      transpose_scale_kernel4<<<gridsize_transpose_scale4, blocksize_transpose_scale4, 0, conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.dbuf_out[dbufout_offset], conf.nsamp2, conf.ddat_offs, conf.ddat_scl);   
	      //transpose_float_kernel<<<gridsize_transpose_float, blocksize_transpose_float, 0, conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.dbuf_out[dbufout_offset], conf.nsamp2);   
	      
	      /* Copy the final output to host */
	      CudaSafeCall(cudaMemcpyAsync(&conf.hdu_out->data_block->curbuf[hbufout_offset], &conf.dbuf_out[dbufout_offset], conf.sbufout_size, cudaMemcpyDeviceToHost, conf.streams[j]));
	      
	      //CudaSafeCall(cudaStreamSynchronize(conf.streams[j])); // Sync here equal to single stream
	    }
	  CudaSynchronizeCall(); // Sync here is for multiple streams
	  
#ifdef DEBUG
	  CudaSafeCall(cudaEventRecord(stop_event));
	  CudaSafeCall(cudaEventSynchronize(stop_event));
	  CudaSafeCall(cudaEventElapsedTime(&elapsed_event, start_event, stop_event));
	  fprintf(stdout, "elapsed time for GPU process of %d data frame steps is %f s\n", conf.stream_ndf * conf.nstream, elapsed_event/1.0E3);
#endif
	}
      
#ifdef DEBUG
      clock_gettime(CLOCK_REALTIME, &start);
#endif      
	  
      /* Close current buffer */
      if(ipcio_close_block_write(conf.hdu_out->data_block, conf.rbufout_size) < 0)
	{
	  multilog (runtime_log, LOG_ERR, "close_buffer: ipcio_close_block_write failed\n");
	  fprintf(stderr, "close_buffer: ipcio_close_block_write failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  return EXIT_FAILURE;
	}
      conf.hdu_out->data_block->curbuf = ipcio_open_block_write(conf.hdu_out->data_block, &block_id);   /* Open buffer to write */
	  
#ifdef DEBUG
      clock_gettime(CLOCK_REALTIME, &stop);
      elapsed_time = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1000000000.0L;
      fprintf(stdout, "elapsed time to write %d data frame steps is %f s\n\n", conf.rbufin_ndf , elapsed_time);
#endif
#ifdef DEBUG
      clock_gettime(CLOCK_REALTIME, &start);
#endif
      
      ipcio_close_block_read(conf.hdu_in->data_block, conf.hdu_in->data_block->curbufsz);
      conf.hdu_in->data_block->curbuf = ipcio_open_block_read(conf.hdu_in->data_block, &curbufsz, &block_id);
    }

  ipcio_close_block_read(conf.hdu_in->data_block, conf.hdu_in->data_block->curbufsz);
  
  if (ipcio_close_block_write(conf.hdu_out->data_block, conf.rbufout_size) < 0)
    {
      multilog (runtime_log, LOG_ERR, "close_buffer: ipcio_close_block_write failed\n");
      fprintf(stderr, "close_buffer: ipcio_close_block_write failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }

  return EXIT_SUCCESS;
}

int dat_offs_scl(conf_t conf)
{
  CudaSafeCall(cudaSetDevice(conf.device_id));
  
  size_t i, j;
  dim3 gridsize_unpack, blocksize_unpack;
  dim3 gridsize_swap_select_transpose_swap, blocksize_swap_select_transpose_swap;
  dim3 gridsize_scale, blocksize_scale;  
  dim3 gridsize_mean, blocksize_mean;
  dim3 gridsize_sum1, blocksize_sum1;
  dim3 gridsize_sum2, blocksize_sum2;
  dim3 gridsize_transpose_pad, blocksize_transpose_pad;
  size_t hbufin_offset, dbufin_offset, bufrt1_offset, bufrt2_offset;
  size_t curbufsz, block_id;
  char fname[MSTR_LEN];
  FILE *fp=NULL;
    
  gridsize_unpack                      = conf.gridsize_unpack;
  blocksize_unpack                     = conf.blocksize_unpack;
  gridsize_swap_select_transpose_swap  = conf.gridsize_swap_select_transpose_swap;   
  blocksize_swap_select_transpose_swap = conf.blocksize_swap_select_transpose_swap; 
  gridsize_transpose_pad               = conf.gridsize_transpose_pad;
  blocksize_transpose_pad              = conf.blocksize_transpose_pad;
  	         	               						       
  gridsize_sum1                        = conf.gridsize_sum1;	       
  blocksize_sum1                       = conf.blocksize_sum1;
  gridsize_sum2                        = conf.gridsize_sum2;	       
  blocksize_sum2                       = conf.blocksize_sum2;
  gridsize_scale                       = conf.gridsize_scale;	       
  blocksize_scale                      = conf.blocksize_scale;	         							       
  gridsize_mean                        = conf.gridsize_mean;	       
  blocksize_mean                       = conf.blocksize_mean;

  conf.hdu_in->data_block->curbuf = ipcio_open_block_read(conf.hdu_in->data_block, &curbufsz, &block_id);
  if(conf.hdu_in->data_block->curbuf == NULL)
    {
      multilog (runtime_log, LOG_ERR, "Can not get buffer block from input ring buffer, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf(stderr, "Can not get buffer block from input ring buffer, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
#ifdef DEBUG
  float elapsed_event;
  cudaEvent_t start_event, stop_event;
  CudaSafeCall(cudaEventCreate(&start_event));
  CudaSafeCall(cudaEventCreate(&stop_event));
  CudaSafeCall(cudaEventRecord(start_event));
#endif
  
  for(i = 0; i < conf.rbufin_size; i += conf.bufin_size)
    {
      for (j = 0; j < conf.nstream; j++)
	{
	  hbufin_offset = j * conf.hbufin_offset + i;
	  dbufin_offset = j * conf.dbufin_offset; 
	  bufrt1_offset = j * conf.bufrt1_offset;
	  bufrt2_offset = j * conf.bufrt2_offset;
	  
	  /* Copy data into device */
	  CudaSafeCall(cudaMemcpyAsync(&conf.dbuf_in[dbufin_offset], &conf.hdu_in->data_block->curbuf[hbufin_offset], conf.sbufin_size, cudaMemcpyHostToDevice, conf.streams[j]));

	  /* Unpack raw data into cufftComplex array */
	  unpack_kernel<<<gridsize_unpack, blocksize_unpack, 0, conf.streams[j]>>>(&conf.dbuf_in[dbufin_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp1);

	  /* Do forward FFT */
	  CufftSafeCall(cufftExecC2C(conf.fft_plans1[j], &conf.buf_rt1[bufrt1_offset], &conf.buf_rt1[bufrt1_offset], CUFFT_FORWARD));

	  /* Prepare for inverse FFT */
	  swap_select_transpose_swap_kernel<<<gridsize_swap_select_transpose_swap, blocksize_swap_select_transpose_swap, 0, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.buf_rt2[bufrt2_offset], conf.nsamp1, conf.nsamp2); 

	  /* Do inverse FFT */
	  CufftSafeCall(cufftExecC2C(conf.fft_plans2[j], &conf.buf_rt2[bufrt2_offset], &conf.buf_rt2[bufrt2_offset], CUFFT_INVERSE));

	  /* Transpose the data from PTFT to FTP for later calculation */
	  transpose_pad_kernel<<<gridsize_transpose_pad, blocksize_transpose_pad, 0, conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], conf.nsamp2, &conf.buf_rt1[bufrt1_offset]);

	  /* Get the sum of samples and square of samples */
	  sum_kernel<<<gridsize_sum1, blocksize_sum1, blocksize_sum1.x * sizeof(cufftComplex), conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.buf_rt2[bufrt2_offset]);
	  sum_kernel<<<gridsize_sum2, blocksize_sum2, blocksize_sum2.x * sizeof(cufftComplex), conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.buf_rt1[bufrt1_offset]);

	  //CudaSafeCall(cudaStreamSynchronize(conf.streams[j])); // Sync here equal to single stream
	}
      CudaSynchronizeCall(); // Sync here is for multiple streams

      mean_kernel<<<gridsize_mean, blocksize_mean>>>(conf.buf_rt1, conf.bufrt1_offset, conf.ddat_offs, conf.dsquare_mean, conf.nstream, conf.scl_ndim);
    }
  
  /* Get the scale of each chanel */
  scale_kernel<<<gridsize_scale, blocksize_scale>>>(conf.ddat_offs, conf.dsquare_mean, conf.ddat_scl);
  CudaSynchronizeCall();
  
#ifdef DEBUG
  CudaSafeCall(cudaEventRecord(stop_event));
  CudaSafeCall(cudaEventSynchronize(stop_event));
  CudaSafeCall(cudaEventElapsedTime(&elapsed_event, start_event, stop_event));
  fprintf(stdout, "elapsed time to get scale with %.0f data is %f s\n", conf.scl_ndim, elapsed_event/1.0E3);
#endif
  
  CudaSafeCall(cudaMemcpy(conf.hdat_offs, conf.ddat_offs, sizeof(float) * NCHAN_FINAL, cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(conf.hdat_scl, conf.ddat_scl, sizeof(float) * NCHAN_FINAL, cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(conf.hsquare_mean, conf.dsquare_mean, sizeof(float) * NCHAN_FINAL, cudaMemcpyDeviceToHost));
  
#ifdef DEBUG
  for (i = 0; i< NCHAN_FINAL; i++)
    fprintf(stdout, "DAT_OFFS:\t%E\tDAT_SCL:\t%E\n", conf.hdat_offs[i], conf.hdat_scl[i]);
#endif
  /* Record scale into file */
  sprintf(fname, "%s/%s_scale.txt", conf.dir, conf.utc_start);
  fp = fopen(fname, "w");
  if(fp == NULL)
    {
      multilog (runtime_log, LOG_ERR, "Can not open scale file, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf(stderr, "Can not open scale file, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  for (i = 0; i< NCHAN_FINAL; i++)
    fprintf(fp, "%E\t%E\n", conf.hdat_offs[i], conf.hdat_scl[i]);
  
  fclose(fp);
  return EXIT_SUCCESS;
}

int destroy_process(conf_t conf)
{
  int i;
  CudaSafeCall(cudaSetDevice(conf.device_id));
  
  for (i = 0; i < conf.nstream; i++)
    {
      CudaSafeCall(cudaStreamDestroy(conf.streams[i]));
      CufftSafeCall(cufftDestroy(conf.fft_plans1[i]));
      CufftSafeCall(cufftDestroy(conf.fft_plans2[i]));
    }
  
  cudaFree(conf.dbuf_in);
  cudaFree(conf.dbuf_out);

  cudaFreeHost(conf.hdat_offs);
  cudaFreeHost(conf.hsquare_mean);
  cudaFreeHost(conf.hdat_scl);

  cudaFree(conf.ddat_offs);
  cudaFree(conf.dsquare_mean);
  cudaFree(conf.ddat_scl);
  
  cudaFree(conf.buf_rt1);
  cudaFree(conf.buf_rt2);

  dada_cuda_dbunregister(conf.hdu_in);
  
  dada_hdu_unlock_write(conf.hdu_out);
  dada_hdu_disconnect(conf.hdu_out);
  dada_hdu_destroy(conf.hdu_out);

  dada_hdu_unlock_read(conf.hdu_in);
  dada_hdu_disconnect(conf.hdu_in);
  dada_hdu_destroy(conf.hdu_in);

  free(conf.streams);
  free(conf.fft_plans1);
  free(conf.fft_plans2);
  
  return EXIT_SUCCESS;
}

int register_header(conf_t *conf)
{
  size_t hdrsz;
  double freq;
  
  conf->hdrbuf_in  = ipcbuf_get_next_read(conf->hdu_in->header_block, &hdrsz);  
  if (!conf->hdrbuf_in)
    {
      multilog(runtime_log, LOG_ERR, "get next header block error.\n");
      fprintf(stderr, "Error getting header_buf, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  if(hdrsz != DADA_HDR_SIZE)
    {
      multilog(runtime_log, LOG_ERR, "get next header block error.\n");
      fprintf(stderr, "Header size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }

  conf->hdrbuf_out = ipcbuf_get_next_write(conf->hdu_out->header_block);
  if (!conf->hdrbuf_out)
    {
      multilog(runtime_log, LOG_ERR, "get next header block error.\n");
      fprintf(stderr, "Error getting header_buf, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  if(conf->stream)
    {
      if (fileread(conf->hfname, conf->hdrbuf_out, DADA_HDR_SIZE) < 0)
	{
	  multilog(runtime_log, LOG_ERR, "cannot read header from %s\n", conf->hfname);
	  fprintf(stderr, "Error reading header file, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  return EXIT_FAILURE;
	}  
      
      /* Pass utc_start from hdrin to hdrout */
      if (ascii_header_get(conf->hdrbuf_in, "UTC_START", "%s", conf->utc_start) < 0)  
	{
	  multilog(runtime_log, LOG_ERR, "failed ascii_header_get UTC_START\n");
	  fprintf(stderr, "Error getting UTC_START, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  return EXIT_FAILURE;
	}
      fprintf(stdout, "\nGet UTC_START at process stage:\t\t%s\n", conf->utc_start);
      if (ascii_header_set(conf->hdrbuf_out, "UTC_START", "%s", conf->utc_start) < 0)  
	{
	  multilog(runtime_log, LOG_ERR, "failed ascii_header_get UTC_START\n");
	  fprintf(stderr, "Error setting UTC_START, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  return EXIT_FAILURE;
	}
      fprintf(stdout, "Set UTC_START at process stage:\t\t%s\n", conf->utc_start);
      multilog(runtime_log, LOG_INFO, "UTC_START:\t%s\n", conf->utc_start);
      
      /* Pass picoseconds from hdrin to hdrout */
      if (ascii_header_get(conf->hdrbuf_in, "PICOSECONDS", "%"PRIu64, &(conf->picoseconds)) < 0)  
	{
	  multilog(runtime_log, LOG_ERR, "failed ascii_header_get PICOSECONDS\n");
	  fprintf(stderr, "Error getting PICOSECONDS, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  return EXIT_FAILURE;
	}
      fprintf(stdout, "Get PICOSECONDS at process stage:\t%"PRIu64"\n", conf->picoseconds);
      if (ascii_header_set(conf->hdrbuf_out, "PICOSECONDS", "%"PRIu64, conf->picoseconds) < 0)  
	{
	  multilog(runtime_log, LOG_ERR, "failed ascii_header_get PICOSECONDS\n");
	  fprintf(stderr, "Error setting PICOSECONDS, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  return EXIT_FAILURE;
	}
      fprintf(stdout, "Set PICOSECONDS at process stage:\t%"PRIu64"\n\n", conf->picoseconds);
      multilog(runtime_log, LOG_INFO, "PICOSECONDS:\t%"PRIu64"\n", conf->picoseconds);
      
      /* Pass frequency from hdrin to hdrout */
      if (ascii_header_get(conf->hdrbuf_in, "FREQ", "%lf", &freq) < 0)   // RA and DEC also need to pass from hdrin to hdrout
	{
	  multilog(runtime_log, LOG_ERR, "failed ascii_header_get FREQ\n");
	  fprintf(stderr, "Error getting FREQ, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  return EXIT_FAILURE;
	}
      if (ascii_header_set(conf->hdrbuf_out, "FREQ", "%.1lf", freq) < 0)  
	{
	  multilog(runtime_log, LOG_ERR, "failed ascii_header_get FREQ\n");
	  fprintf(stderr, "Error setting FREQ, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  return EXIT_FAILURE;
	}
    }
  else
    {
      memcpy(conf->hdrbuf_out, conf->hdrbuf_in, DADA_HDR_SIZE);
      if (ascii_header_get(conf->hdrbuf_in, "UTC_START", "%s", conf->utc_start) < 0)  
	{
	  multilog(runtime_log, LOG_ERR, "failed ascii_header_get UTC_START\n");
	  fprintf(stderr, "Error getting UTC_START, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  return EXIT_FAILURE;
	}
    }  
  
  if(ipcbuf_mark_cleared (conf->hdu_in->header_block))  // We are the only one reader, so that we can clear it after read;
    {
      multilog(runtime_log, LOG_ERR, "Could not clear header block\n");
      fprintf(stderr, "Error header_clear, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }

  /* donot set header parameters anymore - acqn. doesn't start */
  if (ipcbuf_mark_filled (conf->hdu_out->header_block, conf->hdrsz) < 0)
    {
      multilog(runtime_log, LOG_ERR, "Could not mark filled header block\n");
      fprintf(stderr, "Error header_fill, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
 
  return EXIT_SUCCESS;
}