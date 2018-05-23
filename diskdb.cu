#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdlib.h>
#include <string.h>

#include "paf_diskdb.cuh"
#include "diskdb.cuh"
#include "cudautil.cuh"

int init_diskdb(conf_t *conf)
{
  ipcbuf_t *db = NULL;
  
  conf->fp = NULL;
  conf->fp = fopen(conf->fname, "r");
  if(conf->fp == NULL)
    {
      fprintf(stderr, "Can not open file: %s, which happens at \"%s\", line [%d].\n", conf->fname, __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  conf->hdu = dada_hdu_create(conf->log);
  dada_hdu_set_key(conf->hdu, conf->key);

  if (dada_hdu_connect(conf->hdu) < 0)
    {
      multilog(conf->log, LOG_ERR, "could not connect to hdu\n");
      fprintf (stderr, "Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;    
    }
  db = (ipcbuf_t *) conf->hdu->data_block;
  conf->rbufsz = ipcbuf_get_bufsz(db);
    
  conf->hdrsz = ipcbuf_get_bufsz(conf->hdu->header_block);  
  if(conf->hdrsz != DADA_HDR_SIZE)    // This number should match
    {
      multilog(conf->log, LOG_ERR, "data buffer size mismatch\n");
      fprintf (stderr, "Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;    
    }
  
  /* make ourselves the write client */
  if (dada_hdu_lock_write (conf->hdu) < 0)
    {
      multilog (conf->log, LOG_ERR, "open_hdu: could not lock write\n");
      fprintf(stderr, "Error locking HDU, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }

  if(conf->sod)
    {      
      if(ipcbuf_enable_sod(db, 0, 0) < 0)  // We start at the beginning
  	{
  	  fprintf(stderr, "Can not write data before start, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  	  return EXIT_FAILURE;
  	}
    }
  else
    {
      if(ipcbuf_disable_sod(db) < 0)
  	{
  	  fprintf(stderr, "Can not write data before start, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  	  return EXIT_FAILURE;
  	}
    }
  
  fseek(conf->fp, DADA_HDR_SIZE, SEEK_SET);
    
  return EXIT_SUCCESS;
}

int do_diskdb(conf_t conf)
{
  char *hdrbuf, *curbuf;
  uint64_t block_id;
  
  hdrbuf = ipcbuf_get_next_write(conf.hdu->header_block);
  if (fileread(conf.hfname, hdrbuf, DADA_HDR_SIZE) < 0)
    {
      multilog(conf.log, LOG_ERR, "cannot read header from %s\n", conf.hfname);
      fprintf(stderr, "Error reading header file, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }

  /* donot set header parameters anymore - acqn. doesn't start */
  if (ipcbuf_mark_filled (conf.hdu->header_block, DADA_HDR_SIZE) < 0)
    {
      multilog(conf.log, LOG_ERR, "Could not mark filled header block\n");
      fprintf(stderr, "Error header_fill, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }

#ifdef DEBUG
  double elapsed_time;
  struct timespec start, stop;
  clock_gettime(CLOCK_REALTIME, &start);
#endif

  int index = 0;
  size_t rbufsz;
  while(!feof(conf.fp))
    {
      curbuf = ipcio_open_block_write (conf.hdu->data_block, &block_id);
      
      rbufsz = fread(curbuf, sizeof(char), conf.rbufsz, conf.fp);
      
      ipcio_close_block_write (conf.hdu->data_block, rbufsz);
      
#ifdef DEBUG
      clock_gettime(CLOCK_REALTIME, &stop);
      elapsed_time = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1000000000.0L;
      fprintf(stdout, "elapsed time to read %zu  bytes is %f s\n", conf.rbufsz, elapsed_time);
#endif
      
#ifdef DEBUG
      clock_gettime(CLOCK_REALTIME, &start);
#endif
      index++;
    }

  return EXIT_SUCCESS;
}

int destroy_diskdb(conf_t conf)
{  
  dada_hdu_unlock_write(conf.hdu);
  dada_hdu_disconnect(conf.hdu);
  dada_hdu_destroy(conf.hdu);
  fclose(conf.fp);
  
  return EXIT_SUCCESS;
}