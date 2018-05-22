#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <stdbool.h>
#include <sched.h>
#include <math.h>

#include "ipcbuf.h"
#include "sync.h"
#include "capture.h"

extern char *cbuf;
extern pthread_mutex_t hdr_ref_mutex[MPORT_NIC];
//extern pthread_mutex_t force_switch_mutex;

extern int transit[MPORT_NIC];
extern uint64_t tail[MPORT_NIC];
extern int force_switch;
extern char *tbuf;
extern int finish[MPORT_NIC];
extern hdr_t hdr_ref[MPORT_NIC];
extern int quit;
extern multilog_t *runtime_log;

int threads(conf_t *conf)
{
  int i, ret[MPORT_NIC + 1], node;
  pthread_t thread[MPORT_NIC + 1];
  pthread_attr_t attr;
  cpu_set_t cpus;
  unsigned char *ip = (unsigned char *)&conf->sock[0].sa.sin_addr.s_addr;  // The way to bind thread into different NUMA node;
  node = (int)ip[3] - 1;   // Count from zero;
  ip = NULL;
  int active_ports = conf->active_ports;

#ifdef DEBUG
  fprintf(stdout, "*********************************************\n");
  fprintf(stdout, "WE ARE ON CAPTURE PART...\n");
  fprintf(stdout, "*********************************************\n");
#endif
  
  for(i = 0; i < active_ports; i++)   // Create threads
    {
      pthread_attr_init(&attr);  
      CPU_ZERO(&cpus);
      CPU_SET(i + node * NCPU_NUMA, &cpus);
      
      pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);	
      ret[i] = pthread_create(&thread[i], &attr, capture_thread, (void *)conf);
      //ret[i] = pthread_create(&thread[i], NULL, capture_thread, (void *)conf);
      
      pthread_attr_destroy(&attr);
    }
  
  pthread_attr_init(&attr);  
  CPU_ZERO(&cpus);
  CPU_SET(active_ports + node * NCPU_NUMA, &cpus);
  
  pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);	
  ret[active_ports] = pthread_create(&thread[active_ports], &attr, sync_thread, (void *)conf);
  //ret[active_ports] = pthread_create(&thread[active_ports], NULL, sync_thread, (void *)conf);
  pthread_attr_destroy(&attr);
  
  for(i = 0; i < active_ports + 1; i++)   // Join threads and unbind cpus
    pthread_join(thread[i], NULL);

  return EXIT_SUCCESS;
}

void *sync_thread(void *conf)
{
  conf_t *captureconf = (conf_t *)conf;
  int i, active_chunks = captureconf->active_chunks, ntransit, nfinish;
  struct timespec start, stop;// sleep_time;
  uint64_t cbuf_loc, tbuf_loc, ntail;
  int ifreq, idf;
  uint64_t block_id = 0;
  
  //sleep_time.tv_sec  = 0;
  //sleep_time.tv_nsec = SLP_NS; 
  
  while(true)
    {
      ntransit = 0; 
      for(i = 0; i < captureconf->active_ports; i++)
	ntransit += transit[i];
      
      /* To see if we need to move to next buffer block */
      if((ntransit > active_chunks) || force_switch)                   // Once we have more than active_links data frames on temp buffer, we will move to new ring buffer block
	{
#ifdef DEBUG
	  clock_gettime(CLOCK_REALTIME, &start);
#endif	  
	  /* Close current buffer */
	  if(ipcio_close_block_write(captureconf->hdu->data_block, captureconf->rbufsz) < 0)
	    //if(ipcbuf_mark_filled ((ipcbuf_t*)captureconf->hdu->data_block, captureconf->rbufsz) < 0)
	    {
	      multilog (runtime_log, LOG_ERR, "close_buffer: ipcio_close_block_write failed\n");
	      fprintf(stderr, "close_buffer: ipcio_close_block_write failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	      return NULL;
	    }

	  cbuf = ipcio_open_block_write(captureconf->hdu->data_block, &block_id);
	  //cbuf = ipcbuf_get_next_write ((ipcbuf_t*)captureconf->hdu->data_block);
	  
#ifdef DEBUG
	  clock_gettime(CLOCK_REALTIME, &stop);
#endif
	  for(i = 0; i < captureconf->active_ports; i++)
	    {
	      // Update the reference hdr, once capture thread get the updated reference, the data will go to the next block;
	      // We have to put a lock here as partial update of reference hdr will be a trouble to other threads;
	      pthread_mutex_lock(&hdr_ref_mutex[i]);
	      hdr_ref[i].idf += captureconf->rbuf_ndf;
	      if(hdr_ref[i].idf >= NDF_PRD)                       // Here I assume that we could not lose one period;
		{
		  hdr_ref[i].sec += PRD_SEC;
		  hdr_ref[i].idf -= NDF_PRD;
		}
	      pthread_mutex_unlock(&hdr_ref_mutex[i]);
	    }
	  
#ifdef DEBUG
	  clock_gettime(CLOCK_REALTIME, &start);
#endif
	  while(true) // Wait until all threads are on new buffer block
	    {
	      ntransit = 0;
	      for(i = 0; i < captureconf->active_ports; i++)
		ntransit += transit[i];
	      if(ntransit == 0)
		break;
	    }
	  
	  /* To see if we need to copy data from temp buffer into ring buffer */
	  ntail = 0;
	  for(i = 0; i < captureconf->active_ports; i++)
	    ntail = (tail[i] > ntail) ? tail[i] : ntail;
	  
#ifdef DEBUG
	  fprintf(stdout, "Temp copy:\t%"PRIu64" positions need to be checked.\n", ntail);
#endif
	  
	  for(i = 0; i < ntail; i++)
	    {
	      tbuf_loc = (uint64_t)(i * (captureconf->pkt_size + 1));	      
	      if(tbuf[tbuf_loc] == 'Y')
		{		  
		  //idf = (int)(i / NCHK_NIC);
		  //ifreq = i - idf * NCHK_NIC;
		  cbuf_loc = (uint64_t)(i * captureconf->pkt_size);  // This is for the TFTFP order temp buffer copy;
		  //cbuf_loc = (uint64_t)(ifreq * RBUF_NDF + idf) * captureconf->pkt_size;  // This is for the FTFP order temp buffer copy;
		
		  memcpy(cbuf + cbuf_loc, tbuf + tbuf_loc + 1, captureconf->pkt_size);
		  
		  tbuf[tbuf_loc + 1] = 'N';  // Make sure that we do not copy the data later;
		  // If we do not do that, we may have too many data frames to copy later
		}
	    }
#ifdef DEBUG
	  clock_gettime(CLOCK_REALTIME, &stop);
#endif	  
	  for(i = 0; i < MPORT_NIC; i++)
	    tail[i] = 0;  // Reset the location of tbuf;

	  //pthread_mutex_lock(&force_switch_mutex);
	  force_switch = 0;
	  //pthread_mutex_unlock(&force_switch_mutex);
	}

      /* To see if we need to stop */
      nfinish = 0;
      for(i = 0; i < captureconf->active_ports; i++)
	nfinish += finish[i];
      if(nfinish == captureconf->active_ports)
	{
	  //if(ipcbuf_mark_filled ((ipcbuf_t*)captureconf->hdu->data_block, captureconf->rbufsz) < 0)
	  if(ipcio_close_block_write (captureconf->hdu->data_block, captureconf->rbufsz) < 0)  // This should enable eod at current buffer
	    {
	      multilog (runtime_log, LOG_ERR, "close_buffer: ipcio_close_block_write failed\n");
	      fprintf(stderr, "close_buffer: ipcio_close_block_write failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	      return NULL;
	    }
	  pthread_exit(NULL);
	  return NULL;
	}
      if(quit == 1)
	{
	  //if(ipcbuf_mark_filled ((ipcbuf_t*)captureconf->hdu->data_block, captureconf->rbufsz) < 0)
	  if (ipcio_close_block_write (captureconf->hdu->data_block, captureconf->rbufsz) < 0) // This should enable eod at current buffer
	    {
	      multilog (runtime_log, LOG_ERR, "close_buffer: ipcio_close_block_write failed\n");
	      fprintf(stderr, "close_buffer: ipcio_close_block_write failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	      return NULL;
	    }
	  pthread_exit(NULL);
	  return NULL;
	}
      //nanosleep(&sleep_time, NULL);     // Sleep for a while to save the usage of CPU    
    }
  
  /* Exit */
  //if(ipcbuf_mark_filled ((ipcbuf_t*)captureconf->hdu->data_block, captureconf->rbufsz) < 0)
  if (ipcio_close_block_write (captureconf->hdu->data_block, captureconf->rbufsz) < 0)  // This should enable eod at current buffer
    {
      multilog (runtime_log, LOG_ERR, "close_buffer: ipcio_close_block_write failed\n");
      fprintf(stderr, "close_buffer: ipcio_close_block_write failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return NULL;
    }
  
  pthread_exit(NULL);
  return NULL;
}
