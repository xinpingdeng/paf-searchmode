#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <pthread.h>
#include <arpa/inet.h>
#include <sys/socket.h>

#include "ipcbuf.h"
#include "capture.h"
#include "multilog.h"
#include "paf_capture.h"

/* 
   The code drops much data frames, two options could be done:
   1. REUSEPORT, does not help;
   2. PACKET socket with ring buffer;

   The problem may from force_switch, not GPU node or capture code, because:
   1. The links with force_switch forwarding is worse;
   2. Only catpure with one beam is okay, but two beams will crush the capture;
   3. The CPU useage is low;
*/

char *cbuf = NULL;
int transit[MPORT_NIC];
int finish[MPORT_NIC];
uint64_t tail[MPORT_NIC];
hdr_t hdr_ref[MPORT_NIC];

int quit;
int force_switch;
int ithread_extern;

extern multilog_t *runtime_log;

char *tbuf = NULL;

pthread_mutex_t ithread_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutexattr_t ithread_mutex_attr;

pthread_mutex_t quit_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutexattr_t quit_mutex_attr;

pthread_mutex_t force_switch_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutexattr_t force_switch_mutex_attr;

pthread_mutex_t hdr_ref_mutex[MPORT_NIC] = {PTHREAD_MUTEX_INITIALIZER};
pthread_mutexattr_t hdr_ref_mutex_attr[MPORT_NIC];

int check_connection(sock_t *sock, int *active_ports, int *active_chunks)
{
  int i, j, k, duplicate, chunk_index;
  double freq;
  char df[DF_SIZE];

#ifdef DEBUG
  fprintf(stdout, "*********************************************\n");
  fprintf(stdout, "WE ARE ON CHECK_CONNECTION PART...\n");
  fprintf(stdout, "*********************************************\n");
#endif
  
  (*active_ports) = 0;
  (*active_chunks) = 0;

  for(i = 0; i < NPORT_NIC; i++)   // Create threads
    {
      /* Default have zero chunks per port */
      chunk_index = 0;
      
      /* The port is default to be active */
      sock[i].active = 1;
      (*active_ports) ++;
      
      /* Set default frequency value to zero */
      for (j = 0; j < MCHK_PORT; j++)
	sock[i].freq[j] = 0.0;

      /* Capture NDF_CHECK data frames, check the port and get available frequency chunks */
      for (j = 0; j < NDF_CHECK; j++)
	{
	  if (recv(sock[i].sock, (void *)df, DF_SIZE, 0) < 0)
	    // If the size is -1, means we get time out and the port is not active
	    {
	      multilog(runtime_log, LOG_ERR, "Can not receive data from %s:%d, which happens at \"%s\", line [%d].\n", inet_ntoa(sock[i].sa.sin_addr), ntohs(sock[i].sa.sin_port), __FILE__, __LINE__);
	      fprintf (stderr, "Can not receive data from %s:%d, which happens at \"%s\", line [%d].\n", inet_ntoa(sock[i].sa.sin_addr), ntohs(sock[i].sa.sin_port), __FILE__, __LINE__);
	      sock[i].active = 0;
	      (*active_ports)--;
	      close(sock[i].sock);
	      return EXIT_FAILURE;
	    }
	  else               // If the port is active, we check the available frequency chunks
	    {
	      freq = hdr_freq(df);
	      
	      if(freq==0)    // Frequency can not be zero
		{
		  multilog(runtime_log, LOG_ERR, "The data received on %s:%d is not right, which happens at \"%s\", line [%d].\n", inet_ntoa(sock[i].sa.sin_addr), ntohs(sock[i].sa.sin_port), __FILE__, __LINE__);
		  fprintf (stderr, "The data received on %s:%d is not right, which happens at \"%s\", line [%d].\n", inet_ntoa(sock[i].sa.sin_addr), ntohs(sock[i].sa.sin_port), __FILE__, __LINE__);
		  return EXIT_FAILURE;
		}

	      if(j == 0) // The frequency in the first data frame must to be recorded 
		{
		  sock[i].freq[chunk_index] = freq;
		  chunk_index ++ ;
		}
	      else      // Record the frequency in later data frames if it is not a duplicate of any previous frequencies 
		{
		  duplicate = 0; // Default is that the new frequency is not a duplicate of previous frequencies
		  for (k = 0; k < chunk_index; k++)
		    if(freq == sock[i].freq[k])
		      duplicate = 1;
		  if (duplicate == 0)
		    {
		      sock[i].freq[chunk_index] = freq;
		      chunk_index ++ ;
		    }
		}
	    }
	}

      sock[i].chunks = chunk_index; // Update the number of available chunks to its real value
      (*active_chunks) += chunk_index;
#ifdef DEBUG
      fprintf(stdout, "%d chunks available on %s:%d\n", sock[i].chunks, inet_ntoa(sock[i].sa.sin_addr), ntohs(sock[i].sa.sin_port));
#endif
    }

#ifdef DEBUG
  fprintf(stdout, "\n");
#endif

  /* Sort sockets to make sure that active sockets are at the beginning of array */
  sock_sort(sock);
  
  return EXIT_SUCCESS;
}

int init_sockets(sock_t *sock, char *ip, int *ports)
{  
  int i;
  struct timeval time_out={PRD_SEC, 0}; 
  // Force to timeout if we could not receive data frames for one period.
  
  for (i = 0; i < NPORT_NIC; i++)
    {
      sock[i].active = 1;
      sock[i].ndf    = 0;
      sock[i].chunks = 0;
      sock[i].sock   = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
      setsockopt(sock[i].sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&time_out, sizeof(time_out));
            
      memset(&sock[i].sa, 0x00, sizeof(sock[i].sa));
      sock[i].sa.sin_family      = AF_INET;
      sock[i].sa.sin_port        = htons(ports[i]);
      sock[i].sa.sin_addr.s_addr = inet_addr(ip);
      
      if (-1 == bind(sock[i].sock, (struct sockaddr *)&sock[i].sa, sizeof(sock[i].sa)))
	{
	  multilog(runtime_log, LOG_ERR, "Bind to %s:%d failed, which happens at \"%s\", line [%d].\n", inet_ntoa(sock[i].sa.sin_addr), ntohs(sock[i].sa.sin_port), __FILE__, __LINE__);
	  fprintf (stderr, "Bind to %s:%d failed, which happens at \"%s\", line [%d].\n", inet_ntoa(sock[i].sa.sin_addr), ntohs(sock[i].sa.sin_port), __FILE__, __LINE__);
	  close(sock[i].sock);
	  sock[i].active = 0;
	  return EXIT_FAILURE;
	}
    }
    
  return EXIT_SUCCESS;
}

/* 
   Sort socket array to get active sockets at the beginning of socket array 
*/
int sock_sort(sock_t *sock)
{
  int i, j;
  sock_t sock_temp;

  for(i = 0; i < NPORT_NIC; i++)  
    {
      if(sock[i].active == 0)
	{
	  for(j = NPORT_NIC - 1; j > i; j--)
	    {
	      if(sock[j].active == 1)
		{
		  sock_temp = sock[i];
		  sock[i] = sock[j];
		  sock[j] = sock_temp;
		}
	    }
	}
    }   
  return EXIT_SUCCESS;
}

int init_capture(conf_t *conf, char *ip, int *ports)
{  
  int i;
  uint64_t tbufsz;
  int pkt_size;
  double elapsed_time;
  uint64_t rbufsz;
  int active_ports, active_chunks;
  struct timespec start, stop;
  FILE *conf_fp=NULL;
  char line[MSTR_LEN];

  pkt_size = (conf->hdr == 1) ? DF_SIZE : DT_SIZE;
  conf->pkt_size = pkt_size;             
  tbufsz = (uint64_t)TBUF_NDF * NCHK_NIC * (pkt_size + 1); // Size of minor buffer, we get one more byte here, we can check that byte to see if we have data there.
  conf->tbufsz = tbufsz;
  tbuf = (char *)malloc(tbufsz);

  conf->pkt_offset = (conf->hdr == 1) ? 0 : HDR_SIZE;

  rbufsz = (uint64_t)conf->rbuf_ndf * NCHK_NIC * conf->pkt_size;     // Block size of ring buffer;
  conf->rbufsz = rbufsz;
  
  /* Initialise ring buffer */
#ifdef DEBUG
  fprintf(stdout, "Start to initialise ring buffer\n");
  clock_gettime(CLOCK_REALTIME, &start);
#endif
  
  if(init_rbuf(conf) == EXIT_FAILURE)
    {
      multilog(runtime_log, LOG_ERR, "Can not initialise ring buffer, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf(stderr, "Can not initialise ring buffer, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
#ifdef DEBUG
  clock_gettime(CLOCK_REALTIME, &stop);
  elapsed_time = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1.0E9L;
  fprintf(stdout, "%f seconds used for the ring buffer initialisation\n", elapsed_time);
  fprintf(stdout, "End of the initialise\n\n");
#endif
    
  /* Initialise sockets */
  if(init_sockets(conf->sock, ip, ports) == EXIT_FAILURE)
    {
      multilog(runtime_log, LOG_ERR, "Can not initialise sockets, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf (stderr, "Can not initialise sockets, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  /* Check the available ports and frequency chunks */
  if(check_connection(conf->sock, &active_ports, &active_chunks) == EXIT_FAILURE)
    {
      multilog(runtime_log, LOG_ERR, "Can not check the connection, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf (stderr, "Can not check the connection, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }  
  conf->active_ports  = active_ports;
  conf->active_chunks = active_chunks;
    
  /* Align data frames from different sockets, which will make futhre work easier */
  if(align_df(conf->sock, active_ports) == EXIT_FAILURE)
    {
      multilog(runtime_log, LOG_ERR, "Can not align data frames, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf (stderr, "Can not align data frames, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE; 
    }

  /* Get the end condition of capture */
  acquire_hdr_end(conf->sock, conf->length, active_ports);
  
  /* Initialise the catpure status */
  for(i = 0; i < MPORT_NIC; i++)
    {
      transit[i] = 0;
      tail[i] = 0;
    }
  force_switch = 0;
  quit = 0;
  
  for(i = 0; i < MPORT_NIC; i++)
    finish[i] = 0;

  /* Initialise mutex */
  pthread_mutexattr_init(&ithread_mutex_attr);
  pthread_mutexattr_settype(&ithread_mutex_attr, PTHREAD_PROCESS_SHARED);
  pthread_mutexattr_init(&quit_mutex_attr);
  pthread_mutexattr_settype(&quit_mutex_attr, PTHREAD_PROCESS_SHARED);
  pthread_mutex_init(&ithread_mutex, &ithread_mutex_attr);
  pthread_mutexattr_init(&force_switch_mutex_attr);
  pthread_mutexattr_settype(&force_switch_mutex_attr, PTHREAD_PROCESS_SHARED);
  for(i = 0; i < active_ports; i++)
    {      
      pthread_mutexattr_init(&hdr_ref_mutex_attr[i]);
      pthread_mutexattr_settype(&hdr_ref_mutex_attr[i], PTHREAD_PROCESS_SHARED);
      pthread_mutex_init(&hdr_ref_mutex[i], &hdr_ref_mutex_attr[i]);
    }

  /* Register header, get available data block and get start time */
  for(i = 0; i < NPORT_NIC; i++)  // Get the active sock
    if(conf->sock[i].active)
      break;
  acquire_start_time(conf->sock[i].hdr_start, conf->efname, conf->utc_start, &(conf->picoseconds));
  if(register_header(conf))
    {
      multilog(runtime_log, LOG_ERR, "Header register failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf(stderr, "Header register failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }

  /* Get the buffer block ready */
  uint64_t block_id = 0;
  cbuf = ipcio_open_block_write(conf->hdu->data_block, &block_id);
  //cbuf = ipcbuf_get_next_write ((ipcbuf_t*)conf->hdu->data_block);
  
  return EXIT_SUCCESS;
}

/* This function is used to get all sockets ready for real data capture;
   1. Find out the current data frame with each socket;
   2. Find out most recent data frame among different sockets;
   3. Push the socket forward until all sockets drops the most recent data frame;
   4. The most recent data frame will be the reference one for data capture;
   5. Record the information about the most recent data frame for later use;
   We do not need to start at the reference data frame for real data capture as we use header to locate all captured data frames;

   The above is for original design, now we do not need to force all threads start at the same data frame;
   So the mean of this function is that we can get a reference hdr for later use;
*/
int align_df(sock_t *sock, int active_ports)
{
  int i;
  hdr_t hdr, hdr_current;
  char df[DF_SIZE];

#ifdef DEBUG
  fprintf(stdout, "*********************************************\n");
  fprintf(stdout, "WE ARE ON ALIGN_DF PART...\n");
  fprintf(stdout, "*********************************************\n");
#endif
  
  /* Set all members of hdr to zero */
  init_hdr(&hdr);
  hdr_current = hdr;

  /* Get the most recent data frame among different sockets; */
  for(i = 0; i < active_ports; i++)
    {
      if (recv(sock[i].sock, (void *)df, DF_SIZE, 0) < 0)
	// If the size is -1, means we get time out and the port is not active
	{
	  multilog(runtime_log, LOG_ERR, "Can not receive data from %s:%d, which happens at \"%s\", line [%d].\n", inet_ntoa(sock[i].sa.sin_addr), ntohs(sock[i].sa.sin_port), __FILE__, __LINE__);
	  fprintf (stderr, "Can not receive data from %s:%d, which happens at \"%s\", line [%d].\n", inet_ntoa(sock[i].sa.sin_addr), ntohs(sock[i].sa.sin_port), __FILE__, __LINE__);
	  sock[i].active = 0;
	  close(sock[i].sock);
	  return EXIT_FAILURE;
	}
      hdr_keys(df, &hdr_current);
      //hdr = ((hdr_current.idf + NDF_PRD * (hdr_current.sec - hdr.sec)/PRD_SEC) > hdr.idf) ? hdr_current : hdr;
      hdr = ((hdr_current.idf + (hdr_current.sec - hdr.sec)/TDF_SEC) > hdr.idf) ? hdr_current : hdr;  
#ifdef DEBUG
      //fprintf(stdout, "Current data frame is         %"PRIu64"\n", hdr_current.idf);
      fprintf(stdout, "The most recent data frame on port %d is second %"PRIu64", idf %"PRIu64"\n", ntohs(sock[i].sa.sin_port), hdr.sec, hdr.idf);
#endif
    }
#ifdef DEBUG
  fprintf(stdout, "\n");
#endif
  
  /* Drop data frame cross the most recent one */
  for (i = 0; i < active_ports; i++)
    {
      while(true)
	{
	  if (recv(sock[i].sock, (void *)df, DF_SIZE, 0) < 0)
	    // If the size is -1, means we get time out and the port is not active
	    {
	      multilog(runtime_log, LOG_ERR, "Can not receive data from %s:%d, which happens at \"%s\", line [%d].\n", inet_ntoa(sock[i].sa.sin_addr), ntohs(sock[i].sa.sin_port), __FILE__, __LINE__);
	      fprintf (stderr, "Can not receive data from %s:%d, which happens at \"%s\", line [%d].\n", inet_ntoa(sock[i].sa.sin_addr), ntohs(sock[i].sa.sin_port), __FILE__, __LINE__);
	      sock[i].active = 0;
	      close(sock[i].sock);
	      return EXIT_FAILURE;
	    }

	  hdr_keys(df, &sock[i].hdr_start);
	  //if((sock[i].hdr_start.idf + NDF_PRD * (sock[i].hdr_start.sec - hdr.sec)/PRD_SEC) > hdr.idf)
	  if((sock[i].hdr_start.idf + (sock[i].hdr_start.sec - hdr.sec)/TDF_SEC) > hdr.idf)
	    break;
	}
#ifdef DEBUG
      fprintf(stdout, "The most recent data frame after data frame dropping on port %d is second %"PRIu64", idf %"PRIu64"\n", ntohs(sock[i].sa.sin_port), hdr.sec, hdr_idf(df));
#endif
    }

#ifdef DEBUG
  fprintf(stdout, "The reference data frame information: beam %d, epoch %d, second %"PRIu64", idf %"PRIu64", freq %f\n\n", hdr.beam, hdr.epoch, hdr.sec, hdr.idf, hdr.freq);
#endif
  
  return EXIT_SUCCESS;
}

void *capture_thread(void *conf)
{
  conf_t *captureconf = (conf_t *)conf;
  uint64_t end_sec, end_idf, cbuf_loc, tbuf_loc;
  int64_t idf;
  int ifreq, ithread;
  hdr_t hdr;
  socklen_t fromlen;
  sock_t sock;
  char df[DF_SIZE];
  struct sockaddr_in sa;
  fromlen     = sizeof(sa);
  
  struct timespec start, stop;
  clock_gettime(CLOCK_REALTIME, &start);
  
  /* Get right socker for current thread */
  pthread_mutex_lock(&ithread_mutex);
  ithread = ithread_extern;
  ithread_extern++;
  pthread_mutex_unlock(&ithread_mutex);
  
  sock             = captureconf->sock[ithread];
  pthread_mutex_lock(&hdr_ref_mutex[ithread]);
  hdr_ref[ithread] = sock.hdr_start;
  pthread_mutex_unlock(&hdr_ref_mutex[ithread]);
  hdr                = hdr_ref[ithread];

  end_idf            = sock.hdr_end.idf;
  end_sec            = sock.hdr_end.sec;
  
  while((hdr.idf < end_idf || hdr.sec < end_sec) && (quit == 0))
    {      
      if(recvfrom(sock.sock, (void *)df, DF_SIZE, 0, (struct sockaddr *)&sa, &fromlen) == -1)
	{
	  multilog(runtime_log, LOG_ERR,  "Can not receive data from %s:%d, which happens at \"%s\", line [%d].\n", inet_ntoa(sock.sa.sin_addr), ntohs(sock.sa.sin_port), __FILE__, __LINE__);
	  fprintf (stderr, "Can not receive data from %s:%d, which happens at \"%s\", line [%d].\n", inet_ntoa(sock.sa.sin_addr), ntohs(sock.sa.sin_port), __FILE__, __LINE__);

	  /* Force to quit if we have time out */
	  pthread_mutex_lock(&quit_mutex);
	  quit = 1;
	  pthread_mutex_unlock(&quit_mutex);
	  
	  clock_gettime(CLOCK_REALTIME, &stop);
	  sock.hdr_end = hdr;
	  sock.elapsed_time = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1000000000.0L;
	  captureconf->sock[ithread] = sock;
	  conf = (void *)captureconf;

	  pthread_exit(NULL);
	  return NULL;
	}      
      hdr_keys(df, &hdr);               // Get header information, which will be used to get the location of packets
      acquire_ifreq(sa, &ifreq);              // Get the frequency index of received data frame with source ip address;
      
      pthread_mutex_lock(&hdr_ref_mutex[ithread]);
      acquire_idf(hdr, hdr_ref[ithread], &idf);  // How many data frames we get after the reference;
      pthread_mutex_unlock(&hdr_ref_mutex[ithread]);

      if (idf < 0 )
	// Drop data frams which are behind time;
	continue;
      else
	{
	  if(idf >= captureconf->rbuf_ndf)
	    {
	      /*
		Means we can not put the data into current ring buffer block anymore and we have to use temp buffer;
		If the number of chunks we used on temp buffer is equal to active chunks, we have to move to a new ring buffer block;
		If the temp buffer is too small, we may lose packets;
		If the temp buffer is too big, we will waste time to copy the data from it to ring buffer;
	      
		The above is the original plan, but it is too strict that we may stuck on some point;
		THE NEW PLAN IS we count the data frames which are not recorded with ring buffer, later than RBUF_NDF;
		If the counter is bigger than the active links, we active a new ring buffer block;
		Meanwhile, before reference hdr is updated, the data frames can still be put into temp buffer if it is not behind RBUF_NDF + TBUF_NDF;
		The data frames which are behind the limit will have to be dropped;
		the reference hdr update follows tightly and then sleep for about 1 millisecond to wait all capture threads move to new ring buffer block;
		at this point, no data frames will be put into temp buffer anymore and we are safe to copy data from it into the new ring buffer block and reset the temp buffer;
		If the data frames are later than RBUF_NDF + TBUF_NDF, we force the swtich of ring buffer blocks;
		The new plan will drop couple of data frames every ring buffer block, but it guarentee that we do not stuck on some point;
		We force to quit the capture if we do not get any data in one block;
		We also foce to quit if we have time out problem;
	      */
	      transit[ithread]++;
	     	      
	      if(idf >= (2 * captureconf->rbuf_ndf)) // quit
		{
		  /* Force to quit if we do not get any data in one data block */
		  pthread_mutex_lock(&quit_mutex);
		  quit = 1;
		  pthread_mutex_unlock(&quit_mutex);
		  
		  clock_gettime(CLOCK_REALTIME, &stop);
		  sock.hdr_end = hdr;
		  sock.elapsed_time = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1000000000.0L;
		  captureconf->sock[ithread] = sock;
		  conf = (void *)captureconf;
		  
#ifdef DEBUG
		  fprintf(stdout, "Too many temp data frames:\t%d\t%d\t%d\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%"PRId64"\n", ithread, ntohs(sock.sa.sin_port), ifreq, hdr_ref[ithread].sec, hdr_ref[ithread].idf, hdr.sec, hdr.idf, idf);
#endif
		  pthread_exit(NULL);
		  return NULL;
		}
	      else if(((idf >= (captureconf->rbuf_ndf + TBUF_NDF)) && (idf < (2 * captureconf->rbuf_ndf))))   // Force to get a new ring buffer block
		{
		  /* 
		     One possibility here: if we lose more that RBUF_NDF data frames continually, we will miss one data block;
		     for RBUF_NDF = 12500, that will be about 1 second data;
		     Do we need to deal with it?
		     I force the thread quit and also tell other threads quit if we loss one buffer;
		  */
#ifdef DEBUG
		  fprintf(stdout, "Forced force_switch %d\t%"PRIu64"\t%"PRIu64"\t%d\t%"PRIu64"\n", ithread, hdr.sec, hdr.idf, ifreq, idf);
#endif
		  pthread_mutex_lock(&force_switch_mutex);
		  force_switch = 1;
		  pthread_mutex_unlock(&force_switch_mutex);
		}
	      else  // Put data in to temp buffer
		{
		  tail[ithread] = (uint64_t)((idf - captureconf->rbuf_ndf) * NCHK_NIC + ifreq); // This is in TFTFP order
		  tbuf_loc         = (uint64_t)(tail[ithread] * (captureconf->pkt_size + 1));
		  tail[ithread]++;  // Otherwise we will miss the last available data frame in tbuf;
		  
		  tbuf[tbuf_loc] = 'Y';
		  memcpy(tbuf + tbuf_loc + 1, df + captureconf->pkt_offset, captureconf->pkt_size);	  
		  sock.ndf++;
		}	      
	    }
	  else
	    {
	      transit[ithread] = 0;
	      // Put data into current ring buffer block if it is before RBUF_NDF;
	      cbuf_loc = (uint64_t)((idf * NCHK_NIC + ifreq) * captureconf->pkt_size); // This is in TFTFP order
	      //cbuf_loc = (uint64_t)((idf + ifreq * RBUF_NDF) * captureconf->pkt_size);   // This should give us FTTFP (FTFP) order
	      memcpy(cbuf + cbuf_loc, df + captureconf->pkt_offset, captureconf->pkt_size);
	      
	      sock.ndf++;
	    }
	}
    }
  finish[ithread] = 1;
  
  /* Return value for statistics check */
  clock_gettime(CLOCK_REALTIME, &stop);
  sock.elapsed_time = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1000000000.0L;
  sock.hdr_end = hdr;
  captureconf->sock[ithread] = sock;
  conf = (void *)captureconf;
    
  /* Exit */
  pthread_exit(NULL);
  return NULL;
}

int acquire_idf(hdr_t hdr, hdr_t hdr_ref, int64_t *idf)
{
  //*idf = (int64_t)(hdr.idf + (hdr.sec - hdr_ref.sec)/PRD_SEC * NDF_PRD - hdr_ref.idf);
  //*idf = (int64_t)hdr.idf + (int64_t)(hdr.sec - hdr_ref.sec)/PRD_SEC * NDF_PRD - (int64_t)hdr_ref.idf;
  *idf = (int64_t)hdr.idf + (int64_t)(hdr.sec - hdr_ref.sec) / TDF_SEC - (int64_t)hdr_ref.idf;
  return EXIT_SUCCESS;
}

int acquire_ifreq(struct sockaddr_in sa, int *ifreq)
{
  /*
    Here we assume that the last digital of BMF ip address counts from 1;
    10.16.X.1 to 10.16.X.12, X is from 1 to 8;
    If not, we will be in trouble and need to update here;
    The first 48 links are 1, 3, 5, 7, 9, 11 of each BMF
    The second 48 links are 2, 4, 6, 8, 10, 12 of each BMF;
  */
  
  unsigned char *ip = (unsigned char *)&sa.sin_addr.s_addr;
  *ifreq = (int)(ip[2] - 1) * NCHK_BMF + (int)ceil((double)(ip[3]/2.0)) - 1;
  ip = NULL;  
  return EXIT_SUCCESS;
}

int init_rbuf(conf_t *conf)
{
  int i, nbufs;
  ipcbuf_t *db = NULL;
  conf->hdu = dada_hdu_create(runtime_log);
  dada_hdu_set_key(conf->hdu, conf->key);
    
  if (dada_hdu_connect(conf->hdu) < 0)
    {
      multilog(runtime_log, LOG_ERR, "Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf (stderr, "Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;    
    }
  db = (ipcbuf_t *) conf->hdu->data_block;
  if(conf->rbufsz != ipcbuf_get_bufsz((ipcbuf_t *)db))  
    {
      multilog(runtime_log, LOG_ERR, "Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf (stderr, "Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;    
    }
  
  if(ipcbuf_get_bufsz(conf->hdu->header_block) != DADA_HDR_SIZE)    // This number should match
    {
      multilog(runtime_log, LOG_ERR, "Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf (stderr, "Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;    
    }
  
  /* make ourselves the write client */
  if (dada_hdu_lock_write (conf->hdu) < 0)
    {
      multilog (runtime_log, LOG_ERR, "Error locking HDU, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf(stderr, "Error locking HDU, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }

  if(conf->sod)
    {      
      if(ipcbuf_enable_sod((ipcbuf_t *)db, 0, 0) < 0)  // We start at the beginning
  	{
	  multilog (runtime_log, LOG_ERR, "Can not write data before start, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  	  fprintf(stderr, "Can not write data before start, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  	  return EXIT_FAILURE;
  	}
    }
  else
    {
      if(ipcbuf_disable_sod((ipcbuf_t *)db) < 0)
  	{
	  multilog (runtime_log, LOG_ERR, "Can not write data before start, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  	  fprintf(stderr, "Can not write data before start, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  	  return EXIT_FAILURE;
  	}
    }

  return EXIT_SUCCESS;
}

int destroy_capture(conf_t conf)
{  
  free(tbuf);
  destroy_sockets(conf.sock);
  
  /* Destroy mutex */
  int i;
  pthread_mutex_destroy(&ithread_mutex);
  pthread_mutex_destroy(&quit_mutex);
  pthread_mutex_destroy(&force_switch_mutex);
  for(i = 0; i < conf.active_ports; i++)
    pthread_mutex_destroy(&hdr_ref_mutex[i]);

  dada_hdu_unlock_write(conf.hdu);
  dada_hdu_disconnect(conf.hdu);
  dada_hdu_destroy(conf.hdu);

  return EXIT_SUCCESS;
}

int destroy_sockets(sock_t *sock)
{
  int i;
  for(i = 0; i < NPORT_NIC; i++)  // Clean up sockets and ip
    {
      if(sock[i].active)
  	close(sock[i].sock);
    }
  return EXIT_SUCCESS;
}

int acquire_hdr_end(sock_t *sock, double length, int active_ports)
{
  int i;

  for(i = 0; i < active_ports; i++)
    {
      /* We stop at the first data frame, does not matter from which chunks */
      //sock[i].hdr_end.sec = (uint64_t)((int)(length/PRD_SEC) * PRD_SEC + sock[i].hdr_start.sec);
      //sock[i].hdr_end.idf = (uint64_t)((length - (int)(length/PRD_SEC) * PRD_SEC)/TDF_SEC) + sock[i].hdr_start.idf;
      //fprintf(stdout, "%"PRIu64"\t%"PRIu64"\n", sock[i].hdr_end.sec, sock[i].hdr_end.idf);
      
      sock[i].hdr_end.sec = (uint64_t)(length - fmod(length, PRD_SEC) + sock[i].hdr_start.sec);
      sock[i].hdr_end.idf = (uint64_t)(fmod(length, PRD_SEC) / TDF_SEC) + sock[i].hdr_start.idf;
      //fprintf(stdout, "%"PRIu64"\t%"PRIu64"\n", sock[i].hdr_end.sec, sock[i].hdr_end.idf);
      
      if(sock[i].hdr_end.idf >= NDF_PRD)
	{
	  sock[i].hdr_end.sec = sock[i].hdr_end.sec + PRD_SEC;
	  sock[i].hdr_end.idf = sock[i].hdr_end.idf - NDF_PRD;
	}
    }
      
  return EXIT_SUCCESS;
}

int statistics(conf_t conf)
{
  int i;
  uint64_t ndf_expected;
  uint64_t ndf_real;
  sock_t sock;

  multilog(runtime_log, LOG_INFO, "Address\t\tPort\tChunks\tElapsed\tExpected\tReal\tLoss\n");
  fprintf(stdout, "\nAddress\t\tPort\tChunks\tElapsed\tExpected\tReal\tLoss\n");
  for(i = 0; i < conf.active_ports; i++)
    {
      sock = conf.sock[i];
      
      ndf_real = sock.ndf;
      //ndf_expected = (uint64_t)(sock.chunks * (sock.hdr_end.idf - sock.hdr_start.idf + NDF_PRD * (sock.hdr_end.sec - sock.hdr_start.sec)/PRD_SEC));
      //ndf_expected = (uint64_t)(sock.chunks * (sock.hdr_end.idf - sock.hdr_start.idf + (sock.hdr_end.sec - sock.hdr_start.sec)/TDF_SEC));
      //ndf_expected = (uint64_t)(sock.chunks * (sock.hdr_end.idf - sock.hdr_start.idf + (sock.hdr_end.sec - sock.hdr_start.sec)/TDF_SEC));
      ndf_expected = (uint64_t)(sock.chunks * conf.length/TDF_SEC);
      
      //fprintf(stdout, "%f\n", (sock.hdr_end.sec - sock.hdr_start.sec)/TDF_SEC);
      multilog(runtime_log, LOG_INFO, "%s\t\t%d\t%d\t%.3f\t%"PRIu64"\t\t%"PRIu64"\t%.1E\n", inet_ntoa(sock.sa.sin_addr), ntohs(sock.sa.sin_port), sock.chunks, sock.elapsed_time, ndf_expected, ndf_real, (double)((int64_t)(ndf_expected - ndf_real))/ndf_expected);
      fprintf(stdout, "%s\t%d\t%d\t%.3f\t%"PRIu64"\t\t%"PRIu64"\t%.1E\n", inet_ntoa(sock.sa.sin_addr), ntohs(sock.sa.sin_port), sock.chunks, sock.elapsed_time, ndf_expected, ndf_real, (double)((int64_t)(ndf_expected - ndf_real))/ndf_expected);
    }
  
  return EXIT_SUCCESS;
}

int register_header(conf_t *conf)
{
  int i;
  char *hdrbuf = NULL;
  double bytes_per_second, obs_offset;
  
  hdrbuf = ipcbuf_get_next_write (conf->hdu->header_block);
  if (!hdrbuf)
    {
      multilog(runtime_log, LOG_ERR, "Error getting header_buf, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf(stderr, "Error getting header_buf, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
    
  /* if header file is presented, use it. If not set command line attributes */ 
  if (!conf->hfname)
    {
      multilog(runtime_log, LOG_ERR, "Please specify header file, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf(stderr, "Please specify header file, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  if (fileread(conf->hfname, hdrbuf, DADA_HDR_SIZE) < 0)
    {
      multilog(runtime_log, LOG_ERR, "Error reading header file, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf(stderr, "Error reading header file, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }

  //multilog(runtime_log, LOG_INFO, "UTC_START:\t%s\n", conf->utc_start);
  //fprintf(stdout, "Setup UTC_START at capture stage:\t%s\n", conf->utc_start);
  if (ascii_header_set(hdrbuf, "UTC_START", "%s", conf->utc_start) < 0)  // Here we only set the UTC with integer period, not set MJD. fraction of period is shown as obs_offset
    {
      multilog(runtime_log, LOG_ERR, "Error setting UTC_START, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf(stderr, "Error setting UTC_START, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  //if (ascii_header_set(hdrbuf, "PICOSECONDS", "%"PRIu64"", conf->picoseconds) < 0)  // Here we only set the UTC with integer period, not set MJD. fraction of period is shown as obs_offset
  if (ascii_header_set(hdrbuf, "PICOSECONDS", "%"PRIu64, conf->picoseconds) < 0)  // Here we only set the UTC with integer period, not set MJD. fraction of period is shown as obs_offset
    {
      multilog(runtime_log, LOG_ERR, "Error setting PICOSECONDS, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf(stderr, "Error setting PICOSECONDS, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
    
  if (ascii_header_set(hdrbuf, "FREQ", "%.1lf", conf->freq) < 0)
    {
      multilog(runtime_log, LOG_ERR, "Error setting FREQ, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf(stderr, "Error setting FREQ, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }

  /* donot set header parameters anymore - acqn. doesn't start */
  if (ipcbuf_mark_filled (conf->hdu->header_block, DADA_HDR_SIZE) < 0)
    {
      multilog(runtime_log, LOG_ERR, "Error header_fill, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf(stderr, "Error header_fill, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  return EXIT_SUCCESS;
}

int acquire_start_time(hdr_t hdr_start, char efname[MSTR_LEN], char utc_start[MSTR_LEN], uint64_t *picoseconds)
{
  int epoch;
  FILE *fp = NULL;
  char line[MSTR_LEN];
  time_t sec;
  double sec_prd, mjd_epoch, micoseconds;
  
  fp = fopen(efname, "r");
  if(fp == NULL)
    {
      multilog(runtime_log, LOG_ERR, "Can not open epoch file: %s, which happens at \"%s\", line [%d].\n", efname, __FILE__, __LINE__);
      fprintf(stderr, "Can not open epoch file: %s, which happens at \"%s\", line [%d].\n", efname, __FILE__, __LINE__);
      return EXIT_FAILURE;
    }

  while(fgets(line, MSTR_LEN, fp))
    {
      if(!(line[0]=='#'))
	{
	  sscanf(line, "%d %lf %*s", &epoch, &mjd_epoch);
	  if(epoch == hdr_start.epoch)
	    break;
	}
    }
  fclose(fp);
  
  /* YYYY-MM-DD-hh:mm:ss */
  sec_prd = hdr_start.idf * TDF_SEC;  // seconds in one period
  sec = SECDAY * mjd_epoch + hdr_start.sec + floor(sec_prd);  // int seconds from 1970-01-01
  strftime (utc_start, MSTR_LEN, DADA_TIMESTR, gmtime(&sec)); // String start time without fraction second

  /* Faction of second */
  micoseconds  = 1.0E6 * (sec_prd - floor(sec_prd)); // We may have 1 picosecond deviation here, round to intergal will fix that;
  *picoseconds = 1E6 * round(micoseconds); // We know for sure that the timing resolution is 108 microsecond, we can not get finer timing stamps than 1 microsecond;
  
#ifdef DEBUG
  fprintf(stdout, "UTC_START:\t%s\n\n", utc_start);
#endif

  multilog(runtime_log, LOG_INFO, "SEC_START:\t%"PRIu64"\n", hdr_start.sec);
  multilog(runtime_log, LOG_INFO, "IDF_START:\t%"PRIu64"\n", hdr_start.idf);
    
  multilog(runtime_log, LOG_INFO, "SECOND_IN_PERIOD:\t%.12f\n", sec_prd);
  multilog(runtime_log, LOG_INFO, "MICROSECONDS:\t%f\n", micoseconds);
  
  multilog(runtime_log, LOG_INFO, "UTC_START:\t%s\n", utc_start);
  multilog(runtime_log, LOG_INFO, "PICOSECONDS:\t%"PRIu64"\n", *picoseconds);
  
  fprintf(stdout, "SECOND_IN_PERIOD:\t%.12f\tUTC_START:\t%s\tMICROSECONDS:\t%f\tPICOSECONDS:\t%"PRIu64"\n", sec_prd, utc_start, micoseconds, *picoseconds);
  
  return EXIT_SUCCESS;
}
