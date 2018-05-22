#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdlib.h>

#include "paf_capture.h"
#include "sync.h"
#include "capture.h"
#include "multilog.h"
#include "dada_def.h"

extern int quit;
multilog_t *runtime_log;

/*
  1. Finish the ring buffer part, should be working now;
  2. Replace frequency with source ip for the connection check, do not do that as it also give us frequency information;
  3. Better debug option, DONE???
  4. Get log information, kown how to do it now, but do not know how to view it;
  5. Use single thread program to count the data frames, done,we still have capture problem;
  6. Data write out;
  7. Check the write out data;
*/

int main(int argc, char **argv)
{
  /* Initial part */ 
  int i, arg;
  int node_id = 0;                    // Default node id;
  int nic_id = 1;                     // Default nic id;
  char ip[MSTR_LEN];
  double length = 36.000;             // Default observation length in seconds;
  char hostname[HN_LEN + 1];          // The name of host computer;
  conf_t conf;
  FILE *fp_log = NULL;
  
  /* Setup log interface */
  fp_log = fopen("paf_capture.log", "ab+"); // File to record log information
  if(fp_log == NULL)
    {
      fprintf(stderr, "Can not open log file paf_capture.log\n");
      return EXIT_FAILURE;
    }
  runtime_log = multilog_open("paf_capture", 1);
  multilog_add(runtime_log, fp_log);
  multilog(runtime_log, LOG_INFO, "START PAF_CAPTURE\n");
  
  while((arg=getopt(argc,argv,"k:l:n:c:h:f:e:s:r:d:")) != -1)
    {
      switch(arg)
	{
	case 'k':	  	  
	  if (sscanf (optarg, "%x", &conf.key) != 1)
	    {
	      multilog(runtime_log, LOG_INFO, "Could not parse key from %s, which happens at \"%s\", line [%d].\n", optarg, __FILE__, __LINE__);
	      //fprintf (stderr, "Could not parse key from %s, which happens at \"%s\", line [%d].\n", optarg, __FILE__, __LINE__);
	      return EXIT_FAILURE;
	    }
	  break;

	case 'l':
	  sscanf(optarg, "%lf", &conf.length);
	  break;
	  
	case 'n':
	  sscanf(optarg, "%d", &nic_id);
	  break;

	case 'h':	  	  
	  sscanf(optarg, "%s", conf.hfname);
	  break;

	case 'e':
	  sscanf(optarg, "%s", conf.efname);
	  break;
	  
	case 'f':
	  sscanf(optarg, "%lf", &conf.freq);
	  break;
	  
	case 's':
	  sscanf(optarg, "%d", &conf.sod);
	  break;
	  
	case 'r':
	  sscanf(optarg, "%d", &conf.rbuf_ndf);
	  break;
	  
	case 'd':
	  sscanf(optarg, "%d", &conf.hdr);
	  break;
	}
    }

  hostname[HN_LEN] = '0';
  gethostname(hostname, HN_LEN + 1);
  node_id = hostname[HN_LEN - 1] - '0';
  sprintf(ip, "10.17.%d.%d", node_id, nic_id);
    
#ifdef DEBUG
  fprintf(stdout, "*********************************************\n");
  fprintf(stdout, "WE ARE ON INIT PART OF WHOLE PROGRAM...\n");
  fprintf(stdout, "*********************************************\n");
#endif
  
#ifdef DEBUG
  fprintf(stdout, "\nWe are working on %s, nic %d.\n", hostname, nic_id);
  fprintf(stdout, "The IP address is %s.\n\n", ip);
#endif
  
  /* Capture part*/
  int ports[MPORT_NIC];
  for (i = 0; i < NPORT_NIC; i++)
    ports[i] = PORT_BASE + i;
  if(init_capture(&conf, ip, ports) == EXIT_FAILURE)
    {
      multilog(runtime_log, LOG_INFO, "Can not initialise the capture, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      //fprintf (stderr, "Can not initialise the capture, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;      
    }
  if(threads(&conf) == EXIT_FAILURE)
    {
      multilog(runtime_log, LOG_INFO, "Can not capture data, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      //fprintf(stderr, "Can not capture data, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }

  /* Check the result */
  statistics(conf);
  
  /* Cleanup */
  if(destroy_capture(conf) == EXIT_FAILURE)
    {
      multilog(runtime_log, LOG_INFO, "Can not destroy buffers, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      //fprintf(stderr, "Can not destroy buffers, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }

#ifdef DEBUG
  fprintf(stdout, "\n*********************************************\n");
  fprintf(stdout, "WE ARE ON FINISHING PART...\n");
  fprintf(stdout, "*********************************************\n");

  if(quit == 1)
    fprintf(stdout, "Forced quit!\n\n");
  else
    fprintf(stdout, "Finish successful!\n\n");
#endif
  
  /* Destory log interface */
  multilog(runtime_log, LOG_INFO, "FINISH PAF_CAPTURE\n\n");
  multilog_close(runtime_log);
  fclose(fp_log);
  
  return EXIT_SUCCESS;
}
