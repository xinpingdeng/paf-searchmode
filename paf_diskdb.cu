#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdlib.h>

#include "diskdb.cuh"
#include "paf_diskdb.cuh"

void usage()
{
  fprintf (stdout,
	   "paf_diskdb - read dada data file into shared memory \n"
	   "\n"
	   "Usage: paf_capture [options]\n"
	   " -a Hexadecimal shared memory key for capture \n"
	   " -b Directory with data file \n"
	   " -c The name of data file    \n"
	   " -d The name of header file  \n"
	   " -e Enable start-of-data or not \n"
	   " -h Show help    \n");
}

int main(int argc, char **argv)
{
  int arg;
  char fdir[MSTR_LEN], fname[MSTR_LEN];
  conf_t conf;
  
  while((arg=getopt(argc,argv,"a:b:c:d:e:h:")) != -1)
    {
      switch(arg)
	{
	case 'h':
	  usage();
	  return EXIT_FAILURE;
	  
	case 'a':	  	  
	  if (sscanf (optarg, "%x", &conf.key) != 1)
	    {
	      fprintf (stderr, "Could not parse key from %s, which happens at \"%s\", line [%d].\n", optarg, __FILE__, __LINE__);
	      return EXIT_FAILURE;
	    }
	  break;
	 
	case 'b':
	  sscanf(optarg, "%s", fdir);
	  break;
	  
	case 'c':
	  sscanf(optarg, "%s", fname);
	  break;

	case 'd':
	  sscanf(optarg, "%s", conf.hfname);
	  break;
	  
	case 'e':
	  sscanf(optarg, "%d", &conf.sod);
	  break;
	}
    }
  sprintf(conf.fname, "%s/%s", fdir, fname);
  
  init_diskdb(&conf);
  do_diskdb(conf);
  destroy_diskdb(conf);

  return EXIT_SUCCESS;
}
