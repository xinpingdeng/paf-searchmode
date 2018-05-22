#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdlib.h>

#include "diskdb.cuh"
#include "paf_diskdb.cuh"

int main(int argc, char **argv)
{
  int arg;
  char conf_fname[MSTR_LEN];
  conf_t conf;
  char fdir[MSTR_LEN], fname[MSTR_LEN];
  
  while((arg=getopt(argc,argv,"k:c:s:d:n:h:g:")) != -1)
    {
      switch(arg)
	{
	case 'k':	  	  
	  if (sscanf (optarg, "%x", &conf.key) != 1)
	    {
	      fprintf (stderr, "Could not parse key from %s, which happens at \"%s\", line [%d].\n", optarg, __FILE__, __LINE__);
	      return EXIT_FAILURE;
	    }
	  break;
	  
	case 'c':
	  sscanf(optarg, "%s", conf_fname);
	  break;
	  
	case 's':
	  sscanf(optarg, "%d", &conf.sod);
	  break;
	  
	case 'g':
	  sscanf(optarg, "%d", &conf.device_id);
	  break;
	  
	case 'd':
	  sscanf(optarg, "%s", &fdir);
	  break;
	  
	case 'n':
	  sscanf(optarg, "%s", &fname);
	  break;

	case 'h':
	  sscanf(optarg, "%s", conf.hfname);
	  break;
	}
    }
  sprintf(conf.fname, "%s/%s", fdir, fname);
  
  init_diskdb(conf_fname, &conf);
  do_diskdb(conf);
  destroy_diskdb(conf);

  return EXIT_SUCCESS;
}
