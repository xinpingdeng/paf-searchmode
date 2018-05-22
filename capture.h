#ifndef _H
#define _H

#include <netinet/in.h>
#include "hdr.h"
#include "paf_capture.h"

#include "dada_hdu.h"
#include "dada_def.h"
#include "ipcio.h"
#include "ascii_header.h"
#include "daemon.h"
#include "futils.h"

#define DADA_TIMESTR      "%Y-%m-%d-%H:%M:%S"

#define DADA_HDR_SIZE     4096
/* Stream configuration */
#define MCHK_PORT         8   // How many chunks per port are allowed;
#define NCHK_NIC          48  // How many frequency chunks we will receive, we should read the number from metadata
#define NCHK_BMF          6   // How many chunks each BMF process
#define MPORT_NIC         6
#define NPORT_NIC         6   // This number should from metadata
#define PORT_BASE         17100

/* Parameters of data frames */
#define DF_SIZE     	  7232     // The size in byte for one data frame with headr
#define DT_SIZE     	  7168     // The size in byte for one data frame without header
#define HDR_SIZE    	  64       // The size in byte for header
#define TDF_SEC           1.08E-4 // Sampling time in seconds
#define PRD_SEC     	  27       // The time period of streaming, should read from header
#define NDF_PRD  	  250000   // Number of data frames in one period

/* Configure for capture */
#define TBUF_NDF          256       // How many data frames (from all frequency chunks) we will record in minor buffer;
#define SLP_NS            1.08E5    // Sleep time in nano-seconds for capture part;  
#define NDF_CHECK         800 // How may data frames we check on each port to get available frequency chunks

/* NUMA node */
#define NNUMA       	  2
#define NCPU_NUMA   	  10

#define SECDAY            86400.0
#define MJD1970           40587.0
typedef struct sock_t
{
  int sock;
  int active;
  int chunks;
  uint64_t ndf;
  double elapsed_time;
  double freq[MCHK_PORT];
  struct sockaddr_in sa;
  struct hdr_t hdr_start, hdr_end;
}sock_t;

typedef struct conf_t
{
  int pkt_size, pkt_offset;  // pkt_offset is used for the case we do not record header of data frame, if we record header of data frame, the pkt_offset is 0;
  // Also pkt_size is the same with the size of data frame if we record header of each data frame, otherwise it is the size of data block of data frame;
  int active_ports, active_chunks;
  uint64_t rbufsz, tbufsz;
  int sod;
  
  key_t key;
  dada_hdu_t *hdu;
  
  size_t rbuf_ndf;// rbuf_nblk;
  int hdr;

  char utc_start[MSTR_LEN];
  uint64_t picoseconds;
  double freq;
  char hfname[MSTR_LEN];
  char efname[MSTR_LEN];

  double length;

  char dir[MSTR_LEN];
  struct sock_t sock[MPORT_NIC];
}conf_t;

int acquire_ifreq(struct sockaddr_in sa, int *ifreq);
int acquire_idf(hdr_t hdr, hdr_t hdr_ref, int64_t *idf);
int acquire_hdr_end(sock_t *sock, double length, int active_ports);

int init_sockets(sock_t *sock, char *ip, int *ports);
//int init_rbuf(uint64_t rbufsz, int nbufs, key_t key);

  int check_connection(sock_t *sock, int *active_ports, int *active_chunks);
int sock_sort(sock_t *sock);
int align_df(sock_t *sock, int active_ports);

int destroy_capture(conf_t conf);
int destroy_sockets(sock_t *sock);

void *capture_thread(void *conf);
int statistics(conf_t conf);

int init_rbuf(conf_t *conf);
int init_capture(conf_t *conf, char *ip, int *ports);
int register_header(conf_t *conf);
int acquire_start_time(hdr_t hdr_start, char efname[MSTR_LEN], char utc_start[MSTR_LEN], uint64_t *picoseconds);
#endif
