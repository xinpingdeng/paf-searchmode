# paf-foldmode

Phased Array Feed fold mode pipeline and associated source code in c and cuda.

There are three parts with the pipeline:

<<<<<<< HEAD
1. Raw data capture part, which receives data from NiC and put data into a big ring buffer with TFTFP order;
2. Data pre-porcessing part, which
  a. Reorder the data into PFT order;
  b. 32-points FFT the reordered data, and the data will be in PFTF;
  c. Swap the halves of FFT output and make sure that the center frequency is in the central;
  d. Drop the first 3 and last 2 data points to downsample the data;
  e. Drop 440 channls on each side of passband;
  f. Reorder the PFTF data into PTF;
  g. Swap the halves of 32-points segments;
  h. Reverse FFT to the swapped data and get PTFT order data;
  i. Reorder the PTFT data into TFP order;
2. Dspsr to process and fold the data from step 2;

To use the pipeline, fold.py -c fold.conf -d 0 -n 0 -l 10
fold.conf defines basic fold parameters;
-d tell the pipeline to print out debug information or not;
-n tell the pipeline to linse to which NiC;
-l tell the pipeline to linse on the NiC on how long;

The configuration fold.conf here is specified for the Effelsberg Phased Array Feed GPU cluster, you need to update the configuration file accordingly if you want to reuse the code somewhere else (you can find the detail of each parameter in the file). You also need to check your IP address and port number, and if it is necessary, please undate it at capture.c.

Makefile should works to compile the capture (paf-capture) and pre-process (paf-process) part, you need to install dspsr and psrdada on your system. Here we use GPU to accelerate the folding process and make the whole pipeline runs in real-time, if you want to do so, please make sure that your dspsr works with CUDA (check here for detail http://dspsr.sourceforge.net/manuals/dspsr/gpu.shtml), otherwise just remove the -cuda ooption and associated parameters from fold.py.  
=======
Raw data capture part, which receives data from NiC and put data into a big ring buffer with TFTFP order;

Data pre-porcessing part, which a. Reorder the data into PFT order; b. 32-points FFT the reordered data, and the data will be in PFTF; c. Swap the halves of FFT output and make sure that the center frequency is in the central; d. Drop the first 3 and last 2 data points to downsample the data; e. Drop 440 channls on each side of passband; f. Reorder the PFTF data into PTF; g. Swap the halves of 32-points segments; h. Reverse FFT to the swapped data and get PTFT order data; i. Reorder the PTFT data into TFP order;

Dspsr to process and fold the data from step 2;

To use the pipeline, fold.py -c fold.conf -d 0 -n 0 -l 10 -f 0 fold.conf defines basic fold parameters; -d tell the pipeline to print out debug information or not (more detail debug information can be enabled at compile, see later); -n tell the pipeline to linse to which NiC; -l tell the pipeline to linse on the NiC on how long, -f tell the pipeline to create shared memory if it is the first time you run it (-f 0), destroy shared memory if it is the last time you run it (-f 1) or do not do anything to shared memory;

The configuration fold.conf here is specified for the Effelsberg Phased Array Feed GPU cluster, you need to update the configuration file accordingly if you want to reuse the code somewhere else (you can find the detail of each parameter in the file). You also need to check your IP address and port number, and if it is necessary, please undate it at capture.c.

rebuild.py should recompile the capture (paf-capture) and pre-process (paf-process) part (rebuild.py 0 tell program not print out debug information, rebuild.py 1 will enable debug information printout), you need to install dspsr and psrdada on your system. Here we use GPU to accelerate the folding process and make the whole pipeline runs in real-time, if you want to do so, please make sure that your dspsr works with CUDA (check here for detail http://dspsr.sourceforge.net/manuals/dspsr/gpu.shtml), otherwise just remove the -cuda ooption and associated parameters from fold.py.

More detail for myself reference. It uses UTC_START and PICOSECONDS to determine the precise start time of observation.
>>>>>>> dev
