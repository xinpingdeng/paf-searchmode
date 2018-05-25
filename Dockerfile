# Develop a image from xinpingdeng/cuda9.1-cudnn7-devel-ubuntu16.04-with-paf-essential
#FROM xinpingdeng/cuda9.1-cudnn7-devel-ubuntu16.04-with-paf-essential
FROM xinpingdeng/essential
# docker run --runtime=nvidia ....

# Set current user to pulsar and change to $HOME
USER pulsar
ENV HOME /home/pulsar
WORKDIR $HOME

# Fold_mode pipeline
RUN git clone https://github.com/xinpingdeng/paf-searchmode.git 
WORKDIR $HOME/paf-searchmode
RUN git config user.name "xinpingdeng" && \
    git config user.email "xinpingdeng.deng@gmail.com" && \
    git checkout dev 
RUN ./rebuild.py 0

# ./fold_stream.py -a fold_stream.conf -b 0 -c 10 -d /beegfs/DENG/docker/ -e J0218+4232 -f 0
#ENTRYPOINT ["./fold_stream.py"]