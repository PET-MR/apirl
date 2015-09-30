#Para debuggear el APIRL
#export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/sources/CopiasAPIRL/101110/build/debug/bin

#Para release el APIRL
#export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/usr/local/APIRL/bin

export LD_LIBRARY_PATH=$APIRL_PATH/build/debug/bin/:/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
export PATH=$PATH:$APIRL_PATH/build/debug/bin/:/usr/local/cuda/bin/
