mkdir -p build
cd build
rm -rf *
cmake -DCUDA_COMPUTE_CAPABILITY=3.5 ..
make -j32
