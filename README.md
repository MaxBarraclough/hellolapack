ABOUT
=====

Small example C++ program showing example uses of LAPACK and BLAS:
  * Matrix multiplication (various examples)
  * Matrix inversion
  * Polynomial regression (i.e. linear regression)

Uses CMake for build.
Compiled and tested on Ubuntu/AMD64/g++, using OpenBLAS.


Dependencies
------------

Depends on LAPACK and BLAS. On Ubuntu, the following should install what you need:

    sudo apt-get update
    sudo apt-get install build-essential cmake libblas-dev liblapack-dev

Backends other than OpenBLAS should also work, but have not been tested.


Building
--------

The following should configure, build, and run:

    bash ./configure_cmake_unix.sh && ./build_unix_release.sh && ./build/hellolapack

