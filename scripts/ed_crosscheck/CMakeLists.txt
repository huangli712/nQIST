# Build of the Pomerol reference of the ED cross-check:
#
#   cmake -S scripts/ed_crosscheck -B <build-dir>
#   cmake --build <build-dir>
#
# inside an environment providing `pomerol` (conda package, channel
# `krivenko`), a C++ compiler and MPI.
cmake_minimum_required(VERSION 3.11)
project(nqist_ed_crosscheck CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pomerol REQUIRED)

add_executable(pomerol_ref pomerol_ref.cpp)
target_link_libraries(pomerol_ref PRIVATE pomerol::pomerol)
