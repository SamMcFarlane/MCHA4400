# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "D:/University Work/2023/MECHA4400/MCHA4400/Labs/Week6/lab6/build/_deps/nanobench-src"
  "D:/University Work/2023/MECHA4400/MCHA4400/Labs/Week6/lab6/build/_deps/nanobench-build"
  "D:/University Work/2023/MECHA4400/MCHA4400/Labs/Week6/lab6/build/_deps/nanobench-subbuild/nanobench-populate-prefix"
  "D:/University Work/2023/MECHA4400/MCHA4400/Labs/Week6/lab6/build/_deps/nanobench-subbuild/nanobench-populate-prefix/tmp"
  "D:/University Work/2023/MECHA4400/MCHA4400/Labs/Week6/lab6/build/_deps/nanobench-subbuild/nanobench-populate-prefix/src/nanobench-populate-stamp"
  "D:/University Work/2023/MECHA4400/MCHA4400/Labs/Week6/lab6/build/_deps/nanobench-subbuild/nanobench-populate-prefix/src"
  "D:/University Work/2023/MECHA4400/MCHA4400/Labs/Week6/lab6/build/_deps/nanobench-subbuild/nanobench-populate-prefix/src/nanobench-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "D:/University Work/2023/MECHA4400/MCHA4400/Labs/Week6/lab6/build/_deps/nanobench-subbuild/nanobench-populate-prefix/src/nanobench-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "D:/University Work/2023/MECHA4400/MCHA4400/Labs/Week6/lab6/build/_deps/nanobench-subbuild/nanobench-populate-prefix/src/nanobench-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
