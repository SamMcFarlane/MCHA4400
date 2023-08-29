# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

if(EXISTS "D:/University Work/2023/MECHA4400/MCHA4400/Labs/Week7/lab7/build/_deps/nanobench-subbuild/nanobench-populate-prefix/src/nanobench-populate-stamp/nanobench-populate-gitclone-lastrun.txt" AND EXISTS "D:/University Work/2023/MECHA4400/MCHA4400/Labs/Week7/lab7/build/_deps/nanobench-subbuild/nanobench-populate-prefix/src/nanobench-populate-stamp/nanobench-populate-gitinfo.txt" AND
  "D:/University Work/2023/MECHA4400/MCHA4400/Labs/Week7/lab7/build/_deps/nanobench-subbuild/nanobench-populate-prefix/src/nanobench-populate-stamp/nanobench-populate-gitclone-lastrun.txt" IS_NEWER_THAN "D:/University Work/2023/MECHA4400/MCHA4400/Labs/Week7/lab7/build/_deps/nanobench-subbuild/nanobench-populate-prefix/src/nanobench-populate-stamp/nanobench-populate-gitinfo.txt")
  message(STATUS
    "Avoiding repeated git clone, stamp file is up to date: "
    "'D:/University Work/2023/MECHA4400/MCHA4400/Labs/Week7/lab7/build/_deps/nanobench-subbuild/nanobench-populate-prefix/src/nanobench-populate-stamp/nanobench-populate-gitclone-lastrun.txt'"
  )
  return()
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E rm -rf "D:/University Work/2023/MECHA4400/MCHA4400/Labs/Week7/lab7/build/_deps/nanobench-src"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: 'D:/University Work/2023/MECHA4400/MCHA4400/Labs/Week7/lab7/build/_deps/nanobench-src'")
endif()

# try the clone 3 times in case there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "C:/msys64/usr/bin/git.exe" 
            clone --no-checkout --depth 1 --no-single-branch --config "advice.detachedHead=false" "https://github.com/martinus/nanobench.git" "nanobench-src"
    WORKING_DIRECTORY "D:/University Work/2023/MECHA4400/MCHA4400/Labs/Week7/lab7/build/_deps"
    RESULT_VARIABLE error_code
  )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(STATUS "Had to git clone more than once: ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://github.com/martinus/nanobench.git'")
endif()

execute_process(
  COMMAND "C:/msys64/usr/bin/git.exe" 
          checkout "v4.3.11" --
  WORKING_DIRECTORY "D:/University Work/2023/MECHA4400/MCHA4400/Labs/Week7/lab7/build/_deps/nanobench-src"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: 'v4.3.11'")
endif()

set(init_submodules TRUE)
if(init_submodules)
  execute_process(
    COMMAND "C:/msys64/usr/bin/git.exe" 
            submodule update --recursive --init 
    WORKING_DIRECTORY "D:/University Work/2023/MECHA4400/MCHA4400/Labs/Week7/lab7/build/_deps/nanobench-src"
    RESULT_VARIABLE error_code
  )
endif()
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: 'D:/University Work/2023/MECHA4400/MCHA4400/Labs/Week7/lab7/build/_deps/nanobench-src'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy "D:/University Work/2023/MECHA4400/MCHA4400/Labs/Week7/lab7/build/_deps/nanobench-subbuild/nanobench-populate-prefix/src/nanobench-populate-stamp/nanobench-populate-gitinfo.txt" "D:/University Work/2023/MECHA4400/MCHA4400/Labs/Week7/lab7/build/_deps/nanobench-subbuild/nanobench-populate-prefix/src/nanobench-populate-stamp/nanobench-populate-gitclone-lastrun.txt"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: 'D:/University Work/2023/MECHA4400/MCHA4400/Labs/Week7/lab7/build/_deps/nanobench-subbuild/nanobench-populate-prefix/src/nanobench-populate-stamp/nanobench-populate-gitclone-lastrun.txt'")
endif()
