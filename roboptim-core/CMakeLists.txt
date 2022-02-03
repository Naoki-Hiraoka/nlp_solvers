cmake_minimum_required(VERSION 3.0.2)
project(roboptim-core)

set(CMAKE_BUILD_TYPE Release)

include(ExternalProject)

externalproject_add(roboptim-core
  GIT_REPOSITORY https://github.com/roboptim/roboptim-core
  TIMEOUT 120
  CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
  INSTALL_COMMAND echo "install"
  )

install(CODE "execute_process(COMMAND make install WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/roboptim-core-prefix/src/roboptim-core-build)")

find_package(catkin)
catkin_add_env_hooks(99.roboptim-core SHELLS bash zsh
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/env-hooks)
