cmake_minimum_required(VERSION 3.0.2)
project(roboptim_tutorials)

add_compile_options(-std=c++11)
set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED)

find_package(PkgConfig)
pkg_check_modules(roboptim-core roboptim-core REQUIRED)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES roboptim_tutorials
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(
 include
 ${catkin_INCLUDE_DIRS}
 ${roboptim-core_INCLUDE_DIRS}
)

# 相対パスを絶対パスに直す
set(roboptim-core_LIBRARIES_ABS)
foreach(lib ${roboptim-core_LIBRARIES})
  find_library(${lib}_abs NAMES ${lib} PATHS ${roboptim-core_LIBRARY_DIRS})
  set(roboptim-core_LIBRARIES_ABS ${roboptim-core_LIBRARIES_ABS} ${${lib}_abs})
endforeach(lib)

add_executable(ipopt_tutorial src/ipopt_tutorial.cpp)
target_link_libraries(ipopt_tutorial
  ${catkin_LIBRARIES}
  ${roboptim-core_LIBRARIES_ABS}
  )

