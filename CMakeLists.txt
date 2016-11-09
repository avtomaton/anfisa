cmake_minimum_required(VERSION 2.8)

project(anfisa-minimal)

include(CheckIncludeFile)
include(ExternalProject)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if (NOT TARGET aifil-utils-common AND NOT NO_UTILS)
	ExternalProject_Add(
		aifil-utils
		PREFIX aifil-utils
		GIT_REPOSITORY "https://github.com/avtomaton/toolchain-cpp.git"
		GIT_TAG "master"
		INSTALL_COMMAND "")
	ExternalProject_Get_property(aifil-utils source_dir)
	include_directories(${source_dir})
endif()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(OBJ_UTILS
	core/io-structures.cpp
	core/io-structures.hpp
	core/raw-structures.hpp
	classifier/classifier.hpp
)

add_library(anfisa-minimal ${OBJ_UTILS})
