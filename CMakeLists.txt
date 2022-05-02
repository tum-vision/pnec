cmake_minimum_required(VERSION 3.0)
project(pnec)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Make CMake aware of the cmake folder for local FindXXX scripts,
# append rather than set in case the user has passed their own
# additional paths via -D.
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(UpdateCacheVariable)

# ============================= basalt =============================
# eigen, opengv and sophus are found with basalt
# needed before the other so eigen3.4 is found in basalt/thirdparty/basalt-headers/thirdparty/eigen
# instead of local eigen install which is not necessary 3.4
add_subdirectory(third_party/basalt EXCLUDE_FROM_ALL)
# ============================= basalt =============================
find_package(Boost COMPONENTS filesystem REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(SuiteSparse)
find_package(Ceres REQUIRED)

# ============================= suitesparse =============================
add_definitions (-march=native)
if (SUITESPARSE_FOUND)
    # On Ubuntu the system install of SuiteSparse (v3.4.0) up to at least
    # Ubuntu 13.10 cannot be used to link shared libraries.
    if (BUILD_SHARED_LIBS AND
        SUITESPARSE_IS_BROKEN_SHARED_LINKING_UBUNTU_SYSTEM_VERSION)

        message(FATAL_ERROR "You are attempting to build Ceres as a shared "
        "library on Ubuntu using a system package install of SuiteSparse "
        "3.4.0. This package is broken and does not support the "
        "construction of shared libraries (you can still build Ceres as "
        " a static library).  If you wish to build a shared version of Ceres "
        "you should uninstall the system install of SuiteSparse "
        "(libsuitesparse-dev) and perform a source install of SuiteSparse "
        "(we recommend that you use the latest version), "
        "see http://ceres-solver.org/building.html for more information.")
    endif (BUILD_SHARED_LIBS AND
        SUITESPARSE_IS_BROKEN_SHARED_LINKING_UBUNTU_SYSTEM_VERSION)

        message("-- Found SuiteSparse ${SUITESPARSE_VERSION}, "
        "building with SuiteSparse.")
else (SUITESPARSE_FOUND)
    message(FATAL_ERROR "Unable to find SuiteSparse!")
endif (SUITESPARSE_FOUND)

link_directories(${SUITESPARSE_LIBRARY_DIR_HINTS})
# ============================= suitesparse =============================

enable_testing()

include_directories(
    include/odometry
    include/optimization
    include/common
    include/frames
    include/io
    include/features
    include/features/orb
    include/features/tracking
    include/rel_pose_estimation
    include/simulation
  ${EIGEN3_INCLUDE_DIR}
  ${SUITESPARSE_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/third_party/basalt/include
  ${PROJECT_SOURCE_DIR}/third_party/basalt/thirdparty/basalt-headers/include
  ${PROJECT_SOURCE_DIR}/third_party/basalt/thirdparty/basalt-headers/thirdparty/Sophus
  ${PROJECT_SOURCE_DIR}/third_party/basalt/thirdparty/basalt-headers/thirdparty/cereal/include
)

set( THIRD_PARTY_LIBS
    Eigen3::Eigen
    ${SUITESPARSE_LIBRARIES} 
    umfpack
    ${OpenCV_LIBRARIES}
    ${Boost_FILESYSTEM_LIBRARY} 
    -lboost_system
    opengv
    basalt
    ${Ceres_LIBRARIES}
)

# set file lists
file(GLOB_RECURSE ALL_LIBRARY_HDRS "include/*.h")
file(GLOB_RECURSE ALL_LIBRARY_SRCS "src/*.cc")

# remove files from lists
file(GLOB_RECURSE ALL_DOTFILES ".*/*")
if (ALL_DOTFILES)
    list(REMOVE_ITEM ALL_LIBRARY_HDRS ${ALL_DOTFILES})
    list(REMOVE_ITEM ALL_LIBRARY_SRCS ${ALL_DOTFILES})
endif()
file(GLOB_RECURSE MAIN_FILE "src/PNECMRO.cc")
if (MAIN_FILE)
    list(REMOVE_ITEM ALL_LIBRARY_SRCS ${MAIN_FILE})
endif()
file(GLOB_RECURSE EXPERIMENT_CREATION_FILE "src/create_sim_experiments.cc")
if (EXPERIMENT_CREATION_FILE)
    list(REMOVE_ITEM ALL_LIBRARY_SRCS ${EXPERIMENT_CREATION_FILE})
endif()
file(GLOB_RECURSE SIMULATION_FILE "src/run_simulation.cc")
if (SIMULATION_FILE)
    list(REMOVE_ITEM ALL_LIBRARY_SRCS ${SIMULATION_FILE})
endif()

add_library(${PROJECT_NAME} STATIC ${ALL_LIBRARY_HDRS} ${ALL_LIBRARY_SRCS})
target_link_libraries(${PROJECT_NAME} ${THIRD_PARTY_LIBS} opengv)

add_executable(pnec_vo src/pnec_vo.cc ${THIRD_PARTY_FILES})
target_link_libraries(pnec_vo ${THIRD_PARTY_LIBS} opengv ${PROJECT_NAME})

add_executable(create_experiments src/create_sim_experiments.cc ${THIRD_PARTY_FILES})
target_link_libraries(create_experiments ${THIRD_PARTY_LIBS} opengv ${PROJECT_NAME})

add_executable(run_simulation src/run_simulation.cc ${THIRD_PARTY_FILES})
target_link_libraries(run_simulation ${THIRD_PARTY_LIBS} opengv ${PROJECT_NAME})
