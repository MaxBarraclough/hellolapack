
cmake_minimum_required(VERSION 3.16.1)

include(CheckLibraryExists) # For check_library_exists
include(CheckIncludeFile)   # For check_include_file

# The project is written in C++ but we must also specify C so that we can use
# check_include_file to check for a C header file
project(
    hellolapack
    LANGUAGES  CXX C
    VERSION    0.1
)

set(CMAKE_CXX_STANDARD 17) # https://cmake.org/cmake/help/latest/guide/tutorial/index.html
set(CMAKE_CXX_STANDARD_REQUIRED True)


# https://stackoverflow.com/a/10055571/
IF (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    ADD_COMPILE_OPTIONS(-Wall -Wextra -pedantic) # -Werror
ELSEIF(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    ADD_COMPILE_OPTIONS(-Wall -Wextra -pedantic) # -Werror
ENDIF()

find_package(LAPACK REQUIRED)
# find_package(BLAS REQUIRED) # BLA_VENDOR Generic)


#######################################################################
# find_package LAPACK does not support LAPACKE, handle this ourselves #
#######################################################################

# https://cmake.org/cmake/help/latest/module/CheckIncludeFile.html
check_include_file(
    "lapacke.h"           # include
    HAVE_HEADER_LAPACKE_H # variable
)

IF (HAVE_HEADER_LAPACKE_H)
    MESSAGE(STATUS "lapacke.h header found. Continuing.")
ELSE()
    MESSAGE(FATAL_ERROR "lapacke.h header not found. This is fatal.")
ENDIF()

# https://cmake.org/cmake/help/latest/module/CheckLibraryExists.html
# Check specified library exists and given C function is available
check_library_exists(
    lapacke              # library
    LAPACKE_dgels        # function
    ""                   # location    "" means search default lib paths
    HAVE_LIBRARY_LAPACKE # variable
)

IF (HAVE_LIBRARY_LAPACKE)
    MESSAGE(STATUS "LAPACKE library found. Continuing.")
ELSE()
    MESSAGE(FATAL_ERROR "LAPACKE library not found. This is fatal.")
      #  This has the effect of terminating with error
ENDIF()

############################################################################
# END: find_package LAPACK does not support LAPACKE, handle this ourselves #
############################################################################


# MESSAGE (${BLAS_LIBRARIES})
# MESSAGE (${LAPACK_LIBRARIES})

#configure_file(HelloLapackConfig.h.in HelloLapackConfig.h)


add_executable(
    hellolapack
    "${CMAKE_SOURCE_DIR}/hellolapack.cpp"
)

target_include_directories(
    hellolapack PRIVATE "${CMAKE_SOURCE_DIR}"
)


# target_link_libraries(hellolapack PRIVATE ${CMAKE_DL_LIBS} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} )
target_link_libraries(hellolapack PRIVATE ${CMAKE_DL_LIBS} ${LAPACK_LIBRARIES} lapacke )
    # find_package LAPACK does not support LAPACKE, link lapacke directly by name
    # https://stackoverflow.com/a/77349214/

