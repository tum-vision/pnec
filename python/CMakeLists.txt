# add_subdirectory(pybind11)
set(PYTHONVERSION 3.10)
set(PYBIND11_PYTHON_VERSION 3.10 CACHE STRING "")

add_subdirectory(pybind11)

include_directories(${PYTHON_INCLUDE_DIRS})
pybind11_add_module(pypnec pypnec.cpp)
target_link_libraries(pypnec PRIVATE pnec)

# Find where to install python libs.
execute_process(COMMAND
    ${PYTHON_EXECUTABLE} -c "import distutils.sysconfig; print('/'.join(distutils.sysconfig.get_python_lib().split('/')[-3:]))"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
set(PYTHON_INSTALL_DIR
    "${CMAKE_INSTALL_PREFIX}/${PYTHON_SITE_PACKAGES}"
    CACHE PATH "Path where to install pypnec")
install(TARGETS pypnec DESTINATION "${PYTHON_INSTALL_DIR}")
message(python executable ${PYTHON_EXECUTABLE})