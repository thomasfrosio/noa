# Find and populate the project with the necessary FFTW libraries (float and double only).
#
# The workflow
# ============
#
# - A) If the CMake variable NOA_FFTW_USE_OWN is true:
#       - 1) If the system environmental variables NOA_FFTW_LIBRARIES and NOA_FFTW_INCLUDE are set:
#            CMake will try to find and use your fftw libraries (exclusively) under these path.
#            If this step fails, an error will be reported.
#       - 2) If the system environmental variables NOA_FFTW_LIBRARIES and NOA_FFTW_INCLUDE are NOT set:
#            CMake will try to find and use your fftw libraries by itself using the default paths.
#            If this step fails, an error will be reported.
#
# - B) If the CMake variable NOA_FFTW_USE_OWN is false:
#       - 1) The fftw libraries will be fetched from the fftw3 website and made available at generation time.
#
# The option NOA_FFTW_USE_STATIC_LIBS is checked by each step and forces CMake to use the static libraries of fftw.
# This file generates the imported libraries:
#   - NOA_FFTW::Float
#   - NOA_FFTW::Double
# These can be shared or static depending on NOA_FFTW_USE_STATIC_LIBS.
#

# Find it...
if (NOA_FFTW_USE_OWN)
    message(STATUS "Try to find fft3w...")
    find_package(FFTW)

# ... or fetch it.
else ()
    message(FATAL_ERROR "NOA_FFTW_USE_OWN=OFF is currently not supported. fftw must be installed in your system.")

endif ()
message("")
