## `Running tests`

If built with the `NOA_BUILD_TESTS` config option, the project builds and installs the ``noa_tests`` executable.

To run the tests:
1. Clone the [noa-data](https://github.com/thomasfrosio/noa-data) repository, run the setup script to generate the assets.
   Set the environmental variable ``NOA_DATA_PATH`` to the repository path.
   See instructions in [noa-data](https://github.com/thomasfrosio/noa-data).

2. Build the library and the executable if not done already.
   See instructions [here](001_build.md).
   **Building the tests with the CUDA backend can take a lot of memory, so we recommend only using a few threads (e.g. `-j 5`) to make sure the system doesn't run out of memory...** (nvcc isn't particularly lean on the number of subprocesses its spawns...)

3. To run all tests, simply run `./noa_tests` (there are a lot of tests, this can take a few minutes to run).
   The command line arguments are parsed using `Catch2`.
   See instructions in [Catch2](https://github.com/catchorg/Catch2/blob/v2.x/docs/command-line.md).

Note: Some tests may fail due to small floating-point precision errors. We try to have epsilons as low as possible, and in some cases it may be a bit too low, causing tests to fail from time to time. The test suits report stats about the failure, `max_abs_diff` should be the variable to follow closely.
