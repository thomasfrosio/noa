## `Running tests`

The library will install the ``noa_tests`` executable.

To run the tests:
1. Clone the [noa-data](https://github.com/ffyr2w/noa-data) repository, run the setup script to generate the assets.
   Set the environmental variable ``NOA_DATA_PATH`` to the repository path.
   See instructions in [noa-data](https://github.com/ffyr2w/noa-data).

2. Build the library and the executable if not done already.
   See instructions in [Build.md](Build.md)

3. To run all tests, simply run `./noa_tests`.
   The command line arguments are parsed using `Catch2`.
   See instructions in [Catch2](https://github.com/catchorg/Catch2/blob/v2.x/docs/command-line.md).
