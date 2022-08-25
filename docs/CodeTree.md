## `Directory hierarchy`

- `src/`: The source directory contains the main headers that users should include. It is also the only place that
  is supposed to include headers from the "common" headers _and_ the backends.
- `src/common`: Contains everything that can be safely included by the backends. Its content is available in the main
  headers so users should not have to include anything from this directory directly (although there's nothing
  preventing them to do so).
- `src/cpu`: The CPU backend. Can only include headers from `src/common`.
- `src/gpu`: The GPU backends. Can only include headers from `src/common`.
