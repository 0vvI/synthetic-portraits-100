# fitcore

fitcore fits a camera pose and the linear shape coefficients of a landmark model to 2-D landmarks
by Levenberg-Marquardt, with an analytic Jacobian and an optional Gaussian prior on the
coefficients. It is the C++17 counterpart of the numpy reference `fit/solver_ref.py`: both follow
`fit/SPEC.md`, which fixes the maths, the algorithm and the tests.

## Build and test

Needs CMake 3.18 or newer, a C++17 compiler and Eigen 3.4 or newer. From the repository root:

```sh
cmake -S fit/fitcore -B build/fitcore -DCMAKE_BUILD_TYPE=Release
cmake --build build/fitcore -j
ctest --test-dir build/fitcore --output-on-failure
```

With the Python module (needs pybind11, numpy and pytest in the Python you build for):

```sh
cmake -S fit/fitcore -B build/fitcore -DFITCORE_PYTHON=ON \
  -DPython_EXECUTABLE="$(which python)" -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build/fitcore -j
FITCORE_BUILD_DIR=build/fitcore python -m pytest fit/tests -q
```

`-DFITCORE_SANITIZE=ON` builds the library and the tests with AddressSanitizer and UBSan.
`-DFITCORE_BLAS=ON` lets Eigen hand the large matrix product to a BLAS library (Accelerate on macOS).

## API

The same functions in C++ (`include/fitcore/fitcore.hpp`, namespace `fitcore`) and in Python
(module `fitcore`). M is the number of landmarks and K the number of coefficients; K may be 0.

| Function | What it does |
|---|---|
| `exp_so3(omega)` | rotation matrix of a rotation vector |
| `residuals` | residual vector r of length 2M + K at a state (R, t, theta) |
| `residuals_and_jacobian` | r and the Jacobian J, (2M + K) x (6 + K) |
| `solve` | Levenberg-Marquardt fit from a start state: state, cost, accepted steps, status |
| `solve_batch` | `solve` for N sets of observations of one model, on several threads |

In Python the functions take the arguments of `fit/solver_ref.py` in the same order
(`P0, B, x, w, p, f, cx, cy, R, t, theta`) and return the same arrays and dicts, so a script can
swap one module for the other. A wrong shape raises `ValueError` naming the argument.

## How it is verified

| Id | Check | Where |
|---|---|---|
| T1 | `exp_so3` is a rotation; its two branches meet at 1e-12 rad | ctest |
| T2 | Jacobian against central differences | ctest |
| T3 | recovery of the true state from a distant start | ctest |
| T4 | with a prior, over ten seeds: every run ends converged, the cost never rises, the gradient vanishes | ctest |
| T5 | landmarks with weight 0 have no effect | ctest |
| K0 | pose-only fit, K = 0 | ctest |
| T6 | C++ against numpy: residuals, Jacobian and solutions | pytest |
| T7 | `solve_batch` against a loop of `solve` | pytest |

The C++ tests are in `tests/test_fitcore.cpp`; T6 and T7 are in `fit/tests/test_agreement.py`.
The workflow `.github/workflows/fitcore.yml` runs both on Ubuntu with Eigen 3.4.
