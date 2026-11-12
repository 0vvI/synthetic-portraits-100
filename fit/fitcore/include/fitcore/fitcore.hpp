// fitcore: Levenberg-Marquardt fit of a camera pose and linear shape coefficients to 2-D landmarks.
// The maths, the algorithm and the tests are fixed by fit/SPEC.md, sections 3 to 6. M is the number
// of landmarks and K the number of coefficients; K may be 0, which gives a pose-only fit.
#pragma once

#include <vector>

#include <Eigen/Core>

namespace fitcore {

using MatX3 = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using MatX2 = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;
using MatXX = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// The model and the observations of one fit, in camera axes (x right, y down, z forward).
struct Problem {
  MatX3 P0;                      // (M, 3) landmark positions at theta = 0, metres
  MatXX B;                       // (3M, K) linear basis: row 3*i + c holds B[i, c, :]
  MatX2 x;                       // (M, 2) observed landmark pixels
  Eigen::VectorXd w;             // (M) landmark weights >= 0; 0 leaves a landmark out
  Eigen::VectorXd p;             // (K) prior weights; the prior residual is p[k] * theta[k]
  double f = 0, cx = 0, cy = 0;  // focal length and principal point, pixels
};

// The unknowns: landmark i sits at Q_i = R (P0[i] + B[i] theta) + t in camera space.
struct State {
  Eigen::Matrix3d R;      // rotation, in SO(3)
  Eigen::Vector3d t;      // translation, metres
  Eigen::VectorXd theta;  // (K) coefficients
};

// Settings of the Levenberg-Marquardt loop (SPEC section 4).
struct Options {
  int max_iter = 100;   // most outer iterations
  double lam0 = 1e-3;   // initial damping
  double ftol = 1e-14;  // converged when an accepted step lowers the cost by a smaller fraction
  double xtol = 1e-12;  // or when every entry of a step, accepted or rejected, is smaller
};

enum class Status {
  Converged = 0,  // an accepted step met ftol or xtol, or a rejected step was smaller than xtol
  MaxIter = 1,    // max_iter steps were accepted without meeting either
  Stalled = 2,    // no damping up to 1e12 lowered the cost, with steps that were not small
};

struct Result {
  State state;                      // the fitted state
  double cost = 0;                  // 0.5 * |r|^2 at state
  int iterations = 0;               // number of accepted steps
  Status status = Status::MaxIter;  // why the loop stopped
};

// Rotation matrix for the rotation vector omega: I + W below |omega| = 1e-12, else the Rodrigues
// formula, with W = skew(omega).
Eigen::Matrix3d exp_so3(const Eigen::Vector3d& omega);

// Residuals r (2M + K) at state: w[i] * (u_i - x[i, 0]) and w[i] * (v_i - x[i, 1]) for each
// landmark i, then p[k] * theta[k]. Throws std::invalid_argument on inconsistent sizes.
void residuals(const Problem& problem, const State& state, Eigen::VectorXd& r);

// Residuals r (2M + K) and Jacobian J (2M + K, 6 + K) with columns [d_omega (3), d_t (3),
// d_theta (K)]; d_omega perturbs the rotation on the left, R <- exp_so3(d_omega) R. Throws
// std::invalid_argument on inconsistent sizes.
void residuals_and_jacobian(const Problem& problem, const State& state, Eigen::VectorXd& r,
                            MatXX& J);

// Levenberg-Marquardt fit from init, by the algorithm of SPEC section 4. Throws
// std::invalid_argument on inconsistent sizes.
Result solve(const Problem& problem, const State& init, const Options& options = {});

// N fits of one model: fit n uses the observations xs[n] (M, 2) and ws[n] (M) and starts from
// inits[n]. The model supplies P0, B, p, f, cx and cy; its own x and w are ignored. Runs on
// n_threads threads (hardware concurrency when n_threads <= 0). The results are in input order and
// identical to separate solve() calls. Throws std::invalid_argument on inconsistent sizes.
std::vector<Result> solve_batch(const Problem& model, const std::vector<MatX2>& xs,
                                const std::vector<Eigen::VectorXd>& ws,
                                const std::vector<State>& inits, const Options& opt = {},
                                int n_threads = 0);

}  // namespace fitcore
