// Tests T1 to T5 of fit/SPEC.md section 6 for the C++ core, and K0, the pose-only fit (K = 0).
// Usage: fitcore_tests [ID] runs test ID (T1, T2, T3, T4, T5 or K0), or every test when no ID is
// given. It prints one line of measurements per test and exits non-zero if any check failed.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <Eigen/LU>

#include "fitcore/fitcore.hpp"

namespace {

using fitcore::MatXX;

int failed_checks = 0;

// Counts a failed check and prints where it is and the value it measured.
void record(bool ok, const char* condition, double measured, const char* file, int line) {
  if (!ok) {
    ++failed_checks;
    std::printf("  check failed at %s:%d: %s (measured %.3e)\n", file, line, condition, measured);
  }
}

}  // namespace

#define CHECK(condition, measured)                                                           \
  record(static_cast<bool>(condition), #condition, static_cast<double>(measured), __FILE__, \
         __LINE__)

namespace {

// Uniform and normal samples from std::mt19937_64. The standard fixes the engine's output but not
// the algorithms of std::uniform_real_distribution and std::normal_distribution, so the samples
// are drawn here and the test problems are the same with every standard library.
class Rng {
 public:
  explicit Rng(std::uint64_t seed) : engine_(seed) {}

  double uniform(double lo, double hi) {
    // The top 53 bits of the engine's output, scaled to [0, 1).
    return lo + (hi - lo) * static_cast<double>(engine_() >> 11) * 0x1.0p-53;
  }

  double normal() {  // Box-Muller
    const double u1 = 1.0 - uniform(0.0, 1.0);  // in (0, 1], so the log is finite
    const double u2 = uniform(0.0, 1.0);
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * kPi * u2);
  }

 private:
  static constexpr double kPi = 3.141592653589793;
  std::mt19937_64 engine_;
};

Eigen::VectorXd normal_vector(Rng& rng, Eigen::Index n) {
  Eigen::VectorXd v(n);
  for (Eigen::Index k = 0; k < n; ++k) {
    v(k) = rng.normal();
  }
  return v;
}

Eigen::VectorXd uniform_vector(Rng& rng, Eigen::Index n, double lo, double hi) {
  Eigen::VectorXd v(n);
  for (Eigen::Index k = 0; k < n; ++k) {
    v(k) = rng.uniform(lo, hi);
  }
  return v;
}

Eigen::Vector3d random_axis(Rng& rng) {
  return Eigen::Vector3d(normal_vector(rng, 3)).normalized();
}

// The synthetic problem of SPEC section 6 and its true state.
struct Synthetic {
  fitcore::Problem problem;  // w = 1, p = 0, x the exact projections of the true state
  fitcore::State truth;
};

// M landmarks uniform in a 0.2 m cube, B standard normal times 0.01, a true rotation of 0.3 rad
// about a random axis, t = (0.02, -0.01, 0.6), theta standard normal, f = 1500, (cx, cy) =
// (320, 240), and noise-free observations.
Synthetic make_synthetic(std::uint64_t seed, Eigen::Index M = 120, Eigen::Index K = 12) {
  Rng rng(seed);
  Synthetic s;
  fitcore::Problem& pr = s.problem;
  pr.f = 1500.0;
  pr.cx = 320.0;
  pr.cy = 240.0;
  pr.P0.resize(M, 3);
  for (Eigen::Index i = 0; i < M; ++i) {
    for (Eigen::Index c = 0; c < 3; ++c) {
      pr.P0(i, c) = rng.uniform(-0.1, 0.1);
    }
  }
  pr.B.resize(3 * M, K);
  for (Eigen::Index row = 0; row < 3 * M; ++row) {
    for (Eigen::Index k = 0; k < K; ++k) {
      pr.B(row, k) = 0.01 * rng.normal();
    }
  }
  s.truth.R = Eigen::AngleAxisd(0.3, random_axis(rng)).toRotationMatrix();
  s.truth.t = Eigen::Vector3d(0.02, -0.01, 0.6);
  s.truth.theta = normal_vector(rng, K);

  // The projections are computed here rather than by the code under test.
  pr.x.resize(M, 2);
  for (Eigen::Index i = 0; i < M; ++i) {
    const Eigen::Vector3d P =
        pr.P0.row(i).transpose() + pr.B.middleRows(3 * i, 3) * s.truth.theta;
    const Eigen::Vector3d Q = s.truth.R * P + s.truth.t;
    pr.x(i, 0) = pr.f * Q.x() / Q.z() + pr.cx;
    pr.x(i, 1) = pr.f * Q.y() / Q.z() + pr.cy;
  }
  pr.w = Eigen::VectorXd::Ones(M);
  pr.p = Eigen::VectorXd::Zero(K);
  return s;
}

// The T3 start: identity rotation, t = (0, 0, 0.5), theta = 0.
fitcore::State start_state(Eigen::Index K) {
  fitcore::State s;
  s.R.setIdentity();
  s.t = Eigen::Vector3d(0.0, 0.0, 0.5);
  s.theta = Eigen::VectorXd::Zero(K);
  return s;
}

template <typename Derived>
double max_abs(const Eigen::MatrixBase<Derived>& m) {
  return m.size() == 0 ? 0.0 : m.cwiseAbs().maxCoeff();
}

// Largest absolute difference between two states, over R, t and theta.
double state_difference(const fitcore::State& a, const fitcore::State& b) {
  return std::max({max_abs(a.R - b.R), max_abs(a.t - b.t), max_abs(a.theta - b.theta)});
}

std::string sci(double value) {
  char text[32];
  std::snprintf(text, sizeof text, "%.2e", value);
  return text;
}

std::string outcome(const fitcore::Result& result) {
  const char* names[] = {"converged", "max_iter", "stalled"};
  return std::string(names[static_cast<int>(result.status)]) + " after " +
         std::to_string(result.iterations) + " steps";
}

std::string test_T1() {
  Rng rng(1);
  std::vector<Eigen::Vector3d> axes = {Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitY(),
                                       Eigen::Vector3d::UnitZ()};
  for (int k = 0; k < 5; ++k) {
    axes.push_back(random_axis(rng));
  }
  double orthogonality = 0.0, determinant = 0.0, branch_gap = 0.0;
  for (const Eigen::Vector3d& axis : axes) {
    for (const double angle : {0.0, 1e-13, 1e-6, 1.0, 3.0}) {
      const Eigen::Matrix3d R = fitcore::exp_so3(angle * axis);
      orthogonality =
          std::max(orthogonality, max_abs(R.transpose() * R - Eigen::Matrix3d::Identity()));
      determinant = std::max(determinant, std::abs(R.determinant() - 1.0));
    }
    // Just below |omega| = 1e-12 exp_so3 returns I + W, just above it uses the full formula.
    const Eigen::Matrix3d below = fitcore::exp_so3(1e-12 * (1.0 - 1e-9) * axis);
    const Eigen::Matrix3d above = fitcore::exp_so3(1e-12 * (1.0 + 1e-9) * axis);
    branch_gap = std::max(branch_gap, max_abs(below - above));
  }
  CHECK(orthogonality <= 1e-14, orthogonality);
  CHECK(determinant <= 1e-14, determinant);
  CHECK(branch_gap <= 1e-12, branch_gap);
  return "exp_so3: max |R^T R - I| " + sci(orthogonality) + ", max |det R - 1| " +
         sci(determinant) + ", branch gap at 1e-12 rad " + sci(branch_gap);
}

// The state moved by h along unknown j, in the column order of J: [d_omega, d_t, d_theta].
fitcore::State moved(fitcore::State s, Eigen::Index j, double h) {
  if (j < 3) {
    s.R = fitcore::exp_so3(h * Eigen::Vector3d::Unit(j)) * s.R;
  } else if (j < 6) {
    s.t(j - 3) += h;
  } else {
    s.theta(j - 6) += h;
  }
  return s;
}

std::string test_T2() {
  Synthetic s = make_synthetic(0);
  fitcore::Problem& pr = s.problem;
  Rng rng(1000);
  fitcore::State state;
  state.R = Eigen::AngleAxisd(0.05, random_axis(rng)).toRotationMatrix() * s.truth.R;
  state.t = s.truth.t + 0.01 * normal_vector(rng, 3);
  state.theta = s.truth.theta + 0.1 * normal_vector(rng, pr.B.cols());
  pr.w = uniform_vector(rng, pr.P0.rows(), 0.5, 1.5);
  pr.p = uniform_vector(rng, pr.B.cols(), 0.1, 1.0);

  Eigen::VectorXd r, r_plus, r_minus;
  MatXX J;
  fitcore::residuals_and_jacobian(pr, state, r, J);
  const double h = 1e-6;
  MatXX J_fd(J.rows(), J.cols());
  for (Eigen::Index j = 0; j < J.cols(); ++j) {
    fitcore::residuals(pr, moved(state, j, h), r_plus);
    fitcore::residuals(pr, moved(state, j, -h), r_minus);
    J_fd.col(j) = (r_plus - r_minus) / (2.0 * h);
  }
  const double error = max_abs(J - J_fd) / max_abs(J);
  CHECK(error <= 1e-6, error);
  return "Jacobian: max |J - central differences| / max |J| " + sci(error);
}

std::string test_T3() {
  const Synthetic s = make_synthetic(0);
  const fitcore::Result result = fitcore::solve(s.problem, start_state(s.problem.B.cols()));
  const double error = state_difference(result.state, s.truth);
  CHECK(result.status == fitcore::Status::Converged, static_cast<int>(result.status));
  CHECK(error <= 1e-8, error);
  return "recovery: " + outcome(result) + ", max error in R, t, theta " + sci(error);
}

std::string test_T4() {
  // Ten seeds: at the minimum the last accepted step depends on rounding, and every ending must
  // still be reported as converged.
  double largest_change = -std::numeric_limits<double>::infinity();
  double worst_gradient = 0.0;
  int most_steps = 0;
  for (std::uint64_t seed = 0; seed < 10; ++seed) {
    Synthetic s = make_synthetic(seed);
    s.problem.p.setConstant(0.5);
    const fitcore::State start = start_state(s.problem.B.cols());
    const fitcore::Result result = fitcore::solve(s.problem, start);
    CHECK(result.status == fitcore::Status::Converged, static_cast<int>(result.status));
    most_steps = std::max(most_steps, result.iterations);

    // solve() keeps no history, but a run limited to k iterations stops after the k-th accepted
    // step, so the costs of such runs are the cost history.
    fitcore::Options options;
    options.max_iter = 0;
    double previous = fitcore::solve(s.problem, start, options).cost;
    for (int k = 1; k <= result.iterations; ++k) {
      options.max_iter = k;
      const double cost = fitcore::solve(s.problem, start, options).cost;
      largest_change = std::max(largest_change, cost - previous);
      previous = cost;
    }
    CHECK(previous == result.cost, previous - result.cost);

    // The gradient, relative to the sizes of J and r: a cost comparison in double precision
    // cannot push it much below 1e-8.
    Eigen::VectorXd r;
    MatXX J;
    fitcore::residuals_and_jacobian(s.problem, result.state, r, J);
    const Eigen::VectorXd gradient = J.transpose() * r;
    worst_gradient = std::max(worst_gradient, max_abs(gradient) / (J.norm() * r.norm()));
  }
  CHECK(largest_change <= 0.0, largest_change);
  CHECK(worst_gradient <= 1e-6, worst_gradient);
  return "prior: 10 seeds converged in at most " + std::to_string(most_steps) +
         " steps, largest cost change per step " + sci(largest_change) +
         ", max |J^T r| / (|J| |r|) " + sci(worst_gradient);
}

std::string test_T5() {
  const Synthetic s = make_synthetic(0);
  const fitcore::Problem& full = s.problem;
  const Eigen::Index M = full.P0.rows();

  // Every fourth landmark (30 of 120) gets w = 0 and garbage x in one problem and is removed from
  // the other.
  Rng rng(2000);
  fitcore::Problem zeroed = full;
  std::vector<Eigen::Index> kept;
  for (Eigen::Index i = 0; i < M; ++i) {
    if (i % 4 == 0) {
      zeroed.w(i) = 0.0;
      zeroed.x(i, 0) = rng.uniform(-1e4, 1e4);
      zeroed.x(i, 1) = rng.uniform(-1e4, 1e4);
    } else {
      kept.push_back(i);
    }
  }
  fitcore::Problem cut = full;
  const auto n_kept = static_cast<Eigen::Index>(kept.size());
  cut.P0.resize(n_kept, 3);
  cut.B.resize(3 * n_kept, full.B.cols());
  cut.x.resize(n_kept, 2);
  cut.w.resize(n_kept);
  for (Eigen::Index j = 0; j < n_kept; ++j) {
    const Eigen::Index i = kept[static_cast<std::size_t>(j)];
    cut.P0.row(j) = full.P0.row(i);
    cut.B.middleRows<3>(3 * j) = full.B.middleRows<3>(3 * i);
    cut.x.row(j) = full.x.row(i);
    cut.w(j) = full.w(i);
  }

  const fitcore::State start = start_state(full.B.cols());
  const fitcore::Result a = fitcore::solve(zeroed, start);
  const fitcore::Result b = fitcore::solve(cut, start);
  const double difference = std::max(state_difference(a.state, b.state), std::abs(a.cost - b.cost));
  CHECK(difference <= 1e-10, difference);
  return "zero weights: " + std::to_string(M - n_kept) + " landmarks out, max difference " +
         "in R, t, theta, cost " + sci(difference);
}

std::string test_K0() {
  const Synthetic s = make_synthetic(0, 120, 0);
  const fitcore::State start = start_state(0);
  Eigen::VectorXd r;
  MatXX J;
  fitcore::residuals_and_jacobian(s.problem, start, r, J);
  CHECK(r.size() == 240, r.size());
  CHECK(J.rows() == 240, J.rows());
  CHECK(J.cols() == 6, J.cols());

  const fitcore::Result result = fitcore::solve(s.problem, start);
  const double error = state_difference(result.state, s.truth);
  CHECK(result.status == fitcore::Status::Converged, static_cast<int>(result.status));
  CHECK(result.state.theta.size() == 0, result.state.theta.size());
  CHECK(error <= 1e-8, error);
  return "pose only: J " + std::to_string(J.rows()) + " x " + std::to_string(J.cols()) + ", " +
         outcome(result) + ", max error in R, t " + sci(error);
}

}  // namespace

int main(int argc, char** argv) {
  const std::pair<const char*, std::string (*)()> tests[] = {
      {"T1", test_T1}, {"T2", test_T2}, {"T3", test_T3},
      {"T4", test_T4}, {"T5", test_T5}, {"K0", test_K0}};
  const std::string wanted = argc > 1 ? argv[1] : "";
  int ran = 0;
  for (const auto& [id, run] : tests) {
    if (!wanted.empty() && wanted != id) {
      continue;
    }
    const int failed_before = failed_checks;
    const std::string measured = run();
    std::printf("%s %s: %s\n", id, failed_checks == failed_before ? "pass" : "FAIL",
                measured.c_str());
    ++ran;
  }
  if (ran == 0) {
    std::printf("unknown test id %s; the ids are T1, T2, T3, T4, T5 and K0\n", wanted.c_str());
    return 2;
  }
  return failed_checks == 0 ? 0 : 1;
}
