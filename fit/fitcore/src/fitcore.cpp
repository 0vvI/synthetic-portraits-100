// The residuals and analytic Jacobian of SPEC section 3 and the Levenberg-Marquardt loop of
// SPEC section 4. The numpy reference fit/solver_ref.py follows the same lines.
#include "fitcore/fitcore.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Cholesky>

namespace fitcore {
namespace {

using Mat23 = Eigen::Matrix<double, 2, 3>;

// skew(a) * b == a.cross(b).
Eigen::Matrix3d skew(const Eigen::Vector3d& a) {
  Eigen::Matrix3d S;
  S << 0.0, -a.z(), a.y(),
       a.z(), 0.0, -a.x(),
       -a.y(), a.x(), 0.0;
  return S;
}

void expect_size(const std::string& name, Eigen::Index size, const char* expected_name,
                 Eigen::Index expected) {
  if (size != expected) {
    throw std::invalid_argument("fitcore: " + name + " is " + std::to_string(size) +
                                ", expected " + expected_name + " = " + std::to_string(expected));
  }
}

// M = P0.rows() and K = B.cols() fix every other size.
void check_model(const Problem& model) {
  expect_size("B.rows()", model.B.rows(), "3 * P0.rows()", 3 * model.P0.rows());
  expect_size("p.size()", model.p.size(), "B.cols()", model.B.cols());
}

void check_sizes(const Problem& problem, const State& state) {
  check_model(problem);
  expect_size("x.rows()", problem.x.rows(), "P0.rows()", problem.P0.rows());
  expect_size("w.size()", problem.w.size(), "P0.rows()", problem.P0.rows());
  expect_size("theta.size()", state.theta.size(), "B.cols()", problem.B.cols());
}

// Landmark i in camera space: Q_i = R (P0[i] + B[i] theta) + t.
Eigen::Vector3d camera_point(const Problem& problem, const State& state, Eigen::Index i) {
  const Eigen::Vector3d P =
      problem.P0.row(i).transpose() + problem.B.middleRows<3>(3 * i) * state.theta;
  return state.R * P + state.t;
}

// residuals() without the size checks, for the solver's inner loop.
void eval_residuals(const Problem& problem, const State& state, Eigen::VectorXd& r) {
  const Eigen::Index M = problem.P0.rows();
  const Eigen::Index K = problem.B.cols();
  r.resize(2 * M + K);
  for (Eigen::Index i = 0; i < M; ++i) {
    const Eigen::Vector3d Q = camera_point(problem, state, i);
    r(2 * i) = problem.w(i) * (problem.f * Q.x() / Q.z() + problem.cx - problem.x(i, 0));
    r(2 * i + 1) = problem.w(i) * (problem.f * Q.y() / Q.z() + problem.cy - problem.x(i, 1));
  }
  r.tail(K) = problem.p.cwiseProduct(state.theta);
}

// residuals_and_jacobian() without the size checks. r comes from eval_residuals, so it is
// bit-identical to the residuals() of the same state.
void eval_residuals_and_jacobian(const Problem& problem, const State& state, Eigen::VectorXd& r,
                                 MatXX& J) {
  eval_residuals(problem, state, r);
  const Eigen::Index M = problem.P0.rows();
  const Eigen::Index K = problem.B.cols();
  J.resize(2 * M + K, 6 + K);
  for (Eigen::Index i = 0; i < M; ++i) {
    const Eigen::Vector3d Q = camera_point(problem, state, i);
    // D_i = d(u_i, v_i) / dQ_i.
    const double s = problem.f / Q.z();
    Mat23 D;
    D << s, 0.0, s * (-Q.x() / Q.z()),
         0.0, s, s * (-Q.y() / Q.z());
    const Mat23 WD = problem.w(i) * D;
    // Rows 2i and 2i + 1: w_i D_i [-skew(Q_i - t) | I_3 | R B_i]. The theta block is formed as
    // (WD R) B_i, which takes 6K multiplications where WD (R B_i) would take 15K.
    J.block<2, 3>(2 * i, 0).noalias() = -WD * skew(Q - state.t);
    J.block<2, 3>(2 * i, 3) = WD;
    J.block(2 * i, 6, 2, K).noalias() = (WD * state.R) * problem.B.middleRows<3>(3 * i);
  }
  // Prior rows: [0 | 0 | diag(p)].
  J.bottomRows(K).setZero();
  J.bottomRightCorner(K, K).diagonal() = problem.p;
}

}  // namespace

Eigen::Matrix3d exp_so3(const Eigen::Vector3d& omega) {
  const double a = omega.norm();
  const Eigen::Matrix3d W = skew(omega);
  // The first-order form avoids 0 / 0 at omega = 0; below 1e-12 the terms it drops are under 1e-24.
  if (a < 1e-12) {
    return Eigen::Matrix3d::Identity() + W;
  }
  return Eigen::Matrix3d::Identity() + (std::sin(a) / a) * W +
         ((1.0 - std::cos(a)) / (a * a)) * (W * W);
}

void residuals(const Problem& problem, const State& state, Eigen::VectorXd& r) {
  check_sizes(problem, state);
  eval_residuals(problem, state, r);
}

void residuals_and_jacobian(const Problem& problem, const State& state, Eigen::VectorXd& r,
                            MatXX& J) {
  check_sizes(problem, state);
  eval_residuals_and_jacobian(problem, state, r, J);
}

Result solve(const Problem& problem, const State& init, const Options& options) {
  check_sizes(problem, init);
  const Eigen::Index n = 6 + problem.B.cols();  // number of unknowns

  Result result;
  result.state = init;
  State& state = result.state;
  State cand = init;
  Eigen::VectorXd r, r_cand, g(n), delta(n);
  MatXX J;
  Eigen::MatrixXd A(n, n), Ad(n, n);
  Eigen::LLT<Eigen::MatrixXd> llt(n);

  double lam = options.lam0;
  eval_residuals_and_jacobian(problem, state, r, J);
  double cost = 0.5 * r.squaredNorm();
  const Eigen::Index landmark_rows = 2 * problem.P0.rows();
  for (int it = 0; it < options.max_iter; ++it) {
    // A = J^T J is where the time goes, so it is not formed as a plain product. A is symmetric
    // and the factorization reads its lower triangle only, so only that half is computed; and
    // the prior rows of J are diag(p), so they add p^2 to the diagonal without a product.
    A.setZero();
    A.selfadjointView<Eigen::Lower>().rankUpdate(J.topRows(landmark_rows).transpose());
    A.diagonal().tail(n - 6) += problem.p.cwiseAbs2();
    g.noalias() = J.transpose() * r;
    bool accepted = false;
    bool at_minimum = false;
    double rel = 0.0;
    while (true) {
      Ad = A;
      Ad.diagonal() += lam * A.diagonal().cwiseMax(1e-12);
      // Ad is symmetric positive definite, so a Cholesky factorization solves it. If rounding
      // breaks the factorization there is no step to try, and the damping is raised as after
      // any rejected step.
      llt.compute(Ad);
      if (llt.info() == Eigen::Success) {
        delta = llt.solve(-g);
        cand.R = exp_so3(delta.head<3>()) * state.R;
        cand.t = state.t + delta.segment<3>(3);
        cand.theta = state.theta + delta.tail(n - 6);
        eval_residuals(problem, cand, r_cand);
        const double cost_cand = 0.5 * r_cand.squaredNorm();
        if (std::isfinite(cost_cand) && cost_cand < cost) {
          rel = (cost - cost_cand) / std::max(cost, 1e-300);
          state = cand;
          cost = cost_cand;
          lam = std::max(lam / 10, 1e-15);
          accepted = true;
          break;
        }
        // A rejected step this small means the cost cannot be lowered any further in floating
        // point: the state is at the minimum, not stuck. A step that is not finite is not small.
        if (delta.allFinite() && delta.cwiseAbs().maxCoeff() < options.xtol) {
          at_minimum = true;
          break;
        }
      }
      lam = lam * 10;
      if (lam > 1e12) {
        break;
      }
    }
    if (at_minimum) {
      result.status = Status::Converged;
      break;
    }
    if (!accepted) {
      result.status = Status::Stalled;
      break;
    }
    ++result.iterations;
    eval_residuals_and_jacobian(problem, state, r, J);
    if (rel < options.ftol || delta.cwiseAbs().maxCoeff() < options.xtol) {
      result.status = Status::Converged;
      break;
    }
  }
  result.cost = cost;
  return result;
}

std::vector<Result> solve_batch(const Problem& model, const std::vector<MatX2>& xs,
                                const std::vector<Eigen::VectorXd>& ws,
                                const std::vector<State>& inits, const Options& opt,
                                int n_threads) {
  check_model(model);
  const auto count = static_cast<Eigen::Index>(xs.size());
  expect_size("ws.size()", static_cast<Eigen::Index>(ws.size()), "xs.size()", count);
  expect_size("inits.size()", static_cast<Eigen::Index>(inits.size()), "xs.size()", count);
  for (std::size_t n = 0; n < xs.size(); ++n) {
    const std::string at = "[" + std::to_string(n) + "]";
    expect_size("xs" + at + ".rows()", xs[n].rows(), "P0.rows()", model.P0.rows());
    expect_size("ws" + at + ".size()", ws[n].size(), "P0.rows()", model.P0.rows());
    expect_size("inits" + at + ".theta.size()", inits[n].theta.size(), "B.cols()",
                model.B.cols());
  }

  std::vector<Result> results(xs.size());
  const std::size_t hardware = std::max(1u, std::thread::hardware_concurrency());
  const std::size_t workers =
      std::min(n_threads > 0 ? static_cast<std::size_t>(n_threads) : hardware, xs.size());

  // Each worker takes the next fit from a shared counter and runs a plain solve() on its own copy
  // of the model, so every result equals that of a separate solve() call.
  std::atomic<std::size_t> next{0};
  std::vector<std::exception_ptr> errors(workers);
  const auto work = [&](std::size_t k) {
    try {
      Problem problem = model;
      for (std::size_t n = next++; n < xs.size(); n = next++) {
        problem.x = xs[n];
        problem.w = ws[n];
        results[n] = solve(problem, inits[n], opt);
      }
    } catch (...) {
      errors[k] = std::current_exception();  // rethrown below, in the calling thread
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(workers);
  try {
    for (std::size_t k = 0; k < workers; ++k) {
      threads.emplace_back(work, k);
    }
  } catch (...) {
    // A thread that fails to start must not leave the started ones unjoined (std::terminate).
    for (std::thread& thread : threads) {
      thread.join();
    }
    throw;
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
  for (const std::exception_ptr& error : errors) {
    if (error) {
      std::rethrow_exception(error);
    }
  }
  return results;
}

}  // namespace fitcore
