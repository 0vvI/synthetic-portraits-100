// Python module fitcore: the C++ core with the argument order, keyword names, defaults and return
// values of the numpy reference fit/solver_ref.py (SPEC section 5), plus solve_batch.
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "fitcore/fitcore.hpp"

namespace py = pybind11;

namespace {

// Any array-like argument as a C-ordered float64 array; pybind11 copies only when it must.
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
using RowMajor33 = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;

// A shape as numpy prints it, with -1 printed as *: "(120, 3, *)".
std::string shape_text(const py::ssize_t* dims, std::size_t ndim) {
  std::string text = "(";
  for (std::size_t d = 0; d < ndim; ++d) {
    text += (d > 0 ? ", " : "") + (dims[d] < 0 ? std::string("*") : std::to_string(dims[d]));
  }
  return text + (ndim == 1 ? ",)" : ")");
}

// Raises ValueError unless `a` has the shape `dims`, in which -1 matches any extent. `label` names
// the dimensions as SPEC section 3 does, for example "(M, 3, K)".
void require_shape(const Array& a, const char* name, const char* label,
                   const std::vector<py::ssize_t>& dims) {
  bool ok = a.ndim() == static_cast<py::ssize_t>(dims.size());
  for (std::size_t d = 0; ok && d < dims.size(); ++d) {
    ok = dims[d] < 0 || a.shape(static_cast<py::ssize_t>(d)) == dims[d];
  }
  if (!ok) {
    const std::string expected = shape_text(dims.data(), dims.size());
    throw py::value_error(std::string(name) + " must have shape " + label +
                          (expected == label ? "" : " = " + expected) + ", got " +
                          shape_text(a.shape(), static_cast<std::size_t>(a.ndim())));
  }
}

// The model of a fit, copied out of the numpy arguments; x and w are left empty.
fitcore::Problem make_model(const Array& P0, const Array& B, const Array& p, double f, double cx,
                            double cy) {
  require_shape(P0, "P0", "(M, 3)", {-1, 3});
  const py::ssize_t M = P0.shape(0);
  require_shape(B, "B", "(M, 3, K)", {M, 3, -1});
  const py::ssize_t K = B.shape(2);
  require_shape(p, "p", "(K,)", {K});
  fitcore::Problem model;
  model.P0 = Eigen::Map<const fitcore::MatX3>(P0.data(), M, 3);
  // B (M, 3, K) in C order has the memory layout of the (3M, K) row-major matrix of fitcore.hpp.
  model.B = Eigen::Map<const fitcore::MatXX>(B.data(), 3 * M, K);
  model.p = Eigen::Map<const Eigen::VectorXd>(p.data(), K);
  model.f = f;
  model.cx = cx;
  model.cy = cy;
  return model;
}

fitcore::Problem make_problem(const Array& P0, const Array& B, const Array& x, const Array& w,
                              const Array& p, double f, double cx, double cy) {
  fitcore::Problem problem = make_model(P0, B, p, f, cx, cy);
  const py::ssize_t M = problem.P0.rows();
  require_shape(x, "x", "(M, 2)", {M, 2});
  require_shape(w, "w", "(M,)", {M});
  problem.x = Eigen::Map<const fitcore::MatX2>(x.data(), M, 2);
  problem.w = Eigen::Map<const Eigen::VectorXd>(w.data(), M);
  return problem;
}

fitcore::State make_state(const Array& R, const Array& t, const Array& theta, py::ssize_t K) {
  require_shape(R, "R", "(3, 3)", {3, 3});
  require_shape(t, "t", "(3,)", {3});
  require_shape(theta, "theta", "(K,)", {K});
  fitcore::State state;
  state.R = Eigen::Map<const RowMajor33>(R.data());
  state.t = Eigen::Map<const Eigen::Vector3d>(t.data());
  state.theta = Eigen::Map<const Eigen::VectorXd>(theta.data(), K);
  return state;
}

// A new C-ordered float64 array of the given shape holding a copy of m.
template <typename Derived>
py::array_t<double> to_numpy(const Eigen::MatrixBase<Derived>& m, std::vector<py::ssize_t> shape) {
  py::array_t<double> out(std::move(shape));
  Eigen::Map<fitcore::MatXX>(out.mutable_data(), m.rows(), m.cols()) = m;
  return out;
}

const char* status_name(fitcore::Status status) {
  switch (status) {
    case fitcore::Status::Converged:
      return "converged";
    case fitcore::Status::MaxIter:
      return "max_iter";
    case fitcore::Status::Stalled:
      return "stalled";
  }
  return "unknown";  // only for a value outside the enum
}

py::dict to_dict(const fitcore::Result& result) {
  const fitcore::State& s = result.state;
  py::dict d;
  d["R"] = to_numpy(s.R, {3, 3});
  d["t"] = to_numpy(s.t, {3});
  d["theta"] = to_numpy(s.theta, {static_cast<py::ssize_t>(s.theta.size())});
  d["cost"] = result.cost;
  d["iterations"] = result.iterations;
  d["status"] = status_name(result.status);
  return d;
}

py::array_t<double> py_exp_so3(const Array& omega) {
  require_shape(omega, "omega", "(3,)", {3});
  return to_numpy(fitcore::exp_so3(Eigen::Map<const Eigen::Vector3d>(omega.data())), {3, 3});
}

py::array_t<double> py_residuals(const Array& P0, const Array& B, const Array& x, const Array& w,
                                 const Array& p, double f, double cx, double cy, const Array& R,
                                 const Array& t, const Array& theta) {
  const fitcore::Problem problem = make_problem(P0, B, x, w, p, f, cx, cy);
  const fitcore::State state = make_state(R, t, theta, problem.B.cols());
  Eigen::VectorXd r;
  fitcore::residuals(problem, state, r);
  return to_numpy(r, {static_cast<py::ssize_t>(r.size())});
}

py::tuple py_residuals_and_jacobian(const Array& P0, const Array& B, const Array& x,
                                    const Array& w, const Array& p, double f, double cx,
                                    double cy, const Array& R, const Array& t,
                                    const Array& theta) {
  const fitcore::Problem problem = make_problem(P0, B, x, w, p, f, cx, cy);
  const fitcore::State state = make_state(R, t, theta, problem.B.cols());
  Eigen::VectorXd r;
  fitcore::MatXX J;
  fitcore::residuals_and_jacobian(problem, state, r, J);
  return py::make_tuple(to_numpy(r, {static_cast<py::ssize_t>(r.size())}),
                        to_numpy(J, {static_cast<py::ssize_t>(J.rows()),
                                     static_cast<py::ssize_t>(J.cols())}));
}

py::dict py_solve(const Array& P0, const Array& B, const Array& x, const Array& w,
                  const Array& p, double f, double cx, double cy, const Array& R, const Array& t,
                  const Array& theta, int max_iter, double lam0, double ftol, double xtol) {
  const fitcore::Problem problem = make_problem(P0, B, x, w, p, f, cx, cy);
  const fitcore::State init = make_state(R, t, theta, problem.B.cols());
  return to_dict(fitcore::solve(problem, init, {max_iter, lam0, ftol, xtol}));
}

py::list py_solve_batch(const Array& P0, const Array& B, const Array& xs, const Array& ws,
                        const Array& p, double f, double cx, double cy, const Array& Rs,
                        const Array& ts, const Array& thetas, int max_iter, double lam0,
                        double ftol, double xtol, int n_threads) {
  const fitcore::Problem model = make_model(P0, B, p, f, cx, cy);
  const py::ssize_t M = model.P0.rows();
  const py::ssize_t K = model.B.cols();
  require_shape(xs, "xs", "(N, M, 2)", {-1, M, 2});
  const py::ssize_t N = xs.shape(0);
  require_shape(ws, "ws", "(N, M)", {N, M});
  require_shape(Rs, "Rs", "(N, 3, 3)", {N, 3, 3});
  require_shape(ts, "ts", "(N, 3)", {N, 3});
  require_shape(thetas, "thetas", "(N, K)", {N, K});

  // In C order the data of fit n start at n times the size of one fit in each array.
  std::vector<fitcore::MatX2> x_list;
  std::vector<Eigen::VectorXd> w_list;
  std::vector<fitcore::State> inits;
  for (py::ssize_t n = 0; n < N; ++n) {
    x_list.emplace_back(Eigen::Map<const fitcore::MatX2>(xs.data() + n * M * 2, M, 2));
    w_list.emplace_back(Eigen::Map<const Eigen::VectorXd>(ws.data() + n * M, M));
    fitcore::State init;
    init.R = Eigen::Map<const RowMajor33>(Rs.data() + n * 9);
    init.t = Eigen::Map<const Eigen::Vector3d>(ts.data() + n * 3);
    init.theta = Eigen::Map<const Eigen::VectorXd>(thetas.data() + n * K, K);
    inits.push_back(std::move(init));
  }

  std::vector<fitcore::Result> results;
  {
    // The solves touch no Python object, so other Python threads may run meanwhile.
    py::gil_scoped_release release;
    results = fitcore::solve_batch(model, x_list, w_list, inits, {max_iter, lam0, ftol, xtol},
                                   n_threads);
  }
  py::list out;
  for (const fitcore::Result& result : results) {
    out.append(to_dict(result));
  }
  return out;
}

}  // namespace

PYBIND11_MODULE(fitcore, m) {
  using namespace pybind11::literals;
  const fitcore::Options defaults;

  m.doc() = "C++ core of the head-shape fit, a drop-in for fit/solver_ref.py (see fit/SPEC.md).";
  m.attr("__version__") = "0.1.0";

  m.def("exp_so3", &py_exp_so3, "omega"_a,
        "Rotation matrix (3, 3) for the rotation vector omega (3,).");
  m.def("residuals", &py_residuals, "P0"_a, "B"_a, "x"_a, "w"_a, "p"_a, "f"_a, "cx"_a, "cy"_a,
        "R"_a, "t"_a, "theta"_a, "Residual vector r (2M + K,) at the state (R, t, theta).");
  m.def("residuals_and_jacobian", &py_residuals_and_jacobian, "P0"_a, "B"_a, "x"_a, "w"_a, "p"_a,
        "f"_a, "cx"_a, "cy"_a, "R"_a, "t"_a, "theta"_a,
        "Residuals r (2M + K,) and Jacobian J (2M + K, 6 + K), columns [d_omega, d_t, d_theta].");
  m.def("solve", &py_solve, "P0"_a, "B"_a, "x"_a, "w"_a, "p"_a, "f"_a, "cx"_a, "cy"_a, "R"_a,
        "t"_a, "theta"_a, "max_iter"_a = defaults.max_iter, "lam0"_a = defaults.lam0,
        "ftol"_a = defaults.ftol, "xtol"_a = defaults.xtol,
        "Levenberg-Marquardt fit; returns a dict with R, t, theta, cost, iterations, status.");
  m.def("solve_batch", &py_solve_batch, "P0"_a, "B"_a, "xs"_a, "ws"_a, "p"_a, "f"_a, "cx"_a,
        "cy"_a, "Rs"_a, "ts"_a, "thetas"_a, "max_iter"_a = defaults.max_iter,
        "lam0"_a = defaults.lam0, "ftol"_a = defaults.ftol, "xtol"_a = defaults.xtol,
        "n_threads"_a = 0,
        "solve() for N portraits of one model: xs (N, M, 2), ws (N, M), Rs (N, 3, 3), ts (N, 3),\n"
        "thetas (N, K). Releases the GIL; n_threads <= 0 means hardware concurrency. Returns a\n"
        "list of N result dicts in input order, identical to N separate solve() calls.");
}
