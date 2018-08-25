#include "MPC.h"
#include <limits.h>
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

using CppAD::AD;

// Setting these became trial and error for me.
// Any N > 20 is a bit too slow. Most other values of N and dt noticeably cause problems.
size_t N = 10;
double dt = 0.1;

// This value assumes the model presented in the classroom is used.
//
// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;

// The solver input is just one long vector.
// These are for keeping track of the locations of variables within that vector.
size_t idxX = 0;
size_t idxY = idxX + N;
size_t idxPsi = idxY + N;
size_t idxV = idxPsi + N;
size_t idxCTE = idxV + N;
size_t idxEpsi = idxCTE + N;
size_t idxDelta = idxEpsi + N;
size_t idxA = idxDelta + N - 1;

// The reference velocity in mph
double ref_v = 85;

class FG_eval {
 public:
  // Fitted polynomial coefficients
  Eigen::VectorXd coeffs;
  FG_eval(Eigen::VectorXd coeffs) { this->coeffs = coeffs; }

  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;
  void operator()(ADvector& fg, const ADvector& vars) {
    // `fg` a vector of the cost constraints, `vars` is a vector of variable values (state & actuators)
    // NOTE: You'll probably go back and forth between this function and
    // the Solver function below.

    // fg = (cost, X0.., Y0.., Psi0.., V0.., CTE0.., Epsi0...)
    fg[0] = 0;

    // Errors and relative velocity
    for (uint t = 0; t < N; t++) {
      fg[0] += 6*CppAD::pow(vars[idxCTE + t], 2);
      fg[0] += 5*CppAD::pow(vars[idxEpsi + t], 2);
      fg[0] += 1*CppAD::pow(vars[idxV + t] - ref_v, 2);
    }

    // Avoid changing acceleration and steering angle too much
    for (uint t = 0; t < N - 1; t++) {
      fg[0] += 35*CppAD::pow(vars[idxDelta + t] * vars[idxV+t], 2);
      fg[0] += 1*CppAD::pow(vars[idxDelta + t], 2);
      fg[0] += 5*CppAD::pow(vars[idxA + t], 2);
    }

    // Prefer changes in acceleration and steering that are close to the previous
    for (uint t = 0; t < N - 2; t++) {
      fg[0] += 15*CppAD::pow(vars[idxDelta + t + 1] - vars[idxDelta + t], 2);
      fg[0] += 10*CppAD::pow(vars[idxA + t + 1] - vars[idxA + t], 2);
    }

    // Model Equations:
    // x_[t+1] = x[t] + v[t] * cos(psi[t]) * dt
    // y_[t+1] = y[t] + v[t] * sin(psi[t]) * dt
    // psi_[t+1] = psi[t] + v[t] / Lf * delta[t] * dt
    // v_[t+1] = v[t] + a[t] * dt
    // cte[t+1] = f(x[t]) - y[t] + v[t] * sin(epsi[t]) * dt
    // epsi[t+1] = psi[t] - psides[t] + v[t] * delta[t] / Lf * dt

    // Initialize
    fg[1 + idxX] = vars[idxX];
    fg[1 + idxY] = vars[idxY];
    fg[1 + idxPsi] = vars[idxPsi];
    fg[1 + idxV] = vars[idxV];
    fg[1 + idxCTE] = vars[idxCTE];
    fg[1 + idxEpsi] = vars[idxEpsi];

    for (uint t  = 1; t < N; t++) {
      // t vars
      AD<double> x0 = vars[idxX + t - 1];
      AD<double> y0 = vars[idxY + t - 1];
      AD<double> psi0 = vars[idxPsi + t - 1];
      AD<double> v0 = vars[idxV + t - 1];
      AD<double> cte0 = vars[idxCTE + t - 1];
      AD<double> epsi0 = vars[idxEpsi + t - 1];
      AD<double> delta0 = vars[idxDelta + t - 1];
      AD<double> a0 = vars[idxA + t - 1];

      // Adding more coeffs dramatically improved performance
      //AD<double> f0 = coeffs[0] + coeffs[1] * x0;
      //AD<double> psides0 = CppAD::atan(coeffs[1]);
      AD<double> f0 = coeffs[0] + coeffs[1] * x0 + coeffs[2] * CppAD::pow(x0, 2) + coeffs[3] * CppAD::pow(x0, 3);
      AD<double> psides0 = CppAD::atan(coeffs[1] + 2 * coeffs[2] * x0 + 3 * coeffs[3] * CppAD::pow(x0, 2));

      // t+1 vars
      AD<double> x1 = vars[idxX + t];
      AD<double> y1 = vars[idxY + t];
      AD<double> psi1 = vars[idxPsi + t];
      AD<double> v1 = vars[idxV + t];
      AD<double> cte1 = vars[idxCTE + t];
      AD<double> epsi1 = vars[idxEpsi + t];

      fg[1 + idxX + t] = x1 - (x0 + v0 * CppAD::cos(psi0) * dt);
      fg[1 + idxY + t] = y1 - (y0 + v0 * CppAD::sin(psi0) * dt);
      fg[1 + idxPsi + t] = psi1 - (psi0 - v0/Lf * delta0 * dt);
      fg[1 + idxV + t] = v1 - (v0 + a0 * dt);
      fg[1 + idxCTE + t] = cte1 - ((f0 - y0) + (v0 * CppAD::sin(epsi0) * dt));
      fg[1 + idxEpsi + t] = epsi1 - ((psi0 - psides0) - v0/Lf * delta0 * dt);
    }
  }
};

//
// MPC class definition implementation.
//
MPC::MPC() {}
MPC::~MPC() {}

vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {
  bool ok = true;
  typedef CPPAD_TESTVECTOR(double) Dvector;

  // state = (X, Y, Psi, V, CTE, Epsi)
  double x = state[0];
  double y = state[1];
  double psi = state[2];
  double v = state[3];
  double cte = state[4];
  double epsi = state[5];

  // TODO: Set the number of model variables (includes both states and inputs).
  // For example: If the state is a 4 element vector, the actuators is a 2
  // element vector and there are 10 timesteps. The number of variables is:
  //
  // 4 * 10 + 2 * 9

  size_t n_vars = N * 6 + (N - 1) * 2;
  // TODO: Set the number of constraints
  size_t n_constraints = N * 6;

  // Initial value of the independent variables.
  // SHOULD BE 0 besides initial state.
  Dvector vars(n_vars);
  for (uint i = 0; i < n_vars; i++) {
    vars[i] = 0;
  }

  vars[idxX] = x;
  vars[idxY] = y;
  vars[idxPsi] = psi;
  vars[idxV] = v;
  vars[idxCTE] = cte;
  vars[idxEpsi] = epsi;

  Dvector vars_lowerbound(n_vars);
  Dvector vars_upperbound(n_vars);
  for (uint i = 0; i < idxDelta; i++) {
    vars_lowerbound[i] = -1e23;
    vars_upperbound[i] = 1e23;
  }

  for (uint i = idxDelta; i < idxA; i++) {
    // Steering angle between -25, 25 deg
    // 25 deg converted to rad
    vars_lowerbound[i] = -0.436332;
    vars_upperbound[i] = 0.436332;
  }

  for (uint i = idxA; i < n_vars; i++) {
    vars_lowerbound[i] = -1.0;
    vars_upperbound[i] = 1.0;
  }

  // Lower and upper limits for the constraints
  // Should be 0 besides initial state.
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);
  for (uint i = 0; i < n_constraints; i++) {
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }

  constraints_lowerbound[idxX] = x;
  constraints_lowerbound[idxY] = y;
  constraints_lowerbound[idxPsi] = psi;
  constraints_lowerbound[idxV] = v;
  constraints_lowerbound[idxCTE] = cte;
  constraints_lowerbound[idxEpsi] = epsi;

  constraints_upperbound[idxX] = x;
  constraints_upperbound[idxY] = y;
  constraints_upperbound[idxPsi] = psi;
  constraints_upperbound[idxV] = v;
  constraints_upperbound[idxCTE] = cte;
  constraints_upperbound[idxEpsi] = epsi;

  // object that computes objective and constraints
  FG_eval fg_eval(coeffs);

  //
  // NOTE: You don't have to worry about these options
  //
  // options for IPOPT solver
  std::string options;
  // Uncomment this if you'd like more print information
  options += "Integer print_level  0\n";
  // NOTE: Setting sparse to true allows the solver to take advantage
  // of sparse routines, this makes the computation MUCH FASTER. If you
  // can uncomment 1 of these and see if it makes a difference or not but
  // if you uncomment both the computation time should go up in orders of
  // magnitude.
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
  // Change this as you see fit.
  options += "Numeric max_cpu_time          0.5\n";

  // place to return solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // solve the problem
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
      constraints_upperbound, fg_eval, solution);

  // Check some of the solution values
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  // Cost
  auto cost = solution.obj_value;
  std::cout << "Cost " << cost << std::endl;

  // TODO: Return the first actuator values. The variables can be accessed with
  // `solution.x[i]`.
  //
  // {...} is shorthand for creating a vector, so auto x1 = {1.0,2.0}
  // creates a 2 element double vector.
  vector<double> return_val;
  return_val.push_back(solution.x[idxDelta]);
  return_val.push_back(solution.x[idxA]);

  for (uint t = 0; t < N-1; t++) {
    return_val.push_back(solution.x[idxX + t + 1]);
    return_val.push_back(solution.x[idxY + t + 1]);
  }

  return return_val;
}
