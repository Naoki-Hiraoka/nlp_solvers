#include <roboptim/core.hh>
#include <iostream>

class F : public roboptim::TwiceDifferentiableSparseFunction
{
public:
  F () : roboptim::TwiceDifferentiableSparseFunction (4, 1, "x₀ * x₃ * (x₀ + x₁ + x₂) + x₂")
  {
  }

  void
  impl_compute (result_ref result, const_argument_ref x) const override
  {
    result[0] = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];
  }

  void
  impl_gradient (gradient_ref grad, const_argument_ref x, size_type) const override
  {
    grad.coeffRef(0,0) = x[0] * x[3] + x[3] * (x[0] + x[1] + x[2]);
    grad.coeffRef(0,1) = x[0] * x[3];
    grad.coeffRef(0,2) = x[0] * x[3] + 1;
    grad.coeffRef(0,3) = x[0] * (x[0] + x[1] + x[2]);
  }

  void
  impl_hessian (hessian_ref h, const_argument_ref x, size_type) const override
  {
    h.coeffRef(0,0)=2 * x[3]; h.coeffRef(0,1)=x[3]; h.coeffRef(0,2)=x[3]; h.coeffRef(0,3)=2 * x[0] + x[1] + x[2];
    h.coeffRef(1,0)=x[3]; h.coeffRef(1,3)=x[0];
    h.coeffRef(2,0)=x[3]; h.coeffRef(2,3)=x[1];//bug?
    h.coeffRef(3,0)=2 * x[0] + x[1] + x[2]; h.coeffRef(3,1)=x[0]; h.coeffRef(3,2)=x[0];
  }
};


class G0 : public roboptim::TwiceDifferentiableSparseFunction
{
public:
  G0 () : roboptim::TwiceDifferentiableSparseFunction (4, 1, "x₀ * x₁ * x₂ * x₃")
  {
  }

  void
  impl_compute (result_ref result, const_argument_ref x) const override
  {
    result[0] = x[0] * x[1] * x[2] * x[3];
  }

  void
  impl_gradient (gradient_ref grad, const_argument_ref x, size_type) const override
  {
    grad.coeffRef(0,0) = x[1] * x[2] * x[3];
    grad.coeffRef(0,1) = x[0] * x[2] * x[3];
    grad.coeffRef(0,2) = x[0] * x[1] * x[3];
    grad.coeffRef(0,3) = x[0] * x[1] * x[2];
  }

  void
  impl_hessian (hessian_ref h, const_argument_ref x, size_type) const override
  {
    h.coeffRef(0,1)=x[2] * x[3]; h.coeffRef(0,2)=x[1] * x[3]; h.coeffRef(0,3)=x[1] * x[2];
    h.coeffRef(1,0)=x[2] * x[3]; h.coeffRef(1,2)=x[0] * x[3]; h.coeffRef(1,3)=x[0] * x[2];
    h.coeffRef(2,0)=x[1] * x[3]; h.coeffRef(2,1)=x[0] * x[3]; h.coeffRef(2,3)=x[0] * x[1];
    h.coeffRef(3,0)=x[1] * x[2]; h.coeffRef(3,1)=x[0] * x[2]; h.coeffRef(3,2)=x[0] * x[1];
  }
};

class G1 : public roboptim::TwiceDifferentiableSparseFunction
{
public:
  G1 () : roboptim::TwiceDifferentiableSparseFunction (4, 1, "x₀² + x₁² + x₂² + x₃²")
  {
  }

  void
  impl_compute (result_ref result, const_argument_ref x) const override
  {
    result[0] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
  }

  void
  impl_gradient (gradient_ref grad, const_argument_ref x, size_type) const override
  {
    grad.coeffRef(0,0) = 2 * x[0];
    grad.coeffRef(0,1) = 2 * x[1];
    grad.coeffRef(0,2) = 2 * x[2];
    grad.coeffRef(0,3) = 2 * x[3];
  }

  void
  impl_hessian (hessian_ref h, const_argument_ref x, size_type) const override
  {
    h.coeffRef(0,0)=2;
    h.coeffRef(1,1)=2;
    h.coeffRef(2,2)=2;
    h.coeffRef(3,3)=2;
  }
};

int main(void){
  // Create cost function.
  boost::shared_ptr<F> f (new F ());

  // Create problem.
  roboptim::Solver<roboptim::EigenMatrixSparse>::problem_t pb (f);

  // Set bounds for all optimization parameters.
  // 1. < x_i < 5. (x_i in [1.;5.])
  for (roboptim::SparseFunction::size_type i = 0; i < pb.function ().inputSize (); ++i)
    pb.argumentBounds ()[i] = roboptim::SparseFunction::makeInterval (1., 5.);

  // Set the starting point.
  roboptim::SparseFunction::vector_t start (pb.function ().inputSize ());
  start[0] = 1., start[1] = 5., start[2] = 5., start[3] = 1.;
  pb.startingPoint() = start;

  // Create constraints.
  boost::shared_ptr<G0> g0 (new G0 ());
  boost::shared_ptr<G1> g1 (new G1 ());

  F::intervals_t bounds;
  roboptim::Solver<roboptim::EigenMatrixSparse>::problem_t::scaling_t scaling;

  // Add constraints
  bounds.push_back(roboptim::SparseFunction::makeLowerInterval (25.));
  scaling.push_back (1.);
  pb.addConstraint
    (boost::static_pointer_cast<roboptim::TwiceDifferentiableSparseFunction> (g0),
     bounds, scaling);

  bounds.clear ();
  scaling.clear ();

  bounds.push_back(roboptim::SparseFunction::makeInterval (40., 40.));
  scaling.push_back (1.);
  pb.addConstraint
    (boost::static_pointer_cast<roboptim::TwiceDifferentiableSparseFunction> (g1),
     bounds, scaling);

  // Initialize solver.

  // Here we are relying on a dummy solver.
  // You may change this string to load the solver you wish to use:
  //  - Ipopt: "ipopt", "ipopt-sparse", "ipopt-td"
  //  - Eigen: "eigen-levenberg-marquardt"
  //  etc.
  // The plugin is built for a given solver type, so choose it adequately.
  roboptim::SolverFactory<roboptim::Solver<roboptim::EigenMatrixSparse> > factory ("ipopt-sparse", pb);
  roboptim::Solver<roboptim::EigenMatrixSparse>& solver = factory ();

  // Compute the minimum and retrieve the result.
  roboptim::Solver<roboptim::EigenMatrixSparse>::result_t res = solver.minimum ();

  // Display solver information.
  std::cout << solver << std::endl;

  // Check if the minimization has succeeded.

  // Process the result
  switch (res.which ())
    {
    case roboptim::Solver<roboptim::EigenMatrixSparse>::SOLVER_VALUE:
      {
        // Get the result.
        roboptim::Result& result = boost::get<roboptim::Result> (res);

        // Display the result.
        std::cout << "A solution has been found: " << std::endl
                         << result << std::endl;
        for(int i=0;i<result.inputSize;i++){
          std::cout << result.x[i] << std::endl;
        }
        return 0;
      }

    case roboptim::Solver<roboptim::EigenMatrixSparse>::SOLVER_ERROR:
      {
        std::cout << "A solution should have been found. Failing..."
                         << std::endl
                         << boost::get<roboptim::SolverError> (res).what ()
                         << std::endl;

        return 0;
      }

    case roboptim::Solver<roboptim::EigenMatrixSparse>::SOLVER_NO_SOLUTION:
      {
        std::cout << "The problem has not been solved yet."
                         << std::endl;

        return 0;
      }
    }

  // Should never happen.
  assert (0);
  return 0;

}
