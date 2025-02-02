#include "jacobian.hpp"
#include "cg.hpp"

// function prototype
double compute_function(double x, double y)
{
    return sin(p * M_PI * x / a) * sin(q * M_PI * y / a);
}

int main()
{

    std::vector<double> *residuals_jacobian = new std::vector<double>();
    std::vector<double> *residuals_cg = new std::vector<double>();
    double *x = (double *)malloc(L * sizeof(double));
    double *x_tmp = (double *)malloc(L * sizeof(double));
    double *f = (double *)malloc(L * sizeof(double));
    double *res = (double *)malloc(L * sizeof(double));
    int *number_iteration_performed = (int *)malloc(1 * sizeof(int));
    double *residual_reached = (double *)malloc(1 * sizeof(double));

    initialize_x(x);
    initialize_x(x_tmp); // Ensure x_tmp is also initialized
    compute_laplacian(f, compute_function);
    cout << "Jacobian:" << endl;
    double initial_residual = compute_residual_norm(x, f);
    bool result = Jacobian(x, x_tmp, f, residual_reached, number_iteration_performed, residuals_jacobian);

    if (result)
    {
        std::cout << "Number of iteration performed: " << *number_iteration_performed << std::endl;
        std::cout << "Residual reached: " << *residual_reached << std::endl;
    }
    else
    {
        std::cout << "Did not converge within the maximum number of iterations." << std::endl;
        std::cout << "Residual reached: " << *residual_reached << std::endl;
        std::cout << "initial_residual - residual_reached: " << initial_residual - *residual_reached << std::endl;
    }

    initialize_x(x);
    initialize_x(x_tmp);
    initialize_x(res);
    compute_laplacian(f, compute_function);
    cout << "Conjugate Gradient:" << endl;
    result = ConjugateGradient(x, f, number_iteration_performed, residual_reached, residuals_cg);

    if (result)
    {
        std::cout << "Number of iteration performed: " << *number_iteration_performed << std::endl;
        std::cout << "Residual reached: " << *residual_reached << std::endl;
    }
    else
    {
        std::cout << "Did not converge within the maximum number of iterations." << std::endl;
        std::cout << "Residual reached: " << *residual_reached << std::endl;
        // std::cout << "initial_residual - residual_reached: " << initial_residual - *residual_reached << std::endl;
    }

    free(x);
    free(x_tmp);
    free(f);
    free(number_iteration_performed);
    free(residual_reached);

    // Save the result to a file
    std::ofstream file_jacobian("residuals_jacobian.txt");
    std::ofstream file_cg("residuals_cg.txt");

    for (int i = 0; i < residuals_jacobian->size(); i++)
    {
        file_jacobian << residuals_jacobian->at(i) << std::endl;
    }

    for (int i = 0; i < residuals_cg->size(); i++)
    {
        file_cg << residuals_cg->at(i) << std::endl;
    }

    file_jacobian.close();
    file_cg.close();

    delete residuals_jacobian;
    delete residuals_cg;

    return 0;
}