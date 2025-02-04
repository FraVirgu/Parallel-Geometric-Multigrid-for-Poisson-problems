#include "gauss_seidel.hpp"
#include "jacobian.hpp"
#include "cg.hpp"

// function prototype
double compute_function(double x, double y)
{
    return sin(p * M_PI * x / a) * sin(q * M_PI * y / a);
}

void JacobiCall(double *x, double *x_new, double *r, double *f, double *residual_reached, int *number_iteration_performed, std::vector<double> *residuals_jacobian, std::vector<double> *error_jacobian, double *x_true)
{
    initialize_zeros_vector(x);
    initialize_zeros_vector(x_new); // Ensure x_tmp is also initialized
    initialize_zeros_vector(r);
    cout << "_____  Jacobian:" << endl;
    bool result = Jacobian(x, x_new, f, r, residual_reached, number_iteration_performed, residuals_jacobian, error_jacobian, x_true);

    if (result)
    {
        std::cout << "Number of iteration performed: " << *number_iteration_performed << std::endl;
        std::cout << "Residual reached: " << *residual_reached << std::endl;
    }
    else
    {
        std::cout << "Did not converge within the maximum number of iterations." << std::endl;
        std::cout << "Residual reached: " << *residual_reached << std::endl;
    }
}

void ConjugateGradientCall(double *x, double *f, double *r, int *number_iteration_performed, double *residual_reached, std::vector<double> *residuals_cg, std::vector<double> *error_cg, double *x_true)
{
    initialize_zeros_vector(x);
    initialize_zeros_vector(r);
    cout << "_____  Conjugate Gradient:" << endl;
    bool result = ConjugateGradient(x, f, r, number_iteration_performed, residual_reached, residuals_cg, error_cg, x_true);

    if (result)
    {
        std::cout << "Number of iteration performed: " << *number_iteration_performed << std::endl;
        std::cout << "Residual reached: " << *residual_reached << std::endl;
    }
    else
    {
        std::cout << "Did not converge within the maximum number of iterations." << std::endl;
        std::cout << "Residual reached: " << *residual_reached << std::endl;
    }
}

void GaussSeidelCall(double *x, double *f, double *r, double *residual_reached, int *number_iteration_performed, std::vector<double> *residuals_gs, std::vector<double> *error_gs, double *x_true)
{
    initialize_zeros_vector(x);
    initialize_zeros_vector(r);
    cout << "_____  Gauss-Seidel:" << endl;
    bool result = GaussSeidel(x, f, r, residual_reached, number_iteration_performed, residuals_gs, error_gs, x_true);
    if (result)
    {
        std::cout << "Number of iteration performed: " << *number_iteration_performed << std::endl;
        std::cout << "Residual reached: " << *residual_reached << std::endl;
    }
    else
    {
        std::cout << "Did not converge within the maximum number of iterations." << std::endl;
        std::cout << "Residual reached: " << *residual_reached << std::endl;
    }
}

void save_residuals_to_file(std::vector<double> *residuals_jacobian, std::vector<double> *residuals_cg, std::vector<double> *residuals_gs)
{
    std::ofstream file_jacobian("residuals_jacobian.txt");
    if (file_jacobian.is_open())
    {
        for (const auto &residual : *residuals_jacobian)
        {
            file_jacobian << residual << "\n";
        }
        file_jacobian.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Jacobian residuals.\n";
    }

    std::ofstream file_cg("residuals_cg.txt");
    if (file_cg.is_open())
    {
        for (const auto &residual : *residuals_cg)
        {
            file_cg << residual << "\n";
        }
        file_cg.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Conjugate Gradient residuals.\n";
    }

    std::ofstream file_gs("residuals_gs.txt");
    if (file_gs.is_open())
    {
        for (const auto &residual : *residuals_gs)
        {
            file_gs << residual << "\n";
        }
        file_gs.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Gauss-Seidel residuals.\n";
    }
}

void save_error_to_file(std::vector<double> *error_jacobian, std::vector<double> *error_cg, std::vector<double> *error_gs)
{
    std::ofstream file_jacobian("error_jacobian.txt");
    if (file_jacobian.is_open())
    {
        for (const auto &error : *error_jacobian)
        {
            file_jacobian << error << "\n";
        }
        file_jacobian.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Jacobian errors.\n";
    }

    std::ofstream file_cg("error_cg.txt");
    if (file_cg.is_open())
    {
        for (const auto &error : *error_cg)
        {
            file_cg << error << "\n";
        }
        file_cg.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Conjugate Gradient errors.\n";
    }

    std::ofstream file_gs("error_gs.txt");
    if (file_gs.is_open())
    {
        for (const auto &error : *error_gs)
        {
            file_gs << error << "\n";
        }
        file_gs.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Gauss-Seidel errors.\n";
    }
}

int main()
{

    std::vector<double> *residuals_jacobian = new std::vector<double>();
    std::vector<double> *residuals_cg = new std::vector<double>();
    std::vector<double> *residuals_gs = new std::vector<double>();

    std::vector<double> *error_jacobian = new std::vector<double>();
    std::vector<double> *error_cg = new std::vector<double>();
    std::vector<double> *error_gs = new std::vector<double>();

    double *x = new double[L];
    double *x_tmp = new double[L];
    double *x_true = new double[L];
    double *f = new double[L];
    double *res = new double[L];
    int *number_iteration_performed = new int;
    double *residual_reached = new double;

    compute_rhs(f, compute_function);
    compute_laplacian(x_true, compute_function);
    double f_norm = vector_norm(f);

    JacobiCall(x, x_tmp, res, f, residual_reached, number_iteration_performed, residuals_jacobian, error_jacobian, x_true);

    ConjugateGradientCall(x, f, res, number_iteration_performed, residual_reached, residuals_cg, error_cg, x_true);

    GaussSeidelCall(x, f, res, residual_reached, number_iteration_performed, residuals_gs, error_gs, x_true);

    free(x);
    free(x_tmp);
    free(f);
    free(number_iteration_performed);
    free(residual_reached);
    save_residuals_to_file(residuals_jacobian, residuals_cg, residuals_gs);
    save_error_to_file(error_jacobian, error_cg, error_gs);

    delete residuals_jacobian;
    delete residuals_cg;
    delete residuals_gs;

    return 0;
}