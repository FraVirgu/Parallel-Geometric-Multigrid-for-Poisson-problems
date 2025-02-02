#include "./head.hpp"

// function prototype
double compute_function(double x, double y)
{
    return sin(p * M_PI * x / a) * sin(q * M_PI * y / a);
}

int main()
{
    double *x;
    double *x_tmp;
    double *f;
    double *residual_reached;
    int *number_iteration_performed;
    // malloc x of size L
    x = (double *)malloc(L * sizeof(double));
    // malloc x_tmp of size L
    x_tmp = (double *)malloc(L * sizeof(double));
    // malloc f of size L
    f = (double *)malloc(L * sizeof(double));
    // malloc residual_reached of size 1
    residual_reached = (double *)malloc(1 * sizeof(double));
    // malloc number_iteration_performed of size 1
    number_iteration_performed = (int *)malloc(1 * sizeof(int));

    initialize_x(x);
    compute_laplacian(f, compute_function);
    double initial_residual = compute_residual_norm(x, f);
    cout << "Initial residual: " << initial_residual << endl;
    bool result = Jacobian(x, x_tmp, f, number_iteration_performed, residual_reached);

    if (result)
    {
        std::cout << "Number of iteration performed: " << *number_iteration_performed << std::endl;
        std::cout << "Residual reached: " << *residual_reached << std::endl;
    }
    else
    {
        std::cout << "MAX_ITERATION ERROR " << std::endl;
    }
    return 0;
}