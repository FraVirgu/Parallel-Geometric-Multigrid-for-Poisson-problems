#include "head.hpp"

// Perform Gauss-Seidel iterations
bool GaussSeidel(double *x, double *f, double *r, double *residual_reached, int *number_iteration_performed, vector<double> *residuals)
{
    double norm_residual;
    compute_residual(r, x, f);
    norm_residual = vector_norm(r);
    residuals->push_back(norm_residual);
    cout << "Initial residual: " << norm_residual << endl;

    for (int i = 0; i < MAX_ITERATION; i++)
    {
        // Perform Gauss-Seidel iteration (update x in place)
        for (int y = 1; y < H - 1; y++)
        {
            for (int x_pos = 1; x_pos < W - 1; x_pos++)
            {
                int index = y * W + x_pos;
                x[index] = 0.25 * ((h * h * f[index]) + x[index - 1] + x[index + 1] + x[index - W] + x[index + W]);
            }
        }

        // Compute new residual
        compute_residual(r, x, f);
        norm_residual = vector_norm(r);
        residuals->push_back(norm_residual);
        *residual_reached = norm_residual;
        // cout << "Iteration " << i << " - Residual Norm: " << norm_residual << endl;
        //  Convergence check
        if (norm_residual < EPSILON)
        {
            *number_iteration_performed = i;
            return true;
        }
    }

    return false;
}
