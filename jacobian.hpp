#include "head.hpp"

// Perform Jacobi iterations
bool Jacobian(double *x, double *x_new, double *f, double *residual_reached, int *number_iteration_performed, vector<double> *residuals)
{
    double norm_residual;
    double *r = new double[L]; // Residual
    // Compute initial residual
    compute_residual(r, x, f);
    norm_residual = vector_norm(r);
    residuals->push_back(norm_residual);
    cout << "Initial residual: " << norm_residual << endl;
    for (int i = 0; i < MAX_ITERATION; i++)
    {
        // Perform Jacobi iteration
        for (int y = 1; y < H - 1; y++)
        {
            for (int x_pos = 1; x_pos < W - 1; x_pos++)
            {
                int index = y * W + x_pos;
                x_new[index] = 0.25 * ((h * h * f[index]) + x[index - 1] + x[index + 1] + x[index - W] + x[index + W]);
            }
        }

        // Compute new residual
        compute_residual(r, x_new, f);
        norm_residual = vector_norm(r);
        residuals->push_back(norm_residual);
        *residual_reached = norm_residual;

        // Convergence check
        if (norm_residual < EPSILON)
        {
            *number_iteration_performed = i;
            return true;
        }

        // Copy x_new to x for next iteration

        for (int j = 0; j < L; j++)
        {
            x[j] = x_new[j];
        }
    }

    return false;
}
