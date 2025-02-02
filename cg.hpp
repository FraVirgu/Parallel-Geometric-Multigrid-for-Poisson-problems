#include "head.hpp"
// Compute optimal alpha (alpha_opt = (r^T r) / (r^T A r)), using the stencil
double compute_alpha_opt(double *r)
{
    double numerator = 0.0;
    double denominator = 0.0;
    double temp_r;

    for (int y = 1; y < H - 1; y++)
    {
        for (int x_pos = 1; x_pos < W - 1; x_pos++)
        {
            int index = y * W + x_pos;
            numerator += r[index] * r[index];

            // Apply Laplacian operator (A * r) using the same stencil
            temp_r = -4 * r[index] + r[index - 1] + r[index + 1] + r[index - W] + r[index + W];
            denominator += r[index] * temp_r;
        }
    }
    return numerator / denominator;
}

// Perform Conjugate Gradient iterations
bool ConjugateGradient(double *x, double *f, int *number_iteration_performed, double *residual_reached, vector<double> *residuals)
{
    double *r = new double[L]; // Residual
    double alpha_opt;
    double norm_residual;

    // Compute initial residual
    compute_residual(r, x, f);
    norm_residual = vector_norm(r);
    residuals->push_back(norm_residual);
    cout << "Initial residual: " << norm_residual << endl;

    for (int i = 0; i < MAX_ITERATION; i++)
    {
        alpha_opt = compute_alpha_opt(r);

        // Update solution: x[k+1] = x[k] - alpha_opt * r[k]
        for (int j = 0; j < L; j++)
        {
            x[j] -= alpha_opt * r[j];
        }

        // Compute new residual
        compute_residual(r, x, f);
        norm_residual = vector_norm(r);
        residuals->push_back(norm_residual);
        *residual_reached = norm_residual;

        // cout << "Iteration " << i << " - Residual Norm: " << norm_residual << endl;

        // Convergence check
        if (norm_residual < EPSILON)
        {
            *number_iteration_performed = i;
            delete[] r;
            return true;
        }
    }

    delete[] r;
    return false;
}
