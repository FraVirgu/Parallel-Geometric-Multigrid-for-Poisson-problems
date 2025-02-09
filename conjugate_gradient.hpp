#include "head.hpp"

bool conjugate_gradient(double *x, double *f, double *r, int *number_iteration_performed, double *residual_reached, vector<double> *residuals)
{
    double alpha, beta, norm_residual, res_tmp, err_tmp, norm_error;
    double *err = new double[L];
    double *p_d = new double[L];  // Search direction
    double *Ap_d = new double[L]; // Matrix-vector product

    // Compute initial residual: r = f - A * x
    compute_residual(r, x, f);
    norm_residual = vector_norm(r);
    res_tmp = norm_residual;
    residuals->push_back(norm_residual);

    // Initialize search direction: p = r
    for (int j = 0; j < L; j++)
    {
        p_d[j] = r[j];
    }

    for (int i = 0; i < MAX_ITERATION; i++)
    {

        // Compute p_d^(k)^Tr^(k)
        double rTr = compute_inner_product(p_d, r); // Store r^T * r
        // Compute A * p
        compute_inner_product_with_A(r, Ap_d);

        // Compute step size: alpha = (r^T * r) / (p^T * A * p)
        double pAp = compute_inner_product(p_d, Ap_d);
        alpha = rTr / pAp;

        // Update solution: x[k+1] = x[k] + alpha * p_d
        for (int j = 0; j < L; j++)
        {
            x[j] += alpha * p_d[j];
        }

        // Update residual: r[k+1] = r[k] - alpha * A * p
        for (int j = 0; j < L; j++)
        {
            r[j] -= alpha * Ap_d[j];
        }

        norm_residual = vector_norm(r);
        residuals->push_back(norm_residual);

        // Convergence check
        if (norm_residual < EPSILON)
        {
            *residual_reached = norm_residual;
            *number_iteration_performed = i;
            delete[] err;
            delete[] p_d;
            delete[] Ap_d;
            return true;
        }

        // Compute new beta: beta = (r[k+1]^T * r[k+1]) / (r[k]^T * r[k])
        double rTr_new = compute_inner_product(Ap_d, r);
        beta = rTr_new / pAp;

        // Update search direction: p[k+1] = r[k+1] + beta * p[k]
        for (int j = 0; j < L; j++)
        {
            p_d[j] = r[j] + beta * p_d[j];
        }
    }

    delete[] err;
    delete[] p_d;
    delete[] Ap_d;
    return false;
}

// Function to compute the standard inner product
double compute_inner_product(const double *v1, const double *v2)
{
    double result = 0.0;
    for (int i = 0; i < L; i++)
    {
        result += v1[i] * v2[i];
    }
    return result;
}

// Function to compute the inner product with A where A * r is calculated using the Laplacian
void compute_inner_product_with_A(double *r, double *result)
{
    for (int y = 1; y < H - 1; y++)
    {
        for (int x_pos = 1; x_pos < W - 1; x_pos++)
        {
            int index = y * W + x_pos;

            // Apply Laplacian operator (A * r) using the same stencil
            result[index] = -4 * r[index] + r[index - 1] + r[index + 1] + r[index - W] + r[index + W];
        }
    }
}
