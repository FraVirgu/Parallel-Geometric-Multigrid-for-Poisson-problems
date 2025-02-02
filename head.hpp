#include <cmath>
#include <cstdlib>
#include <iostream>

#define N 1000
#define W N
#define H N
#define h (1.0 / N) // Ensure floating-point division
#define L (N * N)
#define MAX_ITERATION 100
#define EPSILON 1e-4
#define a 1.0
#define p 1.0
#define q 1.0

using namespace std;

double vector_norm(const double *f)
{
    double sum = 0.0;
    for (int i = 0; i < L; i++)
    {
        sum += f[i] * f[i]; // Sum of squares
    }
    return sqrt(sum); // Square root of sum
}

// Compute residual norm
double compute_residual_norm(double *x, double *f)
{
    double norm = 0.0;
    for (int y = 1; y < H - 1; y++)
    {
        for (int x_pos = 1; x_pos < W - 1; x_pos++)
        {
            int index = y * W + x_pos;
            norm += pow((h * h) * f[index] - 4 * x[index] + x[index - 1] + x[index + 1] + x[index - W] + x[index + W], 2);
        }
    }
    return sqrt(norm);
}

// Compute Jacobi iteration for a single index
void compute_jacobian(int index, double *x, double *x_new, double *f)
{
    double tmp_sum = (h * h) * f[index] + x[index - 1] + x[index + 1] + x[index - W] + x[index + W];
    x_new[index] = 0.25 * tmp_sum;
}

// Perform Jacobi iterations
bool Jacobian(double *x, double *x_new, double *f, int *number_iteration_performed, double *residual_reached)
{
    double norm_residual = 0.0;
    for (int i = 0; i < MAX_ITERATION; i++)
    {
        for (int y = 1; y < H - 1; y++)
        {
            for (int x_pos = 1; x_pos < W - 1; x_pos++)
            {
                int index = y * W + x_pos;
                compute_jacobian(index, x, x_new, f);
            }
        }

        norm_residual = compute_residual_norm(x_new, f);
        *residual_reached = norm_residual;
        //  cout << "Iteration " << i << " - Residual Norm: " << norm_residual << endl;

        if (norm_residual < EPSILON)
        {
            *number_iteration_performed = i;
            return true;
        }

        for (int j = 0; j < L; j++)
        {
            x[j] = x_new[j];
        }
    }
    return false;
}

// Initialize vector x to zero
void initialize_x(double *x)
{
    for (int i = 0; i < L; i++)
    {
        x[i] = 0.0;
    }
}
// Compute Laplacian using finite differences
void compute_laplacian(double *f, double (*func)(double, double))
{
    double dx = a / W;
    double dy = a / H;
    double dx2 = dx * dx;
    double dy2 = dy * dy;

    for (int y = 1; y < H - 1; y++)
    {
        for (int x = 1; x < W - 1; x++)
        {
            double x_val = x * dx;
            double y_val = y * dy;

            // Discrete Laplacian: -1/h^2 * (u[i-1] + u[i+1] + u[j-1] + u[j+1] - 4*u[i,j])
            f[y * W + x] = -(func(x_val - dx, y_val) + func(x_val + dx, y_val) +
                             func(x_val, y_val - dy) + func(x_val, y_val + dy) -
                             4 * func(x_val, y_val)) /
                           dx2;
        }
    }

    // Set boundary conditions to zero
    for (int x = 0; x < W; x++)
    {
        f[x] = 0.0;               // Top boundary
        f[(H - 1) * W + x] = 0.0; // Bottom boundary
    }
    for (int y = 0; y < H; y++)
    {
        f[y * W] = 0.0;           // Left boundary
        f[y * W + (W - 1)] = 0.0; // Right boundary
    }
}
