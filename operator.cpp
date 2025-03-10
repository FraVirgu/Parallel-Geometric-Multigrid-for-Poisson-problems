#include <iostream>
#include <vector>
#include "head.hpp"
#include <cmath>
using namespace std;

void compute_residual(double *r, double *x, double *f, int W, int H, double h)
{
    for (int y = 1; y < H - 1; y++)
    {
        for (int x_pos = 1; x_pos < W - 1; x_pos++)
        {
            int index = y * W + x_pos;
            r[index] = f[index] - (4 * x[index] - x[index - 1] - x[index + 1] - x[index - W] - x[index + W]) / (h * h);
        }
    }
}

void directSolve(double *A, double *b, double *x, int size)
{
    // Gaussian elimination for small system A * x = b
    for (int i = 0; i < size; i++)
    {
        // Find the pivot row
        int pivot = i;
        for (int k = i + 1; k < size; k++)
        {
            if (std::abs(A[k * size + i]) > std::abs(A[pivot * size + i]))
                pivot = k;
        }

        // Swap rows
        for (int j = 0; j < size; j++)
            std::swap(A[i * size + j], A[pivot * size + j]);
        std::swap(b[i], b[pivot]);

        // Normalize row i
        double diag = A[i * size + i];
        for (int j = 0; j < size; j++)
        {
            A[i * size + j] /= diag;
        }
        b[i] /= diag;

        // Eliminate remaining rows
        for (int k = i + 1; k < size; k++)
        {
            double factor = A[k * size + i];
            for (int j = 0; j < size; j++)
            {
                A[k * size + j] -= factor * A[i * size + j];
            }
            b[k] -= factor * b[i];
        }
    }

    // Back-substitution
    for (int i = size - 1; i >= 0; i--)
    {
        x[i] = b[i];
        for (int j = i + 1; j < size; j++)
        {
            x[i] -= A[i * size + j] * x[j];
        }
    }
}

void prolungator(double *input, double *output, int input_H, int input_W, int output_H, int output_W)
{
    double weight[9] = {0.25, 0.5, 0.25, 0.5, 1.0, 0.5, 0.25, 0.5, 0.25};

    int index_output_x;
    int index_output_y;

    for (int i = 0; i < input_H; i++)
    {
        for (int j = 0; j < input_W; j++)
        {
            index_output_x = (2 * j) - 1;
            index_output_y = (2 * i) - 1;

            for (int k = 0; k < 3; k++)
            {
                for (int l = 0; l < 3; l++)
                {
                    int x = index_output_x + l;
                    int y = index_output_y + k;

                    if (x >= 0 && x < output_W && y >= 0 && y < output_H)
                    {
                        output[y * output_W + x] += weight[k * 3 + l] * input[i * input_W + j];
                    }
                }
            }
        }
    }
}

void restriction(double *input, double *output, int input_H, int input_W, int output_H, int output_W)
{
    double weight[9] = {0.25, 0.5, 0.25, 0.5, 1.0, 0.5, 0.25, 0.5, 0.25};

    for (int i = 0; i < output_H; i++)
    {
        for (int j = 0; j < output_W; j++)
        {
            double sum = 0.0;
            double weight_sum = 0.0;
            int index_input_x = 2 * j;
            int index_input_y = 2 * i;

            for (int k = -1; k <= 1; k++)
            {
                for (int l = -1; l <= 1; l++)
                {
                    int x = index_input_x + l;
                    int y = index_input_y + k;

                    if (x >= 0 && x < input_W && y >= 0 && y < input_H)
                    {
                        sum += weight[(k + 1) * 3 + (l + 1)] * input[y * input_W + x];
                        weight_sum += weight[(k + 1) * 3 + (l + 1)];
                    }
                }
            }

            output[i * output_W + j] = sum / weight_sum;
        }
    }
}

// Perform Jacobi iterations
bool Jacobian(double *x, double *x_new, double *f, double *r)
{
    double norm_residual;
    double norm_error;
    double *err = new double[L];

    for (int i = 0; i < number_fixed_iteration; i++)
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

        // Copy x_new to x for next iteration
        for (int j = 0; j < L; j++)
        {
            x[j] = x_new[j];
        }
    }
    compute_residual(r, x, f);
    return true;
}

void decrease_global_parameter()
{
    N = N / 2;
    L = N * N;
    W = N;
    H = N;
    h = 1.0 / (N - 1);
}

void MG(double *output, double *initial_solution, double *smoother_output, double *f, double *smoother_residual, int v1, int v2, int level)
{
    // actual level
    int n = N;
    int l = n * n;
    int weight = n;
    int height = n;

    cout << endl;
    cout << "Level: " << level << endl;
    cout << "N: " << N << endl;
    cout << "initial residual : " << compute_residual_norm(initial_solution, f) << endl;

    // PRE-SMOOTHING
    number_fixed_iteration = v1;
    Jacobian(initial_solution, smoother_output, f, smoother_residual);

    cout << "smoother residual : " << compute_residual_norm(smoother_output, f) << endl;

    // COARSE RESIDUAL
    decrease_global_parameter();
    double *r_H = new double[L];
    restriction(smoother_residual, r_H, height, weight, H, W);

    // CONDITION EVALUATION
    double *delta_H = new double[L];
    if (N <= 3) // Smallest grid level, solve directly
    {
        /*
                cout << "level: " << level << endl;
                cout << "N: " << N << endl;
                cout << "smoother residual after update : " << compute_residual_norm(smoother_output, f) << endl;

                double *zero_H = new double[L];
                initialize_zeros_vector(zero_H);
                number_fixed_iteration = 10; // You can adjust the number of iterations as needed
                Jacobian(zero_H, delta_H, r_H, r_H);
        */
    }
    else
    {
        double *zero_H = new double[L];
        double *smoother_output_H = new double[L];
        double *smoother_residual_H = new double[L];
        initialize_zeros_vector(zero_H);
        MG(delta_H, zero_H, smoother_output_H, r_H, smoother_residual_H, v1, v2, level + 1);
    }

    // UPDATE
    double *delta_h = new double[l];
    prolungator(delta_H, delta_h, H, W, height, weight);

    for (int i = 0; i < l; i++)
    {
        smoother_output[i] += delta_h[i];
    }

    // POST-SMOOTHING
    number_fixed_iteration = v2;
    Jacobian(smoother_output, output, f, smoother_residual);
}

// Main function to test the multigrid method
int main()
{
    // Save initial parameters
    int initial_W = W;
    int initial_H = H;

    // Allocate memory
    double *solution = new double[L];        // Final computed solution
    double *initial_guess = new double[L];   // Initial guess (all zeros)
    double *smoother_output = new double[L]; // Smoother output
    double *rhs = new double[L];             // Right-hand side (forcing function)
    double *residual = new double[L];        // Residual vector

    // Initialize vectors
    initialize_zeros_vector(solution);
    initialize_zeros_vector(initial_guess);
    initialize_zeros_vector(smoother_output);
    initialize_zeros_vector(rhs);
    initialize_zeros_vector(residual);

    compute_rhs(rhs);

    // Multigrid parameters
    int v1 = 2; // Number of pre-smoothing steps
    int v2 = 2; // Number of post-smoothing steps
    int level = 0;

    // Call the multigrid solver
    MG(solution, initial_guess, smoother_output, rhs, residual, v1, v2, level);

    // Compute final residual to check convergence
    compute_residual(residual, solution, rhs, W, H, h);

    // Print final solution
    cout << "Final computed solution (u):" << endl;
    for (int i = 0; i < initial_H; i++)
    {
        for (int j = 0; j < initial_W; j++)
        {
            cout << solution[i * W + j] << " ";
        }
        cout << endl;
    }

    // Compute and print residual norm
    double residual_norm = 0.0;
    for (int i = 0; i < L; i++)
    {
        residual_norm += residual[i] * residual[i];
    }
    residual_norm = sqrt(residual_norm);
    cout << "Residual norm: " << residual_norm << endl;

    // Cleanup
    delete[] solution;
    delete[] initial_guess;
    delete[] smoother_output;
    delete[] rhs;
    delete[] residual;

    return 0;
}