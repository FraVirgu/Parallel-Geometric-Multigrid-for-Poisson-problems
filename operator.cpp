#include <iostream>
#include <vector>
#include <cmath>
#include "globals.hpp"
#include <chrono>
#include "main.hpp"

using namespace std;

// Initialize vector to zero
void initialize_vector(double *x, int dim)
{
    fill(x, x + dim, 0.0);
}

// Compute residual r = f - Ax
void compute_residual(double *r, double *x, double *f, int weight, int height, double h_actual)
{
    for (int y = 1; y < height - 1; y++)
    {
        for (int x_pos = 1; x_pos < weight - 1; x_pos++)
        {
            int index = y * W + x_pos;
            r[index] = f[index] - (4 * x[index] - x[index - 1] - x[index + 1] - x[index - W] - x[index + W]) / (h_actual * h_actual);
        }
    }
}

// Compute vector norm
double vector_norm(double *v, int dim)
{
    double sum = 0.0;
    for (int i = 0; i < dim; i++)
    {
        sum += v[i] * v[i];
    }
    return sqrt(sum);
}

// Decrease grid resolution globally
void decrease_global_parameter()
{
    N /= 2;
    L = N * N;
    W = N;
    H = N;
    h = 1.0 / (N - 1);
}

void reset_global_parameter(int n)
{
    N = n;
    L = N * N;
    W = N;
    H = N;
    h = 1.0 / (N - 1);
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
                        if (x == 0 || x == output_W - 1 || y == 0 || y == output_H - 1)
                        {
                            output[y * output_W + x] = 0.0; // Enforce boundary condition
                        }
                        else
                        {
                            output[y * output_W + x] += weight[k * 3 + l] * input[i * input_W + j];
                        }
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
            if (i == 0 || i == output_H - 1 || j == 0 || j == output_W - 1) // Enforce boundary condition
            {
                output[i * output_W + j] = 0.0;
            }
            else
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
}

// Multigrid solver
void MG(double *output, double *initial_solution, double *smoother_output, double *f, double *smoother_residual, int v1, int v2, int level, int alfa = 10)
{
    int n = N, l = N * N, weight = W, height = H;
    int n_succ, l_succ, weight_succ, height_succ;
    double h_actual = h;
    double h_succ;

    // Pre-smoothing
    Jacobian_2(initial_solution, smoother_output, f, v1, height, weight, h_actual, l);

    // Coarse grid residual
    decrease_global_parameter();
    n_succ = N;
    l_succ = L;
    weight_succ = W;
    height_succ = H;
    h_succ = h;
    double *r_H = new double[L];
    initialize_zeros_vector(r_H);
    restriction(smoother_residual, r_H, height, weight, H, W);

    // Initialize vectors for coarse grid
    double *initial_solution_H = new double[L];
    double *delta_H = new double[L];
    double *smoother_output_H = new double[L];
    double *smoother_residual_H = new double[L];
    initialize_zeros_vector(initial_solution_H);
    initialize_zeros_vector(delta_H);
    initialize_zeros_vector(smoother_output_H);
    initialize_zeros_vector(smoother_residual_H);

    if (n <= 4)
    {
        Jacobian_2(initial_solution_H, delta_H, r_H, 1, H, W, h, L);
    }
    else
    {
        MG(delta_H, initial_solution_H, smoother_output_H, r_H, smoother_residual_H, v1, v2, level + 1, alfa);
        reset_global_parameter(n);
        for (int i = 0; i < alfa - 1; i++)
        {
            MG(delta_H, delta_H, smoother_output_H, r_H, smoother_residual_H, v1, v2, level + 1, alfa);
            reset_global_parameter(n);
        }
    }

    double *delta_h = new double[l];
    prolungator(delta_H, delta_h, height_succ, weight_succ, height, weight);

    for (int i = 0; i < l; i++)
    {
        smoother_output[i] += delta_h[i];
    }

    // Post-smoothing
    Jacobian_2(smoother_output, output, f, v2, height, weight, h_actual, l);
}

int main()
{
    // Save initial parameters
    int initial_N = N;
    int initial_W = W, initial_H = H, initial_L = L;
    double initial_h = h;

    double *x = new double[L];
    double *output = new double[L];
    double *smoother_output = new double[L];
    double *f = new double[L];
    double *res = new double[L];
    initialize_zeros_vector(x);
    compute_rhs(f);
    cout << "JACOBI METHOD" << endl;
    auto start = chrono::high_resolution_clock::now();
    Jacobian_2(x, output, f, 10000, H, W, h, L);
    auto end = chrono::high_resolution_clock::now();
    compute_residual(res, output, f);
    cout << "Jacobi time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
    cout << "Residual norm: " << vector_norm(res) << endl;

    int v1 = 50, v2 = 50, level = 0;
    cout << "MULTIGRID METHOD" << endl;
    initialize_zeros_vector(x);
    initialize_zeros_vector(output);
    initialize_zeros_vector(smoother_output);
    initialize_zeros_vector(res);
    auto start_MG = chrono::high_resolution_clock::now();
    MG(output, x, smoother_output, f, res, v1, v2, level, 1);
    auto end_MG = chrono::high_resolution_clock::now();
    cout << "Multigrid time: " << chrono::duration_cast<chrono::milliseconds>(end_MG - start_MG).count() << "ms" << endl;
    reset_global_parameter(initial_N);
    initialize_zeros_vector(res);
    compute_residual(res, output, f);
    cout << "Residual norm: " << vector_norm(res) << endl;
    return 0;
}