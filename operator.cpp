#include <iostream>
#include <vector>
#include <cmath>
#include "globals.hpp"

using namespace std;

// Initialize vector to zero
void initialize_vector(double *x, int dim)
{
    fill(x, x + dim, 0.0);
}

// Compute RHS function f
void compute_rhs(double *f)
{
    double dx = a / W;
    double dy = a / H;
    double factor = (M_PI * M_PI / (a * a)) * (p * p + q * q);

    for (int y = 0; y < H; y++)
    {
        for (int x = 0; x < W; x++)
        {
            double x_val = x * dx;
            double y_val = y * dy;

            f[y * W + x] = (x == 0 || x == W - 1 || y == 0 || y == H - 1) ? 0.0 : factor * sin(p * M_PI * x_val / a) * sin(q * M_PI * y_val / a);
        }
    }
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

// Perform Jacobi iterations
void Jacobi(double *x, double *x_new, double *f, double *r, int height, int weight, int l, double h_actual, int num_iter)
{
    for (int i = 0; i < num_iter; i++)
    {
        for (int y = 1; y < height - 1; y++)
        {
            for (int x_pos = 1; x_pos < weight - 1; x_pos++)
            {
                int index = y * W + x_pos;
                x_new[index] = 0.25 * ((h_actual * h_actual * f[index]) + x[index - 1] + x[index + 1] + x[index - W] + x[index + W]);
            }
        }
        copy(x_new, x_new + l, x);
    }
    compute_residual(r, x, f, weight, height, h_actual);
}

// Decrease grid resolution globally
void decrease_global_parameter()
{
    N /= 2;
    L = N * N;
    W = H = N;
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
void MG(double *output, double *initial_solution, double *smoother_output, double *f, double *smoother_residual, int v1, int v2, int level)
{
    int n = N, l = n * n, weight = n, height = n;
    double h_actual = h;
    vector<double> residual(l);
    compute_residual(residual.data(), initial_solution, f, weight, height, h_actual);

    cout << "Level: " << level << ", N: " << N << ", Initial residual: " << vector_norm(residual.data(), l) << endl;

    // Pre-smoothing
    Jacobi(initial_solution, smoother_output, f, smoother_residual, height, weight, l, h_actual, v1);

    // Coarse grid residual
    decrease_global_parameter();
    vector<double> r_H(L, 0.0);
    restriction(smoother_residual, r_H.data(), height, weight, H, W);

    vector<double> delta_H(L, 0.0);

    if (n <= 2)
    {
        Jacobi(vector<double>(L, 0.0).data(), delta_H.data(), r_H.data(), vector<double>(L, 0.0).data(), H, W, L, h, 10);
    }
    else
    {
        MG(delta_H.data(), vector<double>(L, 0.0).data(), vector<double>(L, 0.0).data(), r_H.data(), vector<double>(L, 0.0).data(), v1, v2, level + 1);
    }

    // Prolongation and correction
    vector<double> delta_h(l, 0.0);
    prolungator(delta_H.data(), delta_h.data(), H, W, height, weight);

    for (int i = 0; i < l; i++)
    {
        smoother_output[i] -= delta_h[i];
    }

    // Post-smoothing
    Jacobi(smoother_output, output, f, smoother_residual, height, weight, l, h_actual, v2);
    cout << "Output norm: " << vector_norm(output, l) << endl;
}

int main()
{
    // Save initial parameters
    int initial_W = W, initial_H = H, initial_L = L;
    double initial_h = h;

    // Allocate memory
    vector<double> solution(L, 0.0);
    vector<double> initial_guess(L, 0.0);
    vector<double> smoother_output(L, 0.0);
    vector<double> rhs(L, 0.0);
    vector<double> residual(L, 0.0);

    compute_rhs(rhs.data());

    int v1 = 4, v2 = 4, level = 0;
    // Jacobi(solution.data(), smoother_output.data(), rhs.data(), residual.data(), initial_H, initial_W, initial_L, initial_h, v1);
    MG(solution.data(), initial_guess.data(), smoother_output.data(), rhs.data(), residual.data(), v1, v2, level);
    cout << "Final computed solution (u):" << endl;
    double *output_residual = new double[initial_L];
    compute_residual(output_residual, solution.data(), rhs.data(), initial_W, initial_H, initial_h);
    cout << "Residual norm: " << vector_norm(output_residual, initial_L) << endl;
    return 0;
}