#include <iostream>
#include <iomanip>

using namespace std;

// Function prototypes
void prolungator(double *input, double *output, int input_H, int input_W, int output_H, int output_W);
void restriction(double *input, double *output, int input_H, int input_W, int output_H, int output_W);

// Function to print a 2D grid
void printGrid(double *grid, int H, int W, const string &name)
{
    cout << "\n"
         << name << " (" << H << "x" << W << "):\n";
    for (int i = 0; i < H; i++)
    {
        for (int j = 0; j < W; j++)
        {
            cout << setw(6) << fixed << setprecision(2) << grid[i * W + j] << " ";
        }
        cout << endl;
    }
}

int main()
{
    // Define grid sizes
    int fine_W = 8, fine_H = 8; // Fine grid size
    int coarse_W = fine_W / 2;  // Coarse grid size
    int coarse_H = fine_H / 2;

    // Allocate memory
    double *fine_grid = new double[fine_W * fine_H];
    double *coarse_grid = new double[coarse_W * coarse_H];
    double *prolonged_grid = new double[fine_W * fine_H];

    // Initialize fine grid with internal values, enforce boundary = 0
    for (int i = 0; i < fine_H; i++)
    {
        for (int j = 0; j < fine_W; j++)
        {
            if (i == 0 || i == fine_H - 1 || j == 0 || j == fine_W - 1)
            {
                fine_grid[i * fine_W + j] = 0.0; // Boundary condition
            }
            else
            {
                fine_grid[i * fine_W + j] = (i + 1) * (j + 1); // Example function inside domain
            }
        }
    }

    // Initialize output grids to zero
    for (int i = 0; i < coarse_W * coarse_H; i++)
        coarse_grid[i] = 0.0;
    for (int i = 0; i < fine_W * fine_H; i++)
        prolonged_grid[i] = 0.0;

    // Print original fine grid
    printGrid(fine_grid, fine_H, fine_W, "Original Fine Grid");

    // Apply restriction (Fine -> Coarse)
    restriction(fine_grid, coarse_grid, fine_H, fine_W, coarse_H, coarse_W);
    printGrid(coarse_grid, coarse_H, coarse_W, "Coarse Grid (After Restriction)");

    // Apply prolongation (Coarse -> Fine)
    prolungator(coarse_grid, prolonged_grid, coarse_H, coarse_W, fine_H, fine_W);

    // Enforce boundary conditions after prolongation
    for (int i = 0; i < fine_H; i++)
    {
        for (int j = 0; j < fine_W; j++)
        {
            if (i == 0 || i == fine_H - 1 || j == 0 || j == fine_W - 1)
            {
                prolonged_grid[i * fine_W + j] = 0.0;
            }
        }
    }

    printGrid(prolonged_grid, fine_H, fine_W, "Fine Grid (After Prolongation)");

    // Clean up memory
    delete[] fine_grid;
    delete[] coarse_grid;
    delete[] prolonged_grid;

    return 0;
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
