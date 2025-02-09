import os
import matplotlib.pyplot as plt

# Define the directory where the plot will be saved
plot_dir = "PLOT"
os.makedirs(plot_dir, exist_ok=True)

# Function to read errors from a file
def read_errors(file_path):
    errors = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                problem_size = int(parts[0])  # Assuming the first column is N*N
                error = float(parts[1])
                errors.append((problem_size, error))
    return errors

# File paths
errors_jacobi_file = "h_errors_jacobian.txt"
errors_steepest_file = "h_errors_steepest_descent.txt"
errors_gs_file = "h_errors_gs.txt"
errors_cg_file = "h_errors_cg.txt"

# Read errors
errors_jacobi = read_errors(errors_jacobi_file)
errors_steepest = read_errors(errors_steepest_file)
errors_gs = read_errors(errors_gs_file)
errors_cg = read_errors(errors_cg_file)

# Extract problem size and error values
sizes_jacobi, errors_jacobi_values = zip(*errors_jacobi)
sizes_steepest, errors_steepest_values = zip(*errors_steepest)
sizes_gs, errors_gs_values = zip(*errors_gs)
sizes_cg, errors_cg_values = zip(*errors_cg)

# Print results summary
def print_summary(method, sizes, errors):
    print(f"{method} Method:")
    print(f"  Largest Problem Size: {sizes[-1]}")
    print(f"  Final Error: {errors[-1]:.6f}")
    print(f"  Average Error per Problem Size: {sum(errors) / len(errors):.6f}\n")

print_summary("Jacobi", sizes_jacobi, errors_jacobi_values)
print_summary("Steepest Descent", sizes_steepest, errors_steepest_values)
print_summary("Gauss-Seidel", sizes_gs, errors_gs_values)
print_summary("Conjugate Gradient", sizes_cg, errors_cg_values)

# Plot the errors
plt.figure(figsize=(10, 6))
plt.plot(sizes_jacobi, errors_jacobi_values, label="Jacobi")
plt.plot(sizes_steepest, errors_steepest_values, label="Steepest Descent")
plt.plot(sizes_gs, errors_gs_values, label="Gauss-Seidel")
plt.plot(sizes_cg, errors_cg_values, label="Conjugate Gradient", linestyle="--")
plt.yscale("log")
plt.xlabel("Problem Size (N)")
plt.ylabel("Error Norm (log scale)")
plt.title("Error Convergence vs Problem Size for Iterative Methods")
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)

# Save the plot as a PDF in the "PLOT" folder
pdf_path = os.path.join(plot_dir, "h_errors_convergence_plot.pdf")
plt.savefig(pdf_path, format="pdf", bbox_inches="tight")

# Show the plot
plt.show()

print(f"Plot saved successfully in {pdf_path}")