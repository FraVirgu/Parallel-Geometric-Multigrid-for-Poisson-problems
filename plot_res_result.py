import os
import matplotlib.pyplot as plt

# Define the directory where the plot will be saved
plot_dir = "PLOT"
os.makedirs(plot_dir, exist_ok=True)

# Read residuals from files
def read_residuals(file_path):
    with open(file_path, 'r') as file:
        residuals = [float(line.strip()) for line in file]
    return residuals

# File paths
jacobi_file = "residuals_jacobian.txt"
cg_file = "residuals_cg.txt"
gs_file = "residuals_gs.txt"

# Read residuals
residuals_jacobi = read_residuals(jacobi_file)
residuals_cg = read_residuals(cg_file)
residuals_gs = read_residuals(gs_file)

# Plot residuals
plt.figure(figsize=(10, 6))
plt.plot(residuals_jacobi, label="Jacobi")
plt.plot(residuals_cg, label="Conjugate Gradient")
plt.plot(residuals_gs, label="Gauss-Seidel")
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Residual Norm (log scale)")
plt.title("Convergence of Jacobi vs Conjugate Gradient vs Gauss-Seidel")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Save the plot as a PDF in the "PLOT" folder
pdf_path = os.path.join(plot_dir, "RES_convergence_plot.pdf")
plt.savefig(pdf_path, format="pdf", bbox_inches="tight")

# Show the plot
plt.show()

print(f"Plot saved successfully in {pdf_path}")