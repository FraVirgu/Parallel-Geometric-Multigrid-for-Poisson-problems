import matplotlib.pyplot as plt
import os

# Create the "PLOT" directory if it doesn't exist
plot_dir = "PLOT"
os.makedirs(plot_dir, exist_ok=True)

# Read residuals from file
def read_residuals(filename):
    with open(filename, "r") as file:
        return [float(line.strip()) for line in file]

# Load residuals
jacobian_residuals = read_residuals("residuals_jacobian.txt")
cg_residuals = read_residuals("residuals_cg.txt")

# Plot residuals
plt.figure(figsize=(10, 6))
plt.plot(jacobian_residuals, label="Jacobi Method", linestyle="-", marker="o", markersize=4)
plt.plot(cg_residuals, label="Conjugate Gradient Method", linestyle="--", marker="s", markersize=4)

# Labels and Title
plt.yscale("log")  # Use logarithmic scale for better visualization
plt.xlabel("Iteration")
plt.ylabel("Residual Norm (log scale)")
plt.title("Convergence of Jacobi vs Conjugate Gradient")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Save the plot as a PDF in the "PLOT" folder
pdf_path = os.path.join(plot_dir, "convergence_plot.pdf")
plt.savefig(pdf_path, format="pdf", bbox_inches="tight")

# Show the plot
plt.show()

print(f"Plot saved successfully in {pdf_path}")