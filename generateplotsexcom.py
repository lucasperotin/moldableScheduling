import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle



# Create directory to save the plots
output_dir = 'plotsexcom'
os.makedirs(output_dir, exist_ok=True)

# Initialization of parameters
w_values = np.arange(2, 8.01, 0.01)
g_values = np.arange(0, 1.001, 0.001)
p_values = range(2, 9)

def compute_t_p(w, p, g):
    """Calculate t(p) for parameters w, p, g."""
    return w / p + p ** g 

def compute_t_min(w, g):
    """Calculate the minimum t(p) for a given pair (w, g) considering integer p."""
    if g == 0:  # Handle the special case where g = 0
        return 1
    p_opt = (w / g) ** (1 / (g + 1))
    p_floor = int(np.floor(p_opt))
    p_ceil = int(np.ceil(p_opt))
    if p_floor < 1:
        p_floor = 1  # Ensure p is within the valid range
    t_floor = compute_t_p(w, p_floor, g)
    t_ceil = compute_t_p(w, p_ceil, g)
    return min(t_floor, t_ceil) 

def compute_alpha_p(w, p, g):
    """Calculate alpha(p) for parameters w, p, g."""
    a_p = w + p ** (g + 1)
    amin = w + 1
    return a_p / amin 

def compute_beta_p(w, p, g, t_min):
    """Calculate beta(p) for parameters w, p, g with given t_min."""
    t_p = compute_t_p(w, p, g)
    return t_p / t_min 

custom_cmap = ListedColormap(['green', 'red'])

label_counter=1
# Generate individual plots for each p
for p in p_values:
    fig, ax = plt.subplots(figsize=(6, 4))
    color_matrix = np.zeros((len(w_values), len(g_values)))

    for i, w in enumerate(w_values):
        for j, g in enumerate(g_values):
            t_min = compute_t_min(w, g)
            alpha = compute_alpha_p(w, p, g)
            beta = compute_beta_p(w, p, g, t_min)
            #print(w,g,alpha,beta,p)
            if alpha <= 2.005 and beta <= 2.005:
                color_matrix[i, j] = 1  # Mark regions satisfying the conditions 

    # Create contour plot for this p
    W, G = np.meshgrid(w_values, g_values)
    contour_set = ax.contourf(W, G, color_matrix.T, levels=[0, 0.5], cmap=custom_cmap)
    if p == 2:
        rect1 = Rectangle((2, 0.3), 6, 0.7, linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3, linestyle='-')
        ax.add_patch(rect1)
        ax.text(5, 0.65, f"{label_counter}", fontsize=12, ha='center', va='center')
        label_counter += 1
        rect2 = Rectangle((2, 0.2), 1, 0.1, linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3, linestyle='-')
        ax.add_patch(rect2)
        ax.text(2.5, 0.25, f"{label_counter}", fontsize=12, ha='center', va='center')
        label_counter += 1

        

    if p == 3:
        rect4 = Rectangle((3, 0.15), 5, 0.15, linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3, linestyle='-')
        ax.add_patch(rect4)
        ax.text(5.5, 0.225, f"{label_counter}", fontsize=12, ha='center', va='center')
        label_counter += 1
        rect3 = Rectangle((2, 0), 1, 0.2, linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3, linestyle='-')
        ax.add_patch(rect3)
        ax.text(2.5, 0.1, f"{label_counter}", fontsize=12, ha='center', va='center')
        label_counter += 1
    
    if p == 4:
        rect6 = Rectangle((3, 0.1), 5, 0.05, linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3, linestyle='-')
        ax.add_patch(rect6)
        ax.text(5.5, 0.125, f"{label_counter}", fontsize=12, ha='center', va='center')
        label_counter += 1
        rect5 = Rectangle((3, 0), 1, 0.1, linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3, linestyle='-')
        ax.add_patch(rect5)
        ax.text(3.5, 0.05, f"{label_counter}", fontsize=12, ha='center', va='center')
        label_counter += 1
    
    if p == 5:
        rect8 = Rectangle((4, 0.075), 4, 0.025, linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3, linestyle='-')
        ax.add_patch(rect8)
        ax.text(6, 0.0875, f"{label_counter}", fontsize=12, ha='center', va='center')
        label_counter += 1
        rect7 = Rectangle((4, 0), 1, 0.075, linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3, linestyle='-')
        ax.add_patch(rect7)
        ax.text(4.5, 0.0375, f"{label_counter}", fontsize=12, ha='center', va='center')
        label_counter += 1
    
    if p == 6:
        rect10 = Rectangle((5, 0.05), 3, 0.025, linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3, linestyle='-')
        ax.add_patch(rect10)
        ax.text(6.5, 0.0625, f"{label_counter}", fontsize=12, ha='center', va='center')
        label_counter += 1
        rect9 = Rectangle((5, 0), 1, 0.05, linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3, linestyle='-')
        ax.add_patch(rect9)
        ax.text(5.5, 0.025, f"{label_counter}", fontsize=12, ha='center', va='center')
        label_counter += 1
    
    if p == 7:
        rect11 = Rectangle((6, 0), 1, 0.05, linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3, linestyle='-')
        ax.add_patch(rect11)
        ax.text(6.5, 0.025, f"{label_counter}", fontsize=12, ha='center', va='center')
        label_counter += 1
    
    if p == 8:
        rect12 = Rectangle((7, 0), 1, 0.05, linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3, linestyle='-')
        ax.add_patch(rect12)
        ax.text(7.5, 0.025, f"{label_counter}", fontsize=12, ha='center', va='center')
        label_counter += 1




    # Add labels, title
    ax.set_xlabel("w")
    ax.set_ylabel("g")
    ax.set_title(f"Regions satisfying conditions for p = {p}")

    plt.savefig(f'{output_dir}/plot_p_{p}.png')
    plt.close(fig)
    
# Create a new figure to combine all rectangles
fig, ax = plt.subplots(figsize=(8, 6))

# Track label numbers
label_counter = 1

# Add rectangles from all plots
# These positions and sizes are based on your previous Python code setup
# Adjust coordinates and dimensions as necessary

# Rectangles for p = 2
ax.add_patch(Rectangle((2, 0.3), 6, 0.7, linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3, linestyle='-'))
ax.text(5, 0.65, "1", fontsize=12, ha='center', va='center')
ax.add_patch(Rectangle((2, 0.2), 1, 0.1, linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3 , linestyle='-'))
ax.text(2.5, 0.25, "2", fontsize=12, ha='center', va='center')
# Rectangles for p = 3
ax.add_patch(Rectangle((2, 0), 1, 0.2, linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3, linestyle='-'))
ax.text(2.5, 0.1, "4", fontsize=12, ha='center', va='center')
ax.add_patch(Rectangle((3, 0.15), 5, 0.15, linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3, linestyle='-'))
ax.text(5.5, 0.225, "3", fontsize=12, ha='center', va='center')
# Rectangles for p = 4
ax.add_patch(Rectangle((3, 0), 1, 0.1, linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3, linestyle='-'))
ax.text(3.5, 0.05, "6", fontsize=12, ha='center', va='center')
ax.add_patch(Rectangle((3, 0.1), 5, 0.05, linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3, linestyle='-'))
ax.text(5.5, 0.125, "5", fontsize=12, ha='center', va='center')
# Rectangles for p = 5
ax.add_patch(Rectangle((4, 0), 1, 0.075, linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3, linestyle='-'))
ax.text(4.5, 0.0375, "8", fontsize=12, ha='center', va='center')
ax.add_patch(Rectangle((4, 0.075), 4, 0.025, linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3, linestyle='-'))
ax.text(6, 0.0875, "7", fontsize=12, ha='center', va='center')
# Rectangles for p = 6
ax.add_patch(Rectangle((5, 0), 1, 0.05, linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3, linestyle='-'))
ax.text(5.5, 0.025, "10", fontsize=12, ha='center', va='center')
ax.add_patch(Rectangle((5, 0.05), 3, 0.025, linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3, linestyle='-'))
ax.text(6.5, 0.0625, "9", fontsize=12, ha='center', va='center')
# Rectangles for p = 7
ax.add_patch(Rectangle((6, 0), 1, 0.05, linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3, linestyle='-'))
ax.text(6.5, 0.025, "11", fontsize=12, ha='center', va='center')
# Rectangles for p = 8
ax.add_patch(Rectangle((7, 0), 1, 0.05, linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3, linestyle='-'))
ax.text(7.5, 0.025, "12", fontsize=12, ha='center', va='center')

# Set limits to frame all rectangles properly
ax.set_xlim(2, 8)
ax.set_ylim(0, 1)
ax.set_xlabel("w")
ax.set_ylabel("g")
ax.set_title("Combined Rectangles from All Plots")
ax.grid(False)  # Optionally remove grid

# Display the plot with all rectangles
plt.savefig(f'{output_dir}/rectanglesall.png')




def compute_p_min(w, g):
    """Calculate the minimum t(p) for a given pair (w, g) considering integer p."""
    if g == 0:  # Handle the special case where g = 0
        return 1
    p_opt = (w / g) ** (1 / (g + 1))
    p_floor = int(np.floor(p_opt))
    p_ceil = int(np.ceil(p_opt))
    if p_floor < 1:
        p_floor = 1  # Ensure p is within the valid range
    t_floor = compute_t_p(w, p_floor, g)
    t_ceil = compute_t_p(w, p_ceil, g)
    if t_floor<=t_ceil:
        return p_floor
    else:
        return p_ceil
    return min(t_floor, t_ceil) 


def gg(w, p, gamma):
    if gamma==0:
        return ((w/p) + p**gamma)
    return ((w/p) + p**gamma) / (compute_t_min(w, gamma))

def f(w, p, gamma):
    return (w + p**(gamma + 1)) / (w + 1)


def generate_latex_table():
    table = "\\begin{tabular}{|c|c|c|c|c|c|c|}\n\\hline\n"
    table += "Rectangle & $(w_1,\\gamma_1)$ & $(w_2,\\gamma_2)$ & $p$ & $p_{min}(w_1,\\gamma_1,p) $& $f(w_1,\\gamma_1,p)$ & $g(w_2,\\gamma_2,p)$ \\\\\n\\hline\n"
    
    for rect in rectangles:
        num, w1, g1, w2, g2, p = rect
        f_val = f(w1, p, g1)
        g_val = gg(w2, p, g2)
        pmin=compute_p_min(w1, g1)
        
        row = f"{num} & $({w1},{g1:.3f})$ & $({w2},{g2:.3f})$ & {p} &{pmin} & {f_val:.2f} & {g_val:.2f} \\\\\n"
        table += row
    
    table += "\\hline\n\\end{tabular}"
    return table

rectangles = [
    (1, 2, 1.0, 8, 0.3, 2),
    (2, 2, 0.3, 3, 0.2, 2),
    (3, 3, 0.3, 8, 0.15, 3),
    (4, 2, 0.2, 3, 0.0, 3),
    (5, 3, 0.15, 8, 0.1, 4),
    (6, 3, 0.1, 4, 0.0, 4),
    (7, 4, 0.1, 8, 0.075, 5),
    (8, 4, 0.075, 5, 0.0, 5),
    (9, 5, 0.075, 8, 0.05, 6),
    (10, 5, 0.05, 6, 0.0, 6),
    (11, 6, 0.05, 7, 0.0, 7),
    (12, 7, 0.05, 8, 0.0, 8)
]
latex_table = generate_latex_table()
print(latex_table)