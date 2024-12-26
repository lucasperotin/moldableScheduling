import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from math import *

# Create directory to save the plots
output_dir = 'plotsgen'
os.makedirs(output_dir, exist_ok=True)

# Initialization of parameters
w_values = np.arange(4, 801, 1)
d_values = np.arange(0, 801, 1)
p_values = [2,3,4,6,8]

def compute_t_p(w, p, d):
    """Calculate t(p) for parameters w, p, g."""
    return w / p + d + p-1 

def compute_t_min(w, g):
    """Calculate the minimum t(p) for a given pair (w, g) considering integer p."""
    p_opt = sqrt(w)
    p_floor = int(np.floor(p_opt))
    p_ceil = int(np.ceil(p_opt))
    if p_floor < 1:
        p_floor = 1  # Ensure p is within the valid range
    t_floor = compute_t_p(w, p_floor, g)
    t_ceil = compute_t_p(w, p_ceil, g)
    return min(t_floor, t_ceil) 


def compute_p_min(w, g):
    """Calculate the minimum t(p) for a given pair (w, g) considering integer p."""
    p_opt = sqrt(w)
    p_floor = int(np.floor(p_opt))
    p_ceil = int(np.ceil(p_opt))
    if p_floor < 1:
        p_floor = 1  # Ensure p is within the valid range
    t_floor = compute_t_p(w, p_floor, g)
    t_ceil = compute_t_p(w, p_ceil, g)
    if (t_floor<=t_ceil):
        return p_floor
    else:
        return p_ceil

def compute_alpha_p(w, p, d):
    """Calculate alpha(p) for parameters w, p, g."""
    a_p = w + p * d + p * (p - 1)
    amin = w + d
    return a_p / amin 

def compute_beta_p(w, p, g, t_min):
    """Calculate beta(p) for parameters w, p, g with given t_min."""
    t_p = compute_t_p(w, p, g)
    return t_p / t_min 

custom_cmap = ListedColormap(['green', 'red'])

label_counter = 1  # Initialize label counter
# Generate individual plots for each p
for p in p_values:
    fig, ax = plt.subplots(figsize=(6, 4))
    color_matrix = np.zeros((len(w_values), len(d_values)))

    for i, w in enumerate(w_values):
        for j, d in enumerate(d_values):
            t_min = compute_t_min(w, d)
            alpha = compute_alpha_p(w, p, d)
            beta = compute_beta_p(w, p, d, t_min)
            if alpha <= 2.02 and beta <= 2.02:
                color_matrix[i, j] = 1  # Mark regions satisfying the conditions 

    # Create contour plot for this p
    W, G = np.meshgrid(w_values, d_values)
    ax.contourf(W, G, color_matrix.T, levels=[0, 0.5], cmap=custom_cmap)

    # Function to add rectangles and labels
    def add_rectangle_and_label(ax, xy, width, height, label):
        rect = Rectangle(xy, width, height, linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3, linestyle='-')
        ax.add_patch(rect)
        cx = xy[0] + width / 2  # Center x
        cy = xy[1] + height / 2  # Center y
        ax.text(cx, cy, str(label), color='black', weight='bold', fontsize=12, ha='center', va='center')

    # Add rectangles and labels
    if p == 2:
        add_rectangle_and_label(ax, (49, 330), 751, 470, label_counter)
        label_counter += 1
        add_rectangle_and_label(ax, (185, 170), 300, 160, label_counter)
        label_counter += 1
        add_rectangle_and_label(ax, (49, 42), 142, 286, label_counter)
        label_counter += 1
        add_rectangle_and_label(ax, (4, 0), 42, 800, label_counter)
        label_counter += 1
    elif p == 3:
        add_rectangle_and_label(ax, (485, 200), 315, 130, label_counter)
        label_counter += 1
        add_rectangle_and_label(ax, (185, 80), 300, 90, label_counter)
        label_counter += 1
        add_rectangle_and_label(ax, (49, 0), 66, 42, label_counter)
        label_counter += 1
    elif p == 4:
        add_rectangle_and_label(ax, (485, 105), 315, 95, label_counter)
        label_counter += 1
        add_rectangle_and_label(ax, (185, 42), 300, 38, label_counter)
        label_counter += 1
        add_rectangle_and_label(ax, (115, 0), 100, 42, label_counter)
        label_counter += 1
    elif p == 6:
        add_rectangle_and_label(ax, (485, 50), 315, 55, label_counter)
        label_counter += 1
        add_rectangle_and_label(ax, (215, 0), 270, 42, label_counter)
        label_counter += 1
    elif p == 8:
        add_rectangle_and_label(ax, (485, 0), 315, 42, label_counter)
        label_counter += 1

    # Set axis limits
    ax.set_xlim([4, 800])
    ax.set_ylim([0, 800])

    # Add labels, title
    ax.set_xlabel("w")
    ax.set_ylabel("g")
    ax.set_title(f"Regions satisfying conditions for p = {p}")

    plt.savefig(f'{output_dir}/plot_p_{p}.png')
    plt.close(fig)

# Create a figure with only the rectangles
fig, ax = plt.subplots(figsize=(6, 4))

# Reset label counter for rectangles only plot
label_counter = 1

# Add all rectangles from the different p values
rectangles = [
    ((49, 330), 751, 570),       # For p = 2
    ((185, 170), 300, 160),      # For p = 2
    ((49, 42), 136, 286),        # For p = 2
    ((4, 0), 45, 800),        # For p = 2
    ((485, 200), 315, 130),      # For p = 3
    ((185, 80), 300, 90),        # For p = 3
    ((49, 0), 66, 42),           # For p = 3
    ((485, 105), 315, 95),       # For p = 4
    ((185, 42), 300, 38),        # For p = 4
    ((115, 0), 100, 42),         # For p = 4
    ((485, 50), 315, 55),        # For p = 6
    ((215, 0), 270, 42),         # For p = 6
    ((485, 0), 315, 50)          # For p = 8
]

rectangles2 = [
    (1, 49, 800, 800, 330, 2),  # Top-left (49, 900), bottom-right (800, 330), for p = 2
    (2, 185, 330, 485, 170, 2), # Top-left (185, 330), bottom-right (485, 170), for p = 2
    (3, 49, 330, 185, 42, 2),   # Top-left (49, 330), bottom-right (185, 44), for p = 2
    (4, 4, 800, 49, 0, 2),   # Top-left (49, 330), bottom-right (185, 44), for p = 2
    (5, 485, 330, 800, 200, 3), # Top-left (485, 330), bottom-right (800, 200), for p = 3
    (6, 185, 170, 485, 80, 3),  # Top-left (185, 170), bottom-right (485, 90), for p = 3
    (7, 49, 42, 115, 0, 3),     # Top-left (49, 44), bottom-right (115, 0), for p = 3
    (8, 485, 200, 800, 105, 4), # Top-left (485, 200), bottom-right (800, 115), for p = 4
    (9, 185, 80, 485, 42, 4),   # Top-left (185, 90), bottom-right (485, 47), for p = 4
    (10, 115, 42, 215, 0, 4),    # Top-left (115, 44), bottom-right (215, 0), for p = 4
    (11, 485, 105, 800, 50, 6), # Top-left (485, 115), bottom-right (800, 50), for p = 6
    (12, 215, 42, 485, 0, 6),   # Top-left (215, 44), bottom-right (485, 0), for p = 6
    (13, 485, 50, 800, 0, 8)    # Top-left (485, 65), bottom-right (785, 0), for p = 8
]


for xy, width, height in rectangles:
    add_rectangle_and_label(ax, xy, width, height, label_counter)
    label_counter += 1

# Set axis limits
ax.set_xlim([4, 800])
ax.set_ylim([0, 800])

# Add labels, title
ax.set_xlabel("w")
ax.set_ylabel("d")
ax.set_title("All Rectangles")

plt.savefig(f'{output_dir}/rectanglesall.png')
plt.close(fig)
def generate_latex_table():
    table = "\\begin{tabular}{|c|c|c|c|c|c|c|}\n\\hline\n"
    table += "Rectangle & $(w_1,\\gamma_1)$ & $(w_2,\\gamma_2)$ & $p$ & $p_{min}(w_1,\\gamma_1,p) $& $f(w_1,\\gamma_1,p)$ & $g(w_2,\\gamma_2,p)$ \\\\\n\\hline\n"
    
    for rect in rectangles2:
        num, w1, g1, w2, g2, p = rect
        f_val = compute_alpha_p(w1, p, g1)
        tmin=compute_t_min(w2, g2)
        g_val = compute_beta_p(w2, p, g2,tmin)
        pmin=compute_p_min(w1, g1)
        print(w2,g2,p, pmin,compute_t_p(w1,p,g1),  tmin,g_val)
        
        row = f"{num} & $({w1},{g1})$ & $({w2},{g2})$ & {p} &{pmin} & {f_val:.2f} & {g_val:.2f} \\\\\n"
        table += row
    
    table += "\\hline\n\\end{tabular}"
    return table

latex_table = generate_latex_table()
print(latex_table)

