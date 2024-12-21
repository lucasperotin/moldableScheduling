import os
import csv
import pandas as pd
from parameters import *

def generate_latex_report(filename="results_visualization.tex"):
    latex_content = [
        r"\documentclass{article}",
        r"\usepackage{graphicx}",
        r"\usepackage{subfig}",
        r"\usepackage[margin=1.95cm]{geometry}",
        r"\usepackage{booktabs}",  # For better-looking tables
        r"\title{Results Visualization}",
        r"\author{Your Name}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
        r"\tableofcontents",
        r"\newpage"
    ]

    sections = ["Lines Figures", "Boxplot Figures", "Bounds Figures"]
    figure_types = ["", "_boxplot", "_bounds"]

    # Generate figures
    for section, figure_type in zip(sections, figure_types):
        latex_content.extend([
            f"\n\section{{{section}}}",
        ])

        for param in parameters:
            latex_content.extend([
                f"\n\subsection{{{param}}}",
                r"\begin{figure}[htbp]",
                r"\centering"
            ])


            
            for i, model in enumerate(MODEL_LIST):
                # Append subfigure with the model's name
                latex_content.append(
                    f"\\subfloat[{model.name}]{{\\includegraphics[width=0.32\\textwidth]{{Results/{param}/{param}_{model.name}{figure_type}}}}}"
                )
            
                # Add a new line after every third item for a new row
                if i == 2:
                    latex_content.append(r"\\[2ex]")
                    # Place \hspace*{\fill} after the first and before the last element in the second row
                    latex_content.append(r"\hspace*{\fill}")  # Start with fill for the second row
                elif i < 2:  # First row between figures
                    latex_content.append("\\hfill")
                elif i == 3:  # Between figures in the second row
                    latex_content.append("\\hfill")
            
            # Ending with \hspace*{\fill} after the last figure to ensure centering on the second row
            if len(MODEL_LIST) == 5:
                latex_content.append(r"\hspace*{\fill}")



            latex_content.extend([
                r"\caption{" + f"{section.rstrip('s')} for {param}" + r"}",
                f"\label{{fig:{section.lower().replace(' ', '_')}_{param}}}",
                r"\end{figure}",
            ])

    # Generate table
    latex_content.extend([
        r"\newpage",
        r"\section{Average Values}",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Average Values for Each Model and Heuristic}",
        r"\label{tab:average_values}",
        r"\begin{tabular}{l" + "r" * len(Heuristics) + "}",
        r"\toprule",
        r"Model & " + " & ".join(Heuristics) + r" \\"
    ])

    # Calculate averages
    model_averages = {}
    for model in MODEL_LIST:
        model_averages[model.name] = []
        for heuristic in Heuristics:
            heuristic_values = []
            for param in parameters:
                csv_path = f"Results/{param}/{model.name}/mean.csv"
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path, header=None)
                    heuristic_index = Heuristics.index(heuristic) + 1  # +1 because the first column is the parameter value
                    heuristic_values.extend(df.iloc[:, heuristic_index].tolist())
            avg_value = sum(heuristic_values) / len(heuristic_values) if heuristic_values else 0
            model_averages[model.name].append(avg_value)

    # Add rows to the table
    for model in MODEL_LIST:
        row = f"{model.name} & " + " & ".join(f"{value:.2f}" for value in model_averages[model.name]) + r" \\"
        latex_content.append(row)

    # Calculate and add the average row
    avg_row = ["Average"]
    for i in range(len(Heuristics)):
        column_avg = sum(model_averages[model.name][i] for model in MODEL_LIST) / len(MODEL_LIST)
        avg_row.append(f"{column_avg:.2f}")
    
    latex_content.extend([
        r"\midrule",
        " & ".join(avg_row) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])

    # Generate maximum values table
    latex_content.extend([
        r"\newpage",
        r"\section{Maximum Values}",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Maximum Values for Each Model and Heuristic}",
        r"\label{tab:maximum_values}",
        r"\begin{tabular}{l" + "r" * len(Heuristics) + "}",
        r"\toprule",
        r"Model & " + " & ".join(Heuristics) + r" \\"
    ])

    # Calculate maximums
    model_maximums = {}
    for model in MODEL_LIST:
        model_maximums[model.name] = []
        for heuristic in Heuristics:
            heuristic_values = []
            for param in parameters:
                csv_path = f"Results/{param}/{model.name}/max.csv"
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path, header=None)
                    heuristic_index = Heuristics.index(heuristic) + 1  # +1 because the first column is the parameter value
                    heuristic_values.extend(df.iloc[:, heuristic_index].tolist())
            max_value = max(heuristic_values) if heuristic_values else 0
            model_maximums[model.name].append(max_value)

    # Add rows to the maximum values table
    for model in MODEL_LIST:
        row = f"{model.name} & " + " & ".join(f"{value:.2f}" for value in model_maximums[model.name]) + r" \\"
        latex_content.append(row)
   # Calculate and add the maximum row

    max_row = ["Maximum"]
    for i in range(len(Heuristics)):
        column_max = max(model_maximums[model.name][i] for model in MODEL_LIST)
        max_row.append(f"{column_max:.2f}")
    
    
    latex_content.extend([
        r"\midrule",
        " & ".join(max_row) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])
    
    

    latex_content.append(r"\end{document}")

    with open(filename, "w") as f:
        f.write("\n".join(latex_content))

# Your existing parameters, model_list, and heuristics
