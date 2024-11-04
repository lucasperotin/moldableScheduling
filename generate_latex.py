import os

def generate_latex_report(parameters, model_list):
    latex_content = [
        r"\documentclass{article}",
        r"\usepackage{graphicx}",
        r"\usepackage{subfig}",
        r"\usepackage[margin=1.95cm]{geometry}",
        r"\title{Results Visualization}",
        r"\author{Your Name}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
        r"\tableofcontents",
        r"\newpage"
    ]

    sections = ["Lines Figures", "Boxplot Figures", "Bounds Figures"]
    figure_types = ["", "_boxplot", "bounds"]

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

            for i, model in enumerate(model_list):
                if i % 4 == 0 and i > 0:
                    latex_content.append(r"\\[2ex]")
                latex_content.append(f"\subfloat[{model.name}]{{\includegraphics[width=0.23\\textwidth]{{Results/{param}/{param}_{model.name}{figure_type}}}}}")
                if i % 4 != 3 and i < len(model_list) - 1:
                    latex_content.append("\\hfill")

            latex_content.extend([
                r"\caption{" + f"{section.rstrip('s')} for {param}" + r"}",
                f"\label{{fig:{section.lower().replace(' ', '_')}_{param}}}",
                r"\end{figure}",
            ])

    latex_content.append(r"\end{document}")

    with open("results_visualization.tex", "w") as f:
        f.write("\n".join(latex_content))

# Your existing parameters and model_list