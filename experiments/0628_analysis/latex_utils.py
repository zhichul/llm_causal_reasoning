from pathlib import Path

def make_latex_table(col_labels, row_labels, values, bold_flags,
                     file_name="table.tex", value_fmt="{:.3f}",
                     caption=None, label=None, font='\\footnotesize'):
    """
    Build a LaTeX table (optionally wrapped in a table float) and save it.

    Parameters
    ----------
    col_labels  : list[str]            # column legend
    row_labels  : list[str]            # row legend
    values      : list[list[float]]    # one list per column, len == len(row_labels)
    bold_flags  : list[list[bool]]     # same shape as `values`
    file_name   : str | Path           # output .tex file
    value_fmt   : str                  # e.g. "{:.2f}", "{:d}", etc.
    caption     : str | None           # if given, adds \caption{...}
    label       : str | None           # if given, adds \label{...}
    """
    n_rows = len(row_labels)
    n_cols = len(col_labels)

    if any(len(col) != n_rows for col in values):
        raise ValueError("All value lists must have length equal to len(row_labels)")
    if any(len(col) != n_rows for col in bold_flags):
        raise ValueError("All bold_flag lists must mirror values shape")

    parts = []

    # Optional outer table environment for caption/label
    if caption or label:
        parts.append(r"\begin{table}[htbp]")
        parts.append(r"\centering")
        if caption:
            parts.append(rf"\caption{{{caption}}}")
        if label:
            parts.append(rf"\label{{{label}}}")

    # Core tabular
    parts.append(font+ r"\begin{tabular}{l" + "c" * n_cols + "}")
    parts.append(r"\toprule")
    parts.append(" & ".join([""] + col_labels) + r" \\")
    parts.append(r"\midrule")

    for i, rlabel in enumerate(row_labels):
        row_cells = []
        for j in range(n_cols):
            cell = value_fmt.format(values[j][i])
            if bold_flags[j][i]:
                cell = r"\textbf{" + cell + "}"
            row_cells.append(cell)
        parts.append(" & ".join([rlabel] + row_cells) + r" \\")

    parts.append(r"\bottomrule")
    parts.append(r"\end{tabular}")

    if caption or label:
        parts.append(r"\end{table}")

    latex_str = "\n".join(parts)
    Path(file_name).write_text(latex_str)
    print(f"LaTeX table written to {file_name}")

    return latex_str