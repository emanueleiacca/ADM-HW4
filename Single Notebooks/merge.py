import nbformat

# Merge Notebooks

def merge_notebooks(output_file, *input_files):
    merged_notebook = nbformat.v4.new_notebook()

    for file in input_files:
        with open(file, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
            merged_notebook.cells.extend(notebook.cells)

    with open(output_file, 'w', encoding='utf-8') as f:
        nbformat.write(merged_notebook, f)

    print(f"Merged notebook saved to {output_file}")

merge_notebooks("main.ipynb", "Single Notebooks/Part1.ipynb", "Single Notebooks/part2and3.ipynb", "Single Notebooks/AQ.ipynb")
