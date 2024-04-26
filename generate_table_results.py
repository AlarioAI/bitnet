import json
import re
from collections import OrderedDict

import numpy as np

from bitnet.config import ProjectConfig

def sort_by_architecture(data: dict):
    sorted_keys = sorted(data.keys(), key=lambda x: (x.split("_")[1], x.split("_")[0]))
    sorted_dict = OrderedDict((key, data[key]) for key in sorted_keys)
    return sorted_dict


def generate_latex_table_and_graph(results):
    header = "\\begin{table}[h]\n" \
             "\\centering\n" \
             "\\begin{tabular}{|c|c|c|c|c|}\n" \
             "\\Xhline{2\\arrayrulewidth}\n" \
             "\\textbf{Architecture} & \\textbf{Dataset} & \\textbf{Type} & \\textbf{Mean Accuracy (\\%)} & \\textbf{Std Dev (\\%)} \\\\\n" \
             "\\Xhline{2\\arrayrulewidth}\n"
    footer = "\\end{tabular}\n" \
             "\\caption{Experimental Results Comparing BitNet and FloatNet across various architectures and datasets.}\n" \
             "\\label{tab:results}\n" \
             "\\end{table}\n\n"

    body = ""
    architectures = set()
    plot_data_params = {}
    plot_data_trainset = {}

    results = sort_by_architecture(results)

    current_architecture: str | None = None

    for experiment, models in results.items():
        dataset, architecture = experiment.split('_')

        architectures.add(architecture)

        if architecture not in plot_data_params:
            plot_data_params[architecture] = []
            plot_data_trainset[architecture] = []

        bitnet_data = models.get('BitNet', {})
        floatnet_data = models.get('FloatNet', {})
        mean_bitnet = np.mean(bitnet_data.get('scores', [0]))
        mean_floatnet = np.mean(floatnet_data.get('scores', [0]))
        std_bitnet = np.std(bitnet_data.get('scores', [0]))
        std_floatnet = np.std(floatnet_data.get('scores', [0]))
        num_params = bitnet_data.get('num_parameters', 0)
        trainset_size = bitnet_data.get('trainset_size', 0)

        discrepancy = 100 * (mean_bitnet - mean_floatnet) / mean_floatnet if mean_floatnet != 0 else 0

        plot_data_params[architecture].append((num_params, discrepancy))
        plot_data_trainset[architecture].append((trainset_size, discrepancy))

        mean_acc_bitnet = f"\\textbf{{{mean_bitnet:.2f}}}" if mean_bitnet > mean_floatnet else f"{mean_bitnet:.2f}"
        mean_acc_floatnet = f"\\textbf{{{mean_floatnet:.2f}}}" if mean_floatnet > mean_bitnet else f"{mean_floatnet:.2f}"
        std_acc_bitnet = f"\\textbf{{{std_bitnet:.2f}}}" if std_floatnet > std_bitnet else f"{std_bitnet:.2f}"
        std_acc_floatnet = f"\\textbf{{{std_floatnet:.2f}}}" if std_bitnet > std_floatnet else f"{std_floatnet:.2f}"

        if current_architecture and current_architecture != architecture:
            body += "\\Xhline{2\\arrayrulewidth}\n"

        body += f"{architecture} & {dataset} & BitNet & {mean_acc_bitnet} & {std_acc_bitnet} \\\\\n"
        body += f"{architecture} & {dataset} & FloatNet & {mean_acc_floatnet} & {std_acc_floatnet} \\\\\n"

        current_architecture = architecture

    body += "\\Xhline{2\\arrayrulewidth}\n"

    all_params = [data.get('num_parameters', 0) for models in results.values() for data in models.values()]
    all_trainset_sizes = [data.get('trainset_size', 0) for models in results.values() for data in models.values()]
    use_log_scale_params = max(all_params) / min(all_params) > 100
    use_log_scale_trainset = max(all_trainset_sizes) / min(all_trainset_sizes) > 100

    colors = ['blue', 'red', 'green', 'brown', 'cyan', 'magenta', 'orange', 'black', 'purple', 'yellow']
    color_map = {arch: color for arch, color in zip(sorted(architectures), colors)}

    graph_params_footer = "\\begin{figure}[h]\n" \
                          "\\centering\n" \
                          "\\begin{tikzpicture}\n" \
                          "\\begin{axis}[\n" \
                          "xlabel={Number of Parameters},\n" \
                          "ylabel={Discrepancy (\%)}," + "\n" \
                          "legend style={at={(0.5,-0.20)},anchor=north,legend columns=-1},\n" \
                          "grid=major," + "\n"

    if use_log_scale_params:
        graph_params_footer += "xmode=log,\nlog basis x={10},\n"

    graph_params_footer += "]\n"
    for arch, color in color_map.items():
        graph_params_footer += f"\\addplot[only marks, mark=*, color={color}] coordinates {{\n"
        for params, discrepancy in plot_data_params[arch]:
            graph_params_footer += f"({params},{discrepancy})\n"
        graph_params_footer += f"}};\n\\addlegendentry{{{arch}}}\n"

    graph_params_footer += "\\end{axis}\n" \
                           "\\end{tikzpicture}\n" \
                           "\\caption{Correlation between number of parameters and metric discrepancy for BitNet and FloatNet.}\n" \
                           "\\label{fig:params_discrepancy_plot}\n" \
                           "\\end{figure}\n"

    graph_trainset_footer = "\\begin{figure}[h]\n" \
                            "\\centering\n" \
                            "\\begin{tikzpicture}\n" \
                            "\\begin{axis}[\n" \
                            "xlabel={Training Dataset Size},\n" \
                            "ylabel={Discrepancy (\%)}," + "\n" \
                            "legend style={at={(0.5,-0.20)},anchor=north,legend columns=-1},\n" \
                            "grid=major," + "\n"

    if use_log_scale_trainset:
        graph_trainset_footer += "xmode=log,\nlog basis x={10},\n"

    graph_trainset_footer += "]\n"
    for arch, color in color_map.items():
        graph_trainset_footer += f"\\addplot[only marks, mark=*, color={color}] coordinates {{\n"
        for trainset_size, discrepancy in plot_data_trainset[arch]:
            graph_trainset_footer += f"({trainset_size},{discrepancy})\n"
        graph_trainset_footer += f"}};\n\\addlegendentry{{{arch}}}\n"

    graph_trainset_footer += "\\end{axis}\n" \
                             "\\end{tikzpicture}\n" \
                             "\\caption{Correlation between training dataset size and metric discrepancy for BitNet and FloatNet.}\n" \
                             "\\label{fig:trainset_discrepancy_plot}\n" \
                             "\\end{figure}\n"

    return f"{header}{body}{footer}{graph_params_footer}{graph_trainset_footer}"


def insert_table_into_document(document_path, table_data):
    with open(document_path, 'r') as file:
        content = file.read()

    pattern = r'% BEGIN_TABLE.*?% END_TABLE'
    replacement = f'% BEGIN_TABLE\n{table_data}\n% END_TABLE'
    replacement = replacement.replace('\\', '\\\\')
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    with open(document_path, 'w') as file:
        file.write(new_content)


def main():
    results_file = ProjectConfig.RESULTS_FILE
    latex_document_path = ProjectConfig.PAPER_TEX_PATH

    with open(results_file, 'r') as f:
        results = json.load(f)

    table_data = generate_latex_table_and_graph(results)
    insert_table_into_document(latex_document_path, table_data)


if __name__ == '__main__':
    main()
