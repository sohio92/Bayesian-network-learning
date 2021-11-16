import matplotlib.pyplot as plt
import os

from progress.bar import Bar

from pc import PC
from pc_stable import PC_stable
from pc_ccs_orientation import PC_ccs_orientation
from pc_ccs_skeleton import PC_ccs_skeleton

import numpy as np
import matplotlib.pyplot as plt

from benchmark import Benchmark

from utils import *

def save(content, filename):
    """Saves the results in a file"""
    with lzma.open(filename+".lzma", 'wb') as file:
        pickle.dump(content, file)


def run(benchmark, algorithm=PC, bar=None):
    """Returns the results of one algorithm on the given benchmark"""
    alpha_results = {}
    for alpha, result in benchmark.run_alpha_test(algorithm=algorithm, bar=bar).items():
        alpha_results[alpha] = unpack_results(result)
    return alpha_results


def run_algorithms(benchmark, algorithms, folder="Results/Benchmarks", bar=None):
    """Run all the algorithms and saves them in a folder"""
    for algo, name in algorithms:
        save(run(benchmark, algorithm=algo, bar=bar), folder+"/save_"+name)


def load(filename):
    """Loads the results from a file"""
    with lzma.open(filename+".lzma", 'rb') as file:
        content = pickle.load(file)
    return content


def load_algorithms(folder="Results/Benchmarks", name="save"):
    """Loads the results from all the algorithms"""
    return {n:load(folder+"/"+name+"_"+n) for n in ["PC", "PC_stable", "PC_ccs_orientation", "PC_ccs_skeleton"]}


def run_benchmark(name="50_nodes_80_arcs", folder="Results/Benchmarks/50_nodes_80_arcs", nb_networks=100, nb_nodes=50, average_degree=1.6, nb_modmax=4,initialize=True, nb_samples=[100, 500, 1000], algorithms=[(PC, "PC"), (PC_stable, "PC_stable"), (PC_ccs_orientation, "PC_ccs_orientation"), (PC_ccs_skeleton, "PC_ccs_skeleton")]):
    """Runs the default benchmark on all the algorithms"""

    try:
        os.mkdir(folder+"/")
    except FileExistsError:
        pass

    try:
        os.mkdir(folder+"/results/")
    except FileExistsError:
        pass

    benchmark = Benchmark(name, folder=folder, nb_samples=nb_samples, nb_networks=nb_networks, nb_nodes=nb_nodes, average_degree=average_degree, initialize=initialize)

    if initialize is False: benchmark.load_bns()
    
    for samples in nb_samples:
        save_folder = folder+"/results/"+str(samples)

        try:
            os.mkdir(save_folder)
        except FileExistsError:
            pass
        
        bar_size = round((benchmark.alpha_max-benchmark.alpha_min)/benchmark.alpha_step)
        bar = Bar("Running on the alpha values for {} samples".format(str(samples)), max=len(algorithms)*bar_size)

        benchmark.load_samples(samples_folder="sampled_bns/"+str(samples))
        run_algorithms(benchmark, folder=save_folder, bar=bar, algorithms=algorithms)

def plot_precision_recall(ax, alpha_results, errors=True, show_label=True, digits_label=2, colors=("orange", "green", "blue", "red"), title="Precision-recall for all algorithms"):
    """
    Plots the precision recall curves depending on alpha
    """

    ax.set_title(title)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)

    colors = iter(colors)

    for algo, alpha_result in alpha_results.items():
        color = next(colors)

        recalls, precisions = [], []
        for i, (alpha, result) in enumerate(alpha_result.items()):
            recall, precision = np.mean(result["recall"]), np.mean(result["precision"])

            if errors is True:
                ax.errorbar(recall, precision, xerr=np.std(result["recall"]), yerr=np.std(result["precision"]), color=color)
            
            if show_label is True and (i == 0 or i == len(alpha_result)-1):
                ax.annotate(str(round(alpha, digits_label)), (recall, precision), color=color)
            
            recalls.append(recall)
            precisions.append(precision)

        line, = ax.plot(recalls, precisions, color=color)
        line.set_label(algo)
        ax.legend()

def plot_bar_time_algos(ax, alpha_results, errors=True, show_mean=True, colors=("orange", "green", "blue", "red"), title="Computing time for all algorithms depending on alpha"):
    """
    Plots the bars of the average computing time for the benchmark for each algorithm and each alpha
    """
    ax.set_title(title)

    alphas = list(list(alpha_results.values())[0].keys())
    min_alpha, max_alpha = min(alphas), max(alphas)
    ax.set_xlabel("Algorithms for alpha between {:.1e} and {:.1e}".format(min_alpha, max_alpha))
    ax.set_ylabel("Computing time in ms")

    ax.xaxis.grid(True)
    ax.yaxis.grid(True)

    colors = iter(colors)
    mean_means, max_time = [], 0
    for j, (algo, alpha_result) in enumerate(alpha_results.items()):
        color = next(colors)

        means = []
        for i, (alpha, result) in enumerate(alpha_result.items()):
            width = len(alpha_result)

            means.append(np.mean(result["time"]))
            ax.bar(j*width+i, means[-1], yerr=np.std(result["time"]) if errors is True else 0, align="center", ecolor="black", color=color)
            max_time = max((max_time, means[-1]))
        else:        
            if show_mean is True:
                mean_means.append(np.mean(means))
                ax.bar((j+.5)*width-.5, mean_means[-1], align="center", alpha=.2, color=color, width=width)

    
    ax.set_yticks(mean_means + list(range(0, round(max_time) + 50, 25)))

    ax.set_xticks([(j+.5) * width for j in range(len(alpha_results))])
    ax.set_xticklabels(alpha_results.keys())

if __name__ == '__main__':
    nb_samples = [100, 500, 1000]
    algorithms = [(PC, "PC"), (PC_stable, "PC_stable"), (PC_ccs_orientation, "PC_ccs_orientation"), (PC_ccs_skeleton, "PC_ccs_skeleton")]
    
    # Default benchmark, needlessly slow and crashes with skeleton
    # run_benchmark(initialize=False, nb_samples=nb_samples, algorithms=algorithms)

    # Medium interactions
    # run_benchmark(initialize=False, name="25_nodes_40_arcs", folder="Results/Benchmarks/25_nodes_40_arcs", nb_nodes=25, average_degree=1.6, nb_modmax=4, algorithms=algorithms, nb_samples=nb_samples)

    # Strong interactions
    # run_benchmark(initialize=False, name="25_nodes_80_arcs", folder="Results/Benchmarks/25_nodes_80_arcs", nb_nodes=25, average_degree=3.2, nb_modmax=4, algorithms=algorithms, nb_samples=nb_samples)

    colors = iter(("orange", "green", "blue", "red"))
    alpha_results = load_algorithms(folder="Results/Benchmarks/25_nodes_40_arcs/results/1000")

    fig, axs = plt.subplots(1, 2)
    plot_precision_recall(axs[0], alpha_results, errors=False, colors=colors)
    plot_bar_time_algos(axs[1], alpha_results)
    plt.show()