import matplotlib.pyplot as plt
import os

from progress.bar import Bar
from pyparsing import col

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


def run(benchmark, algorithm=PC, bar=None, alpha_list=None):
    """Returns the results of one algorithm on the given benchmark"""
    alpha_results = {}
    for alpha, result in benchmark.run_alpha_test(algorithm=algorithm, bar=bar, alpha_list=alpha_list).items():
        alpha_results[alpha] = unpack_results(result)
    return alpha_results


def run_algorithms(benchmark, algorithms, folder="Results/Benchmarks", bar=None, alpha_list=None):
    """Run all the algorithms and saves them in a folder"""
    for algo, name in algorithms:
        save(run(benchmark, algorithm=algo, bar=bar, alpha_list=alpha_list), folder+"/save_"+name)


def load(filename):
    """Loads the results from a file"""
    with lzma.open(filename+".lzma", 'rb') as file:
        content = pickle.load(file)
    return content


def load_algorithms(folder="Results/Benchmarks", name="save", algorithms=["PC", "PC_stable", "PC_ccs_orientation", "PC_ccs_skeleton"]):
    """Loads the results from all the algorithms"""
    return {n:load(folder+"/"+name+"_"+n) for n in algorithms}


def run_benchmark(name="50_nodes_80_arcs", folder="Results/Benchmarks/50_nodes_80_arcs", nb_networks=100, nb_nodes=50, average_degree=1.6, nb_modmax=4,initialize=True, nb_samples=[100, 500, 1000], algorithms=[(PC, "PC"), (PC_stable, "PC_stable"), (PC_ccs_orientation, "PC_ccs_orientation"), (PC_ccs_skeleton, "PC_ccs_skeleton")], alpha_list=None):
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
        
        bar_size = round((benchmark.alpha_max-benchmark.alpha_min)/benchmark.alpha_step) if alpha_list is None else len(alpha_list)
        bar = Bar("Running on the alpha values for {} samples".format(str(samples)), max=len(algorithms)*bar_size)

        benchmark.load_samples(samples_folder="sampled_bns/"+str(samples))
        run_algorithms(benchmark, folder=save_folder, bar=bar, algorithms=algorithms, alpha_list=alpha_list)

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
                ax.errorbar(recall, precision, xerr=np.std(result["recall"]), yerr=np.std(result["precision"]), color=color, linewidth=.5, capsize=5)

            if show_label is True and (i == 0 or i == len(alpha_result)-1):
                ax.annotate("{:.1e}".format(alpha), (recall, precision), color=color)
            
            recalls.append(recall)
            precisions.append(precision)

        line, = ax.plot(recalls, precisions, color=color)
        line.set_label(algo)
        ax.legend(loc="lower left")

def plot_bar_time_algos(ax, alpha_results, errors=True, show_mean=True, colors=("orange", "green", "blue", "red"), title="Computing time for all algorithms depending on alpha", show_alphas=False):
    """
    Plots the bars of the average computing time for the benchmark for each algorithm and each alpha
    """
    ax.set_title(title)

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
            pbar = ax.bar(j*width+i, means[-1], yerr=np.std(result["time"]) if errors is True else 0, align="center", ecolor="black", color=color)
            max_time = max((max_time, means[-1]))
        else:
            pbar.set_label(algo)

            if show_mean is True:
                mean_means.append(np.mean(means))
                ax.bar((j+.5)*width-.5, mean_means[-1], align="center", alpha=.2, color=color, width=width)

    
    ax.legend(loc="upper left")
    ax.set_yticks(mean_means + list(range(0, round(max_time) + 50, 25)))

    alphas = list(list(alpha_results.values())[0].keys())
    min_alpha, max_alpha = min(alphas), max(alphas)
    x_label = "Algorithms for alpha between {:.1e} and {:.1e}".format(min_alpha, max_alpha)
    if show_alphas is False:    x_label += "\n" + str(alphas)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Computing time in ms")
    
    ax.set_xticks(np.concatenate([[i + j*width for i in range(len(alpha_result))] for j in range(len(alpha_results))]))
    ax.set_xticklabels(alphas * len(alpha_results) if show_alphas is True else ["" for _ in alphas] * len(alpha_results))

def compare_PC_PC_stable(ax, n_runs=10, alpha=0.05, title="Average fscore and variance for PC and PC_stable for alpha = {}", colors=("orange","green")):
    """
    Compares the average f_score and variance for PC and PC_stable at a given alpha.
    """
    bn, learner = generate_bn_and_csv(n_data=1000, n_nodes=25, n_arcs=40, save_generated=False).values()

    pc = PC(alpha=alpha)
    pc_stable = PC_stable(alpha=alpha)

    fscore_pc, fscore_stable = [], []
    for _ in range(n_runs):
        pc.learn(bn, learner, verbose=False, save_final=False)
        pc_stable.learn(bn, learner, verbose=False, save_final=False)

        fscore_pc.append(pc.compare_learned_to_bn(bn, save_comparison=False)["skeletonScores"]["fscore"])
        fscore_stable.append(pc_stable.compare_learned_to_bn(bn, save_comparison=False)["skeletonScores"]["fscore"])

    ax.yaxis.grid(True)
    ax.set_yticks(np.arange(0, 1+1, 0.05))

    ax.set_title(title.format(alpha))
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["PC", "PC_stable"])

    ax.set_ylabel("Average fscore")

    plt.bar(1, np.mean(fscore_pc), yerr=np.std(fscore_pc), color=colors[0], align="center", ecolor="black")
    plt.bar(2, np.mean(fscore_stable), yerr=np.std(fscore_stable), color=colors[1], align="center", ecolor="black")


if __name__ == '__main__':
    ## The nb of samples, the alphas, and the algorithms to use for the following benchmark
    nb_samples = [100, 500, 1000]
    alpha_list = [1e-25, 1e-20, 1e-17, 1.0e-15, 1.0e-13, 1.0e-10,
                  8.7e-09, 7.6e-07, 6.6e-05, 5.7e-03, 5.0e-02, 5.0e-01]
    algorithms = [(PC, "PC"), (PC_stable, "PC_stable"), (PC_ccs_orientation, "PC_ccs_orientation")]
    #[(PC, "PC"), (PC_stable, "PC_stable"), (PC_ccs_orientation, "PC_ccs_orientation"), (PC_ccs_skeleton, "PC_ccs_skeleton")]

    ## Run the benchmarks, if initialize is False they will seek previously generated Bayesian Networks in their paths (saves a lot of time)

    # # Weak interactions
    # run_benchmark(initialize=False, name="25_nodes_25_arcs", folder="Results/Benchmarks/25_nodes_25_arcs", nb_nodes=25, average_degree=1, nb_modmax=4, algorithms=algorithms, nb_samples=nb_samples, alpha_list=alpha_list)
    
    # # Medium interactions
    # run_benchmark(initialize=False, name="25_nodes_40_arcs", folder="Results/Benchmarks/25_nodes_40_arcs", nb_nodes=25, average_degree=1.6, nb_modmax=4, algorithms=algorithms, nb_samples=nb_samples, alpha_list=alpha_list)

    # # Strong interactions
    # run_benchmark(initialize=False, name="25_nodes_55_arcs", folder="Results/Benchmarks/25_nodes_55_arcs", nb_nodes=25, average_degree=55/25, nb_modmax=4, algorithms=algorithms, nb_samples=nb_samples, alpha_list=alpha_list)

    ## Plot the results

    colors = ("orange", "green", "red", "blue")  # The algorithms' colors
    list_paths = [["Results/Benchmarks/25_nodes_25_arcs/results/100", "Results/Benchmarks/25_nodes_25_arcs/results/500", "Results/Benchmarks/25_nodes_25_arcs/results/1000"], ["Results/Benchmarks/25_nodes_40_arcs/results/100", "Results/Benchmarks/25_nodes_40_arcs/results/500", "Results/Benchmarks/25_nodes_40_arcs/results/1000"], ["Results/Benchmarks/25_nodes_55_arcs/results/100", "Results/Benchmarks/25_nodes_55_arcs/results/500", "Results/Benchmarks/25_nodes_55_arcs/results/1000"]]

    for benchmarks_paths in list_paths:
        fig, axs = plt.subplots(len(benchmarks_paths), 2)
        for i, benchmark_path in enumerate(benchmarks_paths):
            alpha_results = load_algorithms(folder=benchmark_path, algorithms=[
                                            a[1] for a in algorithms])

            if len(benchmarks_paths) == 1:
                plot_precision_recall(
                    axs[0], alpha_results, errors=True, colors=colors, show_label=False)
                plot_bar_time_algos(axs[1], alpha_results, colors=colors)
            else:
                plot_precision_recall(
                    axs[i, 0], alpha_results, errors=True, colors=colors, show_label=False)
                plot_bar_time_algos(axs[i, 1], alpha_results, colors=colors)

        plt.show()
    