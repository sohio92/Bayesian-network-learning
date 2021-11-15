import matplotlib.pyplot as plt

from progress.bar import Bar

from pc import PC
from pc_stable import PC_stable
from pc_ccs_orientation import PC_ccs_orientation
from pc_ccs_skeleton import PC_ccs_skeleton

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


def benchmark_50_nodes_80_arcs(initialize=True, nb_samples=[100, 500, 1000], algorithms=[(PC, "PC"), (PC_stable, "PC_stable"), (PC_ccs_orientation, "PC_ccs_orientation"), (PC_ccs_skeleton, "PC_ccs_skeleton")]):
    """Runs the default benchmark on all the algorithms"""

    benchmark_50_nodes_80_arcs = Benchmark("50_nodes_80_arcs", folder="Results/Benchmarks/50_nodes_80_arcs", nb_samples=nb_samples, nb_networks=100, nb_nodes=50, initialize=initialize)

    if initialize is False: benchmark_50_nodes_80_arcs.load_bns()
    
    for samples in nb_samples:
        save_folder = "Results/Benchmarks/50_nodes_80_arcs/results/"+str(samples)

        try:
            os.mkdir(save_folder)
        except FileExistsError:
            pass
        
        bar_size = round((benchmark_50_nodes_80_arcs.alpha_max-benchmark_50_nodes_80_arcs.alpha_min)/benchmark_50_nodes_80_arcs.alpha_step)
        bar = Bar("Running on the alpha values for {} samples".format(str(samples)), max=len(algorithms)*bar_size)

        benchmark_50_nodes_80_arcs.load_samples(samples_folder="sampled_bns/"+str(samples))
        run_algorithms(benchmark_50_nodes_80_arcs, folder=save_folder, bar=bar, algorithms=algorithms)


if __name__ == '__main__':
    nb_samples = [100, 500, 1000]
    algorithms = [(PC, "PC"), (PC_stable, "PC_stable"), (PC_ccs_orientation, "PC_ccs_orientation")] # No (PC_ccs_skeleton, "PC_ccs_skeleton") because creates crash -> swigpy memory leak !!

    benchmark_50_nodes_80_arcs(initialize=False, nb_samples=nb_samples, algorithms=algorithms)
    # load_algorithms(folder="Results/Benchmarks/50_nodes_80_arcs/results/500",)
    