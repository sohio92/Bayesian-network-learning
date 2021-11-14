import matplotlib.pyplot as plt

from pc import PC
from pc_stable import PC_stable
from pc_css_orientation import PC_ccs_orientation
from pc_css_skeleton import PC_ccs_skeleton

from benchmark import Benchmark

from utils import *

def run(benchmark, algorithm=PC):
    return unpack_results(benchmark.run_test(algorithm=algorithm))

def save(content, filename):
    with lzma.open(filename+".lzma", 'wb') as file:
        pickle.dump(content, file)

def load(filename):
    with lzma.open(filename+".lzma", 'rb') as file:
        content = pickle.load(file)
    return content

def run_algorithms(benchmark, folder="Results/Benchmark/"):
    save(run(benchmark, algorithm=PC), folder+"save_test_PC")
    save(run(benchmark, algorithm=PC_stable), folder+"save_test_PC_stable")
    save(run(benchmark, algorithm=PC_ccs_orientation), folder+"save_test_PC_ccs_orientation")
    save(run(benchmark, algorithm=PC_ccs_skeleton), folder+"save_test_PC_ccs_skeleton")

def load_algorithms(folder="Results/Benchmark/"):
    return [load(folder+"save_test_"+n) for n in ["PC", "PC_stable", "PC_ccs_orientation", "PC_ccs_skeleton"]]


# a = Benchmark("test", intialize=False)
# a.load_bns()
# run_algorithms(a)

results_PC, results_PC_stable, results_PC_ccs_orientation, results_PC_ccs_skeleton, = load_algorithms()

plt.scatter(results_PC_ccs_orientation['recall'], results_PC_ccs_orientation['precision'], color="blue")
plt.show()
plt.scatter(results_PC_ccs_skeleton['recall'], results_PC_ccs_skeleton['precision'], color="red")
plt.show()