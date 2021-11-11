import time
import lzma
import pickle

from pc import PC
from pc_stable import PC_stable
from pc_css_orientation import PC_ccs_orientation
from pc_css_skeleton import PC_ccs_skeleton

from utils import *

class Benchmark:
    """
    Benchmark to run tests on the implemented algorithms
    """

    def __init__(self, name="Benchmark", folder="Results/Benchmark", verbose=True, nb_networks=100, nb_samples=500, nb_nodes=10, nb_modmax=4, average_degree=1.6, alpha_range=(10e-5, 0.2), alpha_step=10e-2, alpha_default=0.05, intialize=True):
        self.name, self.folder = name, folder

        self.nb_networks, self.nb_samples = nb_networks, nb_samples
        self.nb_nodes, self.nb_modmax, self.nb_arcs = nb_nodes, nb_modmax, round(
            nb_nodes * average_degree)
        self.alpha_min, self.alpha_max = alpha_range
        self.alpha_step = alpha_step
        self.alpha_default = alpha_default

        self.bns, self.leas = [], []

        if verbose is True: print(self)

        if intialize is True:
            self._init_bn(verbose)
            self.save_bns()
            self._sample_bns(verbose)
        
    def __str__(self):
        s = "Benchmark {}".format(self.name)
        s += "\nNetworks: {}\tSamples: {}".format(self.nb_networks, self.nb_samples)
        s += "\n\tNodes: {}\tArcs: {}\tValues: {}".format(self.nb_nodes, self.nb_arcs, self.nb_modmax)
        return s 

    def _init_bn(self, verbose=True):
        """
        Generates all the desired BNs
        """
        if verbose is True: print("Generating BNs..")

        generator = gum.BNGenerator()
        self.bns = [generator.generate(n_nodes=self.nb_nodes, n_arcs=self.nb_arcs, n_modmax=self.nb_modmax)
                    for _ in range(self.nb_networks)]

    def _sample_bns(self, verbose=True):
        """
        Samples all bns
        """
        if verbose is True: print("Sampling BNs..")
        
        for i in range(len(self.bns)):
            name = self.folder + "/sampled_bns/sampled_bn_" + str(i) + ".csv"
            gum.generateCSV(self.bns[i], name_out=name, n=self.nb_samples, show_progress=False, with_labels=False)
            self.leas.append(gum.BNLearner(name))
    
    def save_bns(self, folder=None, filename=None, verbose=True):
        """
        Saves the generated BNs for later use
        """
        if folder is None:  folder = self.folder
        if filename is None:    filename = "saved_bns_" + self.name

        if verbose is True: print("Saving generated BNs..")
        with lzma.open(folder+"/"+filename+".lzma", 'wb') as file:
            pickle.dump([bn.toDot() for bn in self.bns], file)
    
    def load_bns(self, folder=None, filename=None, verbose=True):
        """
        Loads already generated BNs, computes their learners from saved samples
        """
        if folder is None:  folder = self.folder
        if filename is None:    filename = "saved_bns_" + self.name

        if verbose is True: print("Loading generated BNs..")
        with lzma.open(folder+"/"+filename+".lzma", "rb") as file:
            self.bns = [graph_to_bn(pydot.graph_from_dot_data(bn)[0]) for bn in pickle.load(file)]

        self.nb_networks = len(self.bns)
        self.nb_nodes = self.bns[0].size()
        self.nb_arcs = self.bns[0].sizeArcs()
        # self.nb_modmax
        self.leas = [gum.BNLearner(folder+"/sampled_bns/sampled_bn_" + str(i) + ".csv") for i in range(len(self.bns))]

    def run_test(self, alpha=None, algorithm=PC):
        """
        Runs a performance test on a given algorithm
        """
        if alpha is None:   alpha = self.alpha_default
        child = algorithm(alpha=alpha)

        times, scores = [], {"Hamming":[], "Skeleton":[]}
        for i in range(self.nb_networks):
            child.reset({"alpha":alpha})

            times.append(-time.time())

            try:
                child.learn(self.bns[i], self.leas[i], verbose=False, save_final=False).values()
                times[-1] += time.time()
            except RuntimeError:
                times.pop()
                continue

            _, hamming, skeleton = child.compare_learned_to_bn(self.bns[i], save_comparison=False).values()
            scores["Hamming"].append(hamming)
            scores["Skeleton"].append(skeleton)

        return {"times":times, "scores":scores}
    
    def run_alpha_test(self, algorithm=PC):
        """
        Runs a performance test for all values of alpha
        """
        results = {}
        a = self.alpha_min
        while a <= self.alpha_max:
            results[a] = self.run_test(a, algorithm=algorithm)
            a += self.alpha_step
        
        return results

# benchmark_insurance = Benchmark("Insurance benchmark", nb_nodes=27, average_degree=52/27, nb_modmax=984)
# benchmark_hepar2 = Benchmark("Hepar2 benchmark", nb_nodes=70, average_degree=123/70, nb_modmax=1453)
# benchmark_barley = Benchmark("Barley benchmark", nb_nodes=48, average_degree=84/48, nb_modmax=114005)

if __name__ == "__main__":
    a = Benchmark("test", nb_networks=5)
    del a
    a = Benchmark("test", intialize=False)
    a.load_bns()

    times, scores = a.run_test().values()
    print(sum(times)/len(times))
