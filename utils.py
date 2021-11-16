import itertools
import pickle
import os
import lzma

import numpy as np

import pydot
import pydotplus
import pyAgrum as gum
import pyAgrum.lib.image as gimg


def save_graph(graph, filename, folder):
    if graph is None:
        return

    if type(graph) == pydotplus.graphviz.Dot:
        graph.write(folder + filename, format="png")
    else:
        pydot.graph_from_dot_data(graph.toDot())[
            0].write_png(folder + filename)


def save_result(filename, save=True, folder="Results/"):
    def decorator(function):
        def inner(*args, **kwargs):
            result = function(*args, **kwargs)
            if save is True:
                save_graph(result['graph'], filename, folder)
            return result
        return inner
    return decorator


def generate_bn_and_csv(n_nodes=10, n_arcs=12, n_modmax=4, n_data=1000, save_generated=True, folder="Results/", name="sampled_bn.csv"):
    generator = gum.BNGenerator()
    bn = generator.generate(n_nodes=n_nodes, n_arcs=n_arcs, n_modmax=n_modmax)

    gum.generateCSV(bn, name_out=folder+name, n=n_data,
                    show_progress=False, with_labels=False)
    if save_generated is True:
        save_graph(bn, "generated_bn.png", folder)

    return {"graph": bn, "learner": gum.BNLearner(folder + name)}


def is_independant(learner, x, y, z=[], alpha=.05):
    return learner.chi2("n_"+str(x), "n_"+str(y), ["n_"+str(i) for i in z])[1] > alpha


def make_complete_graph(nodes):
    """
    Make complete graph from a given array of nodes
    """
    graph = gum.MixedGraph()

    for n in nodes:
        graph.addNodeWithId(n)

    for x in graph.nodes():
        for y in graph.nodes():
            if x != y:
                graph.addEdge(x, y)

    return graph


def edge_to_arc(graph, x, y, replace_conflicts=False):
    """
    Replace an edge with an arc, replaces conflicting orientations if desired
    """
    if replace_conflicts is True:
        graph.eraseArc(y, x)

    graph.eraseEdge(x, y)
    graph.addArc(x, y)


def copy_mixed_graph(graph):
    new_graph = gum.MixedGraph()

    for node in graph.nodes():
        new_graph.addNodeWithId(node)

    for edge in graph.edges():
        new_graph.addEdge(*edge)

    for arc in graph.arcs():
        new_graph.addArc(*arc)

    return new_graph


def get_missing_edges(graph):
    return [edge for edge in make_complete_graph(graph.nodes()).edges() if edge not in graph.edges()]


def consistent_set(graph, X, Y):
    """
    Returns the consistent set given X and Y, the nodes such that there exists a path from X to Y passing through Z
    """
    other_graph = copy_mixed_graph(graph) # CAN CAUSE CRASH, NEEDS TO BE REPLACED SO IT DOESNT CREATE GRAPHS
    other_graph.eraseNode(X)  # We don't want to go through X twice

    consistent_set = set()
    for Z in [neigh for neigh in graph.adjacents(X) if neigh != Y]:
        if len(other_graph.mixedUnorientedPath(Z, Y)) != 0:
            consistent_set.add(Z)
    del other_graph
    return consistent_set


def graph_to_bn(graph, nb_values=2):
    new_bn = gum.BayesNet()

    node_to_var = {}
    for x in graph.nodes():
        new_var = new_bn.add(gum.LabelizedVariable(
            "n_"+str(x), "value?", nb_values))
        node_to_var[x] = new_var

    for x, y in graph.arcs():
        new_bn.addArc(node_to_var[x], node_to_var[y])

    return new_bn


def test_robustness(algorithm, max_tries=100, n_nodes=10, n_arcs=12, n_modmax=4, n_data=1000, folder="Results/temp/", verbose=True):
    """
    Returns the proportion of failed learnings for a given algorithm.
    """
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass

    algo_obj = algorithm()

    compteur = 0
    for _ in range(max_tries):
        bn, learner = generate_bn_and_csv(
            n_nodes=n_nodes, n_arcs=n_arcs, n_modmax=n_modmax, n_data=n_data, save_generated=False, folder=folder).values()

        try:
            algo_obj.learn(bn, learner, verbose=False,
                           save_final=False, save_steps=False)
        except RuntimeError:
            compteur += 1

        algo_obj.reset()
        if verbose is True:
            print("Testing robustness: {}/{}\t".format(_ + 1, max_tries), end="\r")

    os.remove(folder + "sampled_bn.csv")

    try:
        os.rmdir(folder)
    except OSError:
        pass

    return compteur / max_tries


def mean_std(A, lamb=1):
    return lamb * np.mean(A), lamb * np.std(A)


def unpack_results(test_result):
    """
    Unpack a benchmark's results
    """

    time = [1000 * t for t in test_result['times']]

    hamming, structural_hamming = [], []
    for h in test_result['scores']['Hamming']:
        hamming.append(h['hamming'])
        structural_hamming.append(h['structural hamming'])

    precision, recall, fscore, dist2opt = [], [], [], []
    for s in test_result['scores']['Skeleton']:
        precision.append(s['precision'])
        recall.append(s['recall'])
        fscore.append(s['fscore'])
        dist2opt.append(s['dist2opt'])

    return {'time': time, 'hamming': hamming, 'structural_hamming': structural_hamming, 'precision': precision, 'recall': recall, 'fscore': fscore, 'dist2opt': dist2opt}


def save_results(results, path):
    with lzma.open(path+".lzma", 'wb') as file:
        pickle.dump(results, file)