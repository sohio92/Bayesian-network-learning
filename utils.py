import os
import pydot
import pydotplus
import pyAgrum as gum
import pyAgrum.lib.image as gimg

def save_graph(graph, filename, folder):
    if graph is None:   return

    if type(graph) == pydotplus.graphviz.Dot:
        graph.write(folder + filename, format="png")
    else:
        pydot.graph_from_dot_data(graph.toDot())[0].write_png(folder + filename)

def save_result(filename, save=True, folder="Results/"):
    def decorator(function):
        def inner(*args, **kwargs):
            result = function(*args, **kwargs)
            if save is True : save_graph(result['graph'], filename, folder)
            return result
        return inner
    return decorator

@save_result("generated_bn.png")
def generate_bn_and_csv(n_nodes=10, n_arcs=12, n_modmax=4, n_data=1000, folder="Results/", name="sampled_bn.csv"):
    generator = gum.BNGenerator()
    bn = generator.generate(n_nodes=n_nodes, n_arcs=n_arcs, n_modmax=n_modmax)

    gum.generateCSV(bn,name_out=folder+name, n=n_data, show_progress=False, with_labels=False)
    return {"graph":bn, "learner":gum.BNLearner(folder + name)}

def is_independant(learner, x, y, z=[]):
    return learner.chi2("n_"+str(x), "n_"+str(y), ["n_"+str(i) for i in z])[1] > .05

def edge_to_arc(graph, x, y):
    graph.eraseEdge(x, y)
    graph.addArc(x, y)

def graph_to_bn(graph):
    new_bn = gum.BayesNet()
    
    node_to_var = {}
    for x in graph.nodes():
        new_var = new_bn.add(gum.LabelizedVariable("n_"+str(x), "is true?", 2))
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
    
    compteur = 0
    for _ in range(max_tries):
        bn, learner = generate_bn_and_csv(n_nodes=n_nodes, n_arcs=n_arcs, n_modmax=n_modmax, n_data=n_data, folder=folder).values()

        try:
            algorithm.learn(bn, learner, verbose=False)
        except RuntimeError:
            compteur += 1
        
        algorithm.reset()
        if verbose is True: print("Progress: {}/{}\t".format(_, max_tries), end="\r")
    
    os.remove(folder + "sampled_bn.csv")

    try:
        os.rmdir(folder)
    except OSError:
        pass

    return compteur / max_tries