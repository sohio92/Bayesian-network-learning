import pydot
import pyAgrum as gum
import pyAgrum.lib.image as gimg

def save_result(filename, save=True, folder="Results/"):
    def decorator(function):
        def inner(*args, **kwargs):
            result = function(*args, **kwargs)
            if save is True : pydot.graph_from_dot_data(result['graph'].toDot())[0].write_png(folder + filename)
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