import pyAgrum as gum
import pyAgrum.lib.image as gimg
import pyAgrum.lib.bn_vs_bn as bvb
import itertools
import json

from utils import *

with open("parameters.json", 'r') as file:
    parameters = json.load(file)["parameters"]

save_folder = parameters["save_folder"]
save_prefix = parameters["save_prefix"]
save_steps = parameters["save_steps"]
save_final = parameters["save_final"]
save_compare = parameters["save_compare"]

class PC():
    """
    PC algorithm to learn a Bayesian Network
    """
    def __init__(self):
        self.graph = gum.MixedGraph()
        self.learned_bn = None
    
    @save_result(save_prefix + "_0.png", save=save_steps, folder=save_folder)
    def _init_graph(self, bn):
        self.graph.clear()
        for n in bn.nodes():
            self.graph.addNodeWithId(n)

        for x in self.graph.nodes():
            for y in self.graph.nodes():
                if x != y:  self.graph.addEdge(x,y)
        
        return {"graph":self.graph}
    
    @save_result(save_prefix + "_1.png", save=save_steps, folder=save_folder)
    def _learn_skeleton(self, learner):
        """
        Phase 1: learn the skeleton
        """
        def has_more_neighbours(graph, d):
            """
            Loop condition
            """
            for x in graph.nodes():
                if len(graph.neighbours(x)) > d:    return True
            return False

        d = 0
        SeptSet_xy = {tuple(sorted(edge)):[] for edge in self.graph.edges()}  # Z for every pair X, Y (sorted)
        while has_more_neighbours(self.graph, d):
            for X, Y in self.graph.edges():
                adj_X_excl_Y = self.graph.neighbours(X).copy()
                adj_X_excl_Y.remove(Y)

                if len(adj_X_excl_Y) >= d:
                    # Get all the d-sets of the neighbours of x
                    for Z in itertools.combinations(adj_X_excl_Y, d):
                        # Independance test, knowing the neighbours
                        if is_independant(learner, X, Y, Z):
                            self.graph.eraseEdge(X, Y)

                            SeptSet_xy[tuple(sorted((X, Y)))] += Z
                            
                            # break
                            
            d += 1

        return {"graph":self.graph, "SeptSet_xy":SeptSet_xy}

    @save_result(save_prefix + "_2.png", save=save_steps, folder=save_folder)
    def _orient_edges(self, SeptSet_xy):
        """
        Phase 2: orient the skeleton's edges
        """
        # V-structures
        for z in self.graph.nodes():
            neigh = list(self.graph.neighbours(z))
            for i in range(len(neigh)):
                for y in neigh[i+1:]:
                    x = neigh[i]
                    if self.graph.existsEdge(x, y) is False:
                        # Unshielded triple
                        if z not in SeptSet_xy[tuple(sorted((x, y)))]:
                            edge_to_arc(self.graph, x, z)
                            edge_to_arc(self.graph, y, z)

        # Propagation
        was_oriented = True # Until no edge can be oriented
        while was_oriented is True:
            was_oriented = False
            for x in self.graph.nodes():
                for y in self.graph.nodes():
                    if self.graph.existsEdge(x, y) is False and self.graph.existsArc(x, y) is True:
                        # No v-structure added
                        for z in self.graph.neighbours(y):
                            if self.graph.existsArc(x, z) and self.graph.existsEdge(z, y) is True:
                                edge_to_arc(self.graph, z, y)
                                was_oriented = True
                            
                    elif self.graph.existsEdge(x, y) is True and self.graph.hasDirectedPath(x, y):
                        # No cycle
                        edge_to_arc(self.graph, x, y)
                        was_oriented = True

        return {"graph":self.graph}

    @save_result(save_prefix + "_3.png", save=save_steps, folder=save_folder)
    def _wrap_up_learning(self):
        """
        Phase 3: fill the other orientations without adding any v-structure
        """
        for x, y in self.graph.edges():
            self.graph.eraseEdge(x, y)
            self.graph.addArc(y, x)

            for z in self.graph.neighbours(y):
                if self.graph.existsArc(z, y):    break
            else:
                self.graph.addArc(x, y)
                self.graph.eraseArc(y, x)

        return {"graph":self.graph}
    
    @save_result(save_prefix + "_final.png", save=save_final, folder=save_folder)
    def learn(self, bn, learner, verbose=True):
        if verbose is True: print("Initializing the graph..", end='\r')
        self._init_graph(bn)

        if verbose is True: print("Learning the skeleton..", end='\r')
        SeptSet_xy = self._learn_skeleton(learner)["SeptSet_xy"]

        if verbose is True: print("Orienting the graph's edges..", end='\r')
        self._orient_edges(SeptSet_xy)

        if verbose is True: print("Wrapping up the orientations..", end='\r')
        self._wrap_up_learning()
        
        try:
            self.learned_bn = graph_to_bn(self.graph)
        except:
            raise RuntimeError("Learning failed, learned BN contains cycles.")

        return {"graph":self.graph}

    @save_result("comparated_bn.png", save=save_compare, folder=save_folder)
    def compare_learned_to_bn(self, bn):
        if self.learned_bn is None: return {"graph":None, "hamming":None, "skeletonScores":None}

        comparator = bvb.GraphicalBNComparator(bn, self.learned_bn)

        return {"graph":comparator.dotDiff(), "hamming":comparator.hamming(), "skeletonScores":comparator.skeletonScores()}
    
    def reset(self):
        self.__init__()

if __name__ == "__main__":
    bn, learner = generate_bn_and_csv(folder=save_folder).values()
    pc = PC()

    pc.learn(bn, learner)
    _, hamming, skeleton_scores = pc.compare_learned_to_bn(bn).values()

    print("Hamming: {}\nSkeleton scores: {}\n".format(hamming, skeleton_scores))
    print("\nProportion of failed learnings: {}%".format(round(test_robustness(PC) * 100, 3)))