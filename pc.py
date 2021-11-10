import json
import pyAgrum as gum
import pyAgrum.lib.image as gimg
import pyAgrum.lib.bn_vs_bn as bvb

import itertools
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
	def __init__(self, alpha=.05):
		self.graph = gum.MixedGraph()
		self.learned_bn = None

		self.alpha = alpha

	def _init_graph(self, bn):
		"""
		Start the algorithm with a complete graph
		"""
		self.graph.clear()
		self.graph = make_complete_graph(bn.nodes())
		return {"graph":self.graph}

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
		Sepset_xy = {tuple(sorted(xy)):set() for xy in itertools.combinations(self.graph.nodes(), 2)} # Z for every pair X, Y
		while has_more_neighbours(self.graph, d):
			for X,Y in self.graph.edges():
				adj_X_excl_Y = [neigh for neigh in self.graph.neighbours(X) if neigh != Y]

				if len(adj_X_excl_Y) >= d:
					# Get all the d-sets of the neighbours of X
					for Z in itertools.combinations(adj_X_excl_Y, d):
						# Independance test, knowing the neighbours
						if is_independant(learner, X, Y, Z, alpha=self.alpha):
							self.graph.eraseEdge(X,Y)

							Sepset_xy[tuple(sorted((X,Y)))].add(Z)
							break
			d += 1

		return {"graph":self.graph, "Sepset_xy":Sepset_xy}

	def _orient_edges(self, Sepset_xy, graph=None):
		"""
		Phase 2: orient the skeleton's edges
		"""
		if graph is None:	graph = self.graph
		
		# V-structures

		# Simple way to find unshielded triples:
		# for any node Z, all the triples X, Z, Y such that X and Y are neighbours
		# of Z, are unshielded triples.
		for Z in graph.nodes():
			for X, Y in itertools.combinations(graph.neighbours(Z), 2):
				if Z not in Sepset_xy[tuple(sorted((X, Y)))]:
					#R1 rule
					edge_to_arc(graph, X, Z)
					edge_to_arc(graph, Y, Z)


		was_oriented = True # Until no edge can be oriented
		while was_oriented is True:
			was_oriented = False

			for X,Y in Sepset_xy.keys(): # For every X, Y pairs of nodes
				if not graph.existsEdge(X,Y): # If X and Y are not neighbours

					shared_neighbours = graph.neighbours(X).intersection(graph.neighbours(Y))

					# For all unshielded triples and non fully-oriented v-structures
					for Z in shared_neighbours:
						# R2 rule
						if graph.existsArc(X,Z) and graph.existsEdge(Z,Y):
							edge_to_arc(graph, Z, Y)
							was_oriented = True

				elif graph.existsEdge(X,Y) and graph.hasDirectedPath(X, Y):
					# If there is a cycle going trhough X and Y

					#R3 rule
					edge_to_arc(graph, X, Y)
					was_oriented = True

		return {"graph":graph}

	def _wrap_up_learning(self, graph=None):
		"""
		Phase 3: fill the other orientations without adding any v-structure
		"""
		if graph is None:	graph = self.graph

		for x, y in graph.edges():
			graph.eraseEdge(x, y)
			graph.addArc(y, x)

			for z in graph.neighbours(y):
				if graph.existsArc(z, y):    break
			else:
				graph.addArc(x, y)
				graph.eraseArc(y, x)

		return {"graph":graph}

	def learn(self, bn, learner, verbose=True):

		if verbose is True: print("Initializing the graph..", end='\r')
		self._init_graph(bn)

		if verbose is True: print("Learning the skeleton..", end='\r')
		Sepset_xy = self._learn_skeleton(learner)["Sepset_xy"]

		if verbose is True: print("Orienting the graph's edges..", end='\r')
		self._orient_edges(Sepset_xy)

		if verbose is True: print("Wrapping up the orientations..", end='\r')
		self._wrap_up_learning()

		try:
			self.learned_bn = graph_to_bn(self.graph)
		except:
			raise RuntimeError("Learning failed, learned BN contains cycles.")

		return {"graph":self.graph}

	def compare_learned_to_bn(self, bn):
		if self.learned_bn is None: return

		comparator = bvb.GraphicalBNComparator(bn, self.learned_bn)

		return {"graph":comparator.dotDiff(), "hamming":comparator.hamming(), "skeletonScores":comparator.skeletonScores()}

	def reset(self):
		self.__init__()

if __name__ == "__main__":
	bn, learner = generate_bn_and_csv(n_data=10000).values()

	pc = PC()
	pc.learn(bn, learner)

	_, hamming, skeletonScores = pc.compare_learned_to_bn(bn).values()
	print("Hamming: {}\nSkeleton scores: {}".format(hamming, skeletonScores))

	print("Robustesse : {}%".format(test_robustness(PC, max_tries=1000) * 100))
