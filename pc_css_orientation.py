import json
from itertools import combinations


from pc_stable import PC_stable
from utils import *

with open("parameters.json", 'r') as file:
	parameters = json.load(file)["parameters"]

save_folder = parameters["save_folder"]
save_prefix = parameters["save_prefix"]
save_steps = parameters["save_steps"]
save_final = parameters["save_final"]
save_compare = parameters["save_compare"]

class PC_ccs(PC_stable):
	"""
	Sepset consistente PC algorithm (1st version, orientation consistency)
	"""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def _learn_skeleton(self, G_1, G_2, learner):
		"""
		Phase 1: learn the skeleton of G1|G2, NewStep1
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
		while has_more_neighbours(G_1, d):
			for X,Y in G_1.edges():
				adj_X_excl_Y = set([neigh for neigh in G_1.neighbours(X) if neigh != Y])
				adj_X_excl_Y_inter_consist_Z = adj_X_excl_Y.intersection(consistent_set(G_2, X, Y))

				if len(adj_X_excl_Y_inter_consist_Z) >= d:
					# Get all the d-sets of the neighbours of X
					for Z in itertools.combinations(adj_X_excl_Y_inter_consist_Z, d):
						# Independance test, knowing the neighbours
						if is_independant(learner, X, Y, Z, alpha=self.alpha):
							G_1.eraseEdge(X,Y)

							Sepset_xy[tuple(sorted((X,Y)))].add(Z)
							break

						if X in G_1.neighbours(Y):
							break
			d += 1

		return {"graph":G_1, "Sepset_xy":Sepset_xy}

	def _S(self, G1, G2, learner):
		"""
		Modified version of PC-stable, step1 of algorithm is replaced by NewStep1(G1|G2)
		"""
		_, Sepset_xy = self._learn_skeleton(G1, G2, learner).values()
		G_k = self._wrap_up_learning(self._orient_edges(Sepset_xy, graph=G1)["graph"])["graph"]

		return {"graph":G_k, "Sepset_xy":Sepset_xy}

	def _orientation_consistency(self, learner):
		"""
		Algorithm 3, consistent constraint-based algorithm through an iterative call of S
		"""
		k, G_k = 0, None
		G_0, Sepset_xy_0 = self._learn_skeleton(self.graph, gum.MixedGraph(), learner).values()
		list_G_k = [G_0]
		while any(other == G_k for other in list_G_k) is False:
			k += 1
			G_k, Sepset_xy_k = self._S(G_0, list_G_k[-1], learner).values()

			list_G_k.append(G_k)

		# Discarding the conflicting orientations
		for other in list_G_k[:-1]:
			if other == G_k:	break
			for arc in other.arcs():
				edge_to_arc(self.graph, *arc, replace_conflicts=True)

		return {"graph": self.graph, "Sepset_xy_0":Sepset_xy_0, "Sepset_xy_k":Sepset_xy_k}

	def learn(self, bn, learner, verbose=True):
		if verbose is True: print("Initializing the graph..", end='\r')
		self._init_graph(bn)

		if verbose is True: print("Learning..", end='\r')
		self._orientation_consistency(learner)

		if verbose is True: print("Wrapping up the orientations..", end='\r')
		self._wrap_up_learning()

		try:
			self.learned_bn = graph_to_bn(self.graph)
		except:
			raise RuntimeError("Learning failed, learned BN contains cycles.")

		return {"graph":self.graph}

if __name__ == "__main__":
	bn, learner = generate_bn_and_csv(folder=save_folder).values()
	pc_ccs = PC_ccs()

	pc_ccs.learn(bn, learner)
	_, hamming, skeleton_scores = pc_ccs.compare_learned_to_bn(bn).values()

	print("Hamming: {}\nSkeleton scores: {}\n".format(hamming, skeleton_scores))
	# print("\nProportion of failed learnings: {}%".format(round(test_robustness(PC_ccs) * 100, 3)))
