import json
import itertools


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
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def consistent_set(self, X, Y, adj_X_excl_Y, G):
		consistent_sep_Z = {}
		for Z in adj_X_excl_Y:
			if G.hasUndirectedPath(X, Z) and G.hasUndirectedPath(Z, Y):
				consistent_sep_Z.add(Z)
		return consistent_sep_Z

	def _learn_consistent_skeleton(self, G_1, G_2, learner):
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

		SeptSet_xy = {} # Z for every pair X, Y
		X_Y_pairs = list(combinations(self.graph.nodes(), 2))
		for X,Y in X_Y_pairs:
				SeptSet_xy[(X,Y)] = []

		while has_more_neighbours(self.graph, d):
			for X,Y in self.graph.edges():
				adj_X_excl_Y = self.graph.neighbours(X).copy()
				adj_X_excl_Y.remove(Y)
				adj_X_excl_Y_inter_consist_Z = adj_X_excl_Y.intersection(self.consistent_set( X, Y, adj_X_excl_Y, G))

				if len(adj_X_excl_Y_inter_consist_Z) >= d:
					# Get all the d-sets of the neighbours of X
					for Z in list(combinations(adj_X_excl_Y_inter_consist_Z, d)):
						# Independance test, knowing the neighbours
						if is_independant(learner, X, Y, Z):
							self.graph.eraseEdge(X,Y)

							SeptSet_xy[tuple(sorted((X,Y)))].append(Z)
							break

						if X in self.graph.neighbours(Y):
							break

			d += 1


		return {"graph":self.graph, "SeptSet_xy":SeptSet_xy}

	@save_result(save_prefix + "_final.png", save=save_final)
	def learn(self, bn, learner):
		print("Initializing the graph..")
		self._init_graph(bn)

		print("Learning the skeleton..")
		SeptSet_xy = self._learn_skeleton(learner)["SeptSet_xy"]

		print("Orienting the graph's edges..")
		self._orient_edges(SeptSet_xy)

		print("Wrapping up the orientations..")
		self._wrap_up_learning()

		try:
			self.learned_bn = graph_to_bn(self.graph)
		except:
			print("Learning failed, learned BN contains cycles.")

		return {"graph":self.graph}





if __name__ == "__main__":
	bn, learner = generate_bn_and_csv(folder=save_folder).values()
	pc_css = PC_ccs()

	# pc_stable.learn(bn, learner)
	# _, hamming, skeleton_scores = pc_stable.compare_learned_to_bn(bn).values()
	#
	# print("Hamming: {}\nSkeleton scores: {}\n".format(hamming, skeleton_scores))
	# print("\nProportion of failed learnings: {}%".format(round(test_robustness(PC_stable) * 100, 3)))
