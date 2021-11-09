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
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def consistent_set(self, X, Y, adj_X_excl_Y, G):
		consistent_sep_Z = set()

		for Z in adj_X_excl_Y:

			if (G.mixedUnorientedPath(X, Y) != []) and (G.mixedUnorientedPath(Y, Z) != []):
				consistent_sep_Z.add(Z)

		return consistent_sep_Z

	def NewStep1(self, G_1, G_2, learner):
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
		X_Y_pairs = list(combinations(G_1.nodes(), 2))
		for X,Y in X_Y_pairs:
				SeptSet_xy[(X,Y)] = set()

		while has_more_neighbours(G_1, d):
			for X,Y in G_1.edges():
				adj_X_excl_Y = G_1.neighbours(X).copy()
				adj_X_excl_Y.remove(Y)
				adj_X_excl_Y_inter_consist_Z = adj_X_excl_Y.intersection(self.consistent_set( X, Y, adj_X_excl_Y, G_2))

				if len(adj_X_excl_Y_inter_consist_Z) >= d:
					# Get all the d-sets of the neighbours of X
					for Z in list(combinations(adj_X_excl_Y_inter_consist_Z, d)):
						# Independance test, knowing the neighbours
						if is_independant(learner, X, Y, Z):
							G_1.eraseEdge(X,Y)

							SeptSet_xy[tuple(sorted((X,Y)))].add(Z)
							break

						if X in G_1.neighbours(Y):
							break

			d += 1


		return G_1, SeptSet_xy

	def _orient_edges(self, G, SeptSet_xy):
		"""
		Phase 2: orient the skeleton's edges

		Note : APPARENTLY STILL A WAY TO MAKE DIRECTED CYCLES (RARE)
		"""
		# V-structures

		# Simple way to find unshielded triples:
		# for any node Z, all the triples X, Z, Y such that X and Y are neighbours
		# of Z, are unshielded triples.
		for Z in G.nodes():
			for X, Y in list(combinations(G.neighbours(Z), 2)):
				if Z not in SeptSet_xy[tuple(sorted((X, Y)))]:
					#R1 rule
					edge_to_arc(G, X, Z)
					edge_to_arc(G, Y, Z)


		was_oriented = True # Until no edge can be oriented
		nb_to_orient = len(G.edges()) # Early_stopping

		while was_oriented and nb_to_orient > 0:
			was_oriented = False

			for X,Y in SeptSet_xy.keys(): # For every X, Y pairs of nodes

				if nb_to_orient == 0: # If there are no more edges (only arcs) we stop
					break

				if not G.existsEdge(X,Y): # If X and Y are not neighbours

					shared_neighbours = G.neighbours(X).intersection(G.neighbours(Y))

					# For all unshielded triples and non fully-oriented v-structures
					for Z in shared_neighbours:

						# R2 rule
						if G.existsArc(X,Z) and G.existsEdge(Z,Y):
							edge_to_arc(G, Z, Y)
							nb_to_orient -= 1
							was_oriented = True
							if nb_to_orient == 0:
								break
				else:
					# If there is a cycle going trhough X and Y
					if G.existsEdge(X,Y) and G.hasDirectedPath(X, Y):

						#R3 rule
						edge_to_arc(G, X, Y)
						nb_to_orient -= 1
						was_oriented = True


		return {"graph":G}

	def S_k(self, G_0, G_kmin1, learner):
		_, SeptSet_xy = self.NewStep1(G_0, G_kmin1, learner)
		G_k = self._orient_edges(G_0, SeptSet_xy)["graph"]
		return  G_k

	def algorithm3(self, learner):
		# Init
		G_null = gum.MixedGraph()
		G_c = self.graph
		G_0 , _  = self.NewStep1(G_c, G_null, learner)#
		k = 0
		G_ks = [G_0]
		loop_detected = False
		while not loop_detected:
			k += 1
			G_k = self.S_k(G_0, G_ks[k-1], learner)
			for n in range(len(G_ks)-1 , 0 , -1):
				if G_ks[n] == G_k:
					loop_detected = True

					break
			G_ks.append(G_k)
		arcs = set()

		for j in range(k - n, len(G_ks)) :
			arcs = arcs.union(G_ks[j].arcs())

		to_delete = set()

		for arc in arcs:

			if reversed(arc) in arcs:
				to_delete.add(arc)
				to_delete.add(reversed(arc))

		try:
			for arc in to_delete:
				arcs.remove(arc)
			# arcs.remove(to_delete)
		except KeyError:
			print("no conflict in orientations")


		for arc in arcs:
			print(arc)
			self.graph.addArc(arc[0], arc[1])

		# TODO find "neighbours" equivalent when graph is directed
		ConsSeptSet_xy = {}
		for X,Y in self.graph.arcs():
			adj_X_excl_Y = self.graph.neighbours(X).copy()
			adj_X_excl_Y.remove(Y)
			adj_X_excl_Y_inter_consist_Z = adj_X_excl_Y.intersection(self.consistent_set( X, Y, adj_X_excl_Y, self.graph))
		return {"graph": self.graph, "SeptSet_xy": ConsSeptSet_xy}


	@save_result(save_prefix + "_final.png", save=save_final)
	def learn(self, bn, learner, verbose=True):

		if verbose is True: print("Initializing the graph..", end='\r')
		self._init_graph(bn)

		if verbose is True: print("Learning..", end='\r')
		self.algorithm3(learner)

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
