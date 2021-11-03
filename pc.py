import pyAgrum as gum
import pyAgrum.lib.image as gimg
import pyAgrum.lib.bn_vs_bn as bvb
from itertools import combinations

from utils import *

save_folder = "Results/"
save_prefix = "learned_bn"
save_steps = False
save_final = not(save_steps)
save_compare = True

class PC():
	"""
	PC algorithm to learn a Bayesian Network
	"""
	def __init__(self):
		self.graph = gum.MixedGraph()
		self.learned_bn = None

	@save_result(save_prefix + "_0.png", save=save_steps)
	def _init_graph(self, bn):
		self.graph.clear()
		for n in bn.nodes():
			self.graph.addNodeWithId(n)

		for x in self.graph.nodes():
			for y in self.graph.nodes():
				if x != y:  self.graph.addEdge(x,y)

		return {"graph":self.graph}

	@save_result(save_prefix + "_1.png", save=save_steps)
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
		
		SeptSet_xy = {} # Z for every pair X, Y
		X_Y_pairs = list(combinations(self.graph.nodes(), 2))
		for X,Y in X_Y_pairs:
				SeptSet_xy[(X,Y)] = []

		while has_more_neighbours(self.graph, d):
			for X,Y in self.graph.edges():
				adj_X_excl_Y = self.graph.neighbours(X).copy()
				adj_X_excl_Y.remove(Y)

				if len(adj_X_excl_Y) >= d:
					# Get all the d-sets of the neighbours of X
					for Z in list(combinations(adj_X_excl_Y, d)):
						# Independance test, knowing the neighbours
						if is_independant(learner, X, Y, Z):
							self.graph.eraseEdge(X,Y)

							SeptSet_xy[tuple(sorted((X,Y)))].append([Z])
							break

			d += 1

		# d = 0
		# SeptSet_xy = {}  # Z for every pair X, Y
		# while has_more_neighbours(self.graph, d):
		#     for X, Y in self.graph.edges():
		#         adj_X_excl_Y = self.graph.neighbours(X).copy()
		#         adj_X_excl_Y.remove(Y)
		#
		#         if len(adj_X_excl_Y) >= d:
		#             # Get all the d-sets of the neighbours of x
		#             for Z in itertools.combinations(adj_X_excl_Y, d):
		#                 # Independance test, knowing the neighbours
		#                 if is_independant(learner, X, Y, Z):
		#                     self.graph.eraseEdge(X, Y)
		#
		#                     if (X, Y) in SeptSet_xy.keys(): SeptSet_xy[(X, Y)] += Z
		#                     else:   SeptSet_xy[(X, Y)] = Z
		#
		#                     break
		#     d += 1

		# d = 0
		# SeptSet_xy = {} # Z for every pair X, Y
		# while has_more_neighbours(self.graph, d) is True:
		#     for x in self.graph.nodes():
		#         if len(self.graph.neighbours(x)) >= d + 1:
		#             for y in self.graph.neighbours(x):
		#                 # Get all the d-sets of the neighbours of x
		#                 if d == 0:  new_neigh = [[]]
		#                 else:
		#                     new_neigh = [i for i in self.graph.neighbours(x) if i != y]
		#                     new_neigh = [new_neigh[i:i+d] for i in range(0, len(self.graph.neighbours(x)), d)]

		#                 # Independance test, knowing the neighbours
		#                 for z in new_neigh:
		#                     if is_independant(learner, x, y, z) is True:
		#                         self.graph.eraseEdge(x,y)
		#                         if (x,y) in SeptSet_xy.keys():  SeptSet_xy[(x,y)] += z
		#                         else:   SeptSet_xy[(x,y)] = z

		#     d += 1

		return {"graph":self.graph, "SeptSet_xy":SeptSet_xy}

	@save_result(save_prefix + "_2.png", save=save_steps)
	def _orient_edges(self, SeptSet_xy):
		"""
		Phase 2: orient the skeleton's edges

		Note : APPARENTLY STILL A WAY TO MAKE DIRECTED CYCLES (RARE)
		"""
		# V-structures

		# Simple way to find unshielded triples:
		# for any node Z, all the triples X, Z, Y such that X and Y are neighbours
		# of Z, are unshielded triples.
		for Z in self.graph.nodes():
			for X, Y in list(combinations(self.graph.neighbours(Z), 2)):
				if Z not in SeptSet_xy[tuple(sorted((X, Y)))]:
					#R1 rule
					edge_to_arc(self.graph, X, Z)
					edge_to_arc(self.graph, Y, Z)


		was_oriented = True # Until no edge can be oriented
		nb_to_orient = len(self.graph.edges()) # Early_stopping

		while was_oriented and nb_to_orient > 0:
			was_oriented = False

			for X,Y in SeptSet_xy.keys(): # For every X, Y pairs of nodes

				if nb_to_orient == 0: # If there are no more edges (only arcs) we stop
					break

				if not self.graph.existsEdge(X,Y): # If X and Y are not neighbours

					shared_neighbours = self.graph.neighbours(X).intersection(self.graph.neighbours(Y))

					# For all unshielded triples and non fully-oriented v-structures
					for Z in shared_neighbours:

						# R2 rule
						if self.graph.existsArc(X,Z) and self.graph.existsEdge(Z,Y):
							edge_to_arc(self.graph, Z, Y)
							nb_to_orient -= 1
							was_oriented = True
							if nb_to_orient == 0:
								break
				else:
					# If there is a cycle going trhough X and Y
					if self.graph.existsEdge(X,Y) and self.graph.hasDirectedPath(X, Y):

						#R3 rule
						edge_to_arc(self.graph, X, Y)
						nb_to_orient -= 1
						was_oriented = True

		# for z in self.graph.nodes():
		#     neigh = list(self.graph.neighbours(z))
		#     for i in range(len(neigh)):
		#         for y in neigh[i+1:]:
		#             x = neigh[i]
		#             if self.graph.existsEdge(x, y) is False:
		#                 # Unshielded triple
		#                 if (x,y) in SeptSet_xy.keys() and z not in SeptSet_xy[(x,y)]:
		#                     edge_to_arc(self.graph, x, z)
		#                     edge_to_arc(self.graph, y, z)
		#
		# # Propagation
		# was_oriented = True # Until no edge can be oriented
		# while was_oriented is True:
		#     was_oriented = False
		#     for x in self.graph.nodes():
		#         for y in self.graph.nodes():
		#             if self.graph.existsEdge(x, y) is False and self.graph.existsArc(x, y) is True:
		#                 # No v-structure added
		#                 for z in self.graph.neighbours(y):
		#                     if self.graph.existsArc(x, z) and self.graph.existsEdge(z, y) is True:
		#                         edge_to_arc(self.graph, z, y)
		#                         was_oriented = True
		#
		#             elif self.graph.existsEdge(x, y) is True and self.graph.hasDirectedPath(x, y):
		#                 # No cycle
		#                 edge_to_arc(self.graph, x, y)
		#                 was_oriented = True

		return {"graph":self.graph}

	@save_result(save_prefix + "_3.png", save=save_steps)
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

	@save_result("comparated_bn.png", save=save_compare)
	def compare_learned_to_bn(self, bn):
		if self.learned_bn is None: return

		comparator = bvb.GraphicalBNComparator(bn, self.learned_bn)

		return {"graph":comparator.dotDiff(), "hamming":comparator.hamming(), "skeletonScores":comparator.skeletonScores()}

if __name__ == "__main__":
	bn, learner = generate_bn_and_csv(n_data=10000).values()

	pc = PC()
	pc.learn(bn, learner)

	_, hamming, skeletonScores = pc.compare_learned_to_bn(bn).values()
	print("Hamming: {}\nSkeleton scores: {}".format(hamming, skeletonScores))
