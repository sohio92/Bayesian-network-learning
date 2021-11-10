import json
import itertools

from pc import PC
from utils import *

with open("parameters.json", 'r') as file:
	parameters = json.load(file)["parameters"]

save_folder = parameters["save_folder"]
save_prefix = parameters["save_prefix"]
save_steps = parameters["save_steps"]
save_final = parameters["save_final"]
save_compare = parameters["save_compare"]

class PC_stable(PC):
	"""
	Stable version of PC, minor change to learn_skeleton
	"""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def _learn_skeleton(self, learner):
		"""
		Phase 1: learn the skeleton, PC stable
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

						if X not in self.graph.neighbours(Y):
							break
			d += 1

		return {"graph":self.graph, "Sepset_xy":Sepset_xy}


if __name__ == "__main__":
	bn, learner = generate_bn_and_csv(folder=save_folder).values()
	pc = PC()

	pc.learn(bn, learner)
	_, hamming, skeleton_scores = pc.compare_learned_to_bn(bn).values()

	print("Hamming: {}\nSkeleton scores: {}\n".format(hamming, skeleton_scores))
	print("\nProportion of failed learnings: {}%".format(round(test_robustness(PC) * 100, 3)))
