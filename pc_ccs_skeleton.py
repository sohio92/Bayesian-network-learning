import json
from itertools import combinations


from pc_ccs_orientation import PC_ccs
from utils import *

with open("parameters.json", 'r') as file:
	parameters = json.load(file)["parameters"]

save_folder = parameters["save_folder"]
save_prefix = parameters["save_prefix"]
save_steps = parameters["save_steps"]
save_final = parameters["save_final"]
save_compare = parameters["save_compare"]

class PC_ccs_skeleton(PC_ccs):
	"""
	Sepset consistent PC algorithm (2nd version, skeleton consistency)
	"""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def _skeleton_consistency(self, list_Sepset_xy):
		pass

	def learn(self, bn, learner, verbose=True):
		if verbose is True: print("Initializing the graph..")
		self._init_graph(bn)

		if verbose is True: print("Learning ...")
		list_Sepset_xy = self._orientation_consistency(learner)["list_Sepset_xy"]

		if verbose is True: print("Orienting the graph's edges..", end='\r')
		self._orient_edges(list_Sepset_xy)

		if verbose is True: print("Wrapping up the orientations..", end='\r')
		self._wrap_up_learning()

		try:
			self.learned_bn = graph_to_bn(self.graph)
		except:
			raise RuntimeError("Learning failed, learned BN contains cycles.")

		return {"graph":self.graph}

if __name__ == "__main__":
	bn, learner = generate_bn_and_csv(folder=save_folder).values()
	pc_ccs = PC_ccs_skeleton()

	pc_ccs.learn(bn, learner)
	_, hamming, skeleton_scores = pc_ccs.compare_learned_to_bn(bn).values()

	print("Hamming: {}\nSkeleton scores: {}\n".format(hamming, skeleton_scores))
	# print("\nProportion of failed learnings: {}%".format(round(test_robustness(PC_ccs) * 100, 3)))
