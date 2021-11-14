from pc_ccs_orientation import PC_ccs_orientation
from utils import *

save_folder = "Results/PC_CCS/Skeleton/"

class PC_ccs_skeleton(PC_ccs_orientation):
	"""
	Sepset consistent PC algorithm (2nd version, skeleton consistency)
	"""

	def __init__(self, *args, name="PC_CSS_Skeleton", **kwargs):
		super().__init__(*args, name=name, **kwargs)

	def _skeleton_consistency(self, Sepset_xy_k):
		for removed_edge in get_missing_edges(self.graph):
			if Sepset_xy_k[tuple(sorted(removed_edge))] not in consistent_set(self.graph, *removed_edge) \
				and Sepset_xy_k[tuple(sorted(removed_edge))] not in consistent_set(self.graph, *removed_edge[::-1]):
				self.graph.addEdge(*removed_edge)
		
		return {"graph":self.graph}

	def learn(self, bn, learner, verbose=True, save_steps=False, save_final=True, save_folder="Results/"):
		if verbose is True: print("Initializing the graph..")
		self._init_graph(bn)
		if save_steps is True:	self.save_graph("init", folder=save_folder)

		if verbose is True: print("Learning ...")
		Sepset_xy_0, Sepset_xy_k = self._orientation_consistency(learner)["Sepset_xy_0"], self._orientation_consistency(learner)["Sepset_xy_k"]
		if save_steps is True:	self.save_graph("orientation_consistency", folder=save_folder)

		if verbose is True: print("Orienting the graph's edges..", end='\r')
		self._orient_edges(Sepset_xy_0)
		if save_steps is True:	self.save_graph("oriented", folder=save_folder)

		if verbose is True: print("Propagating the orientations..", end='\r')
		self._propagate_orientations()
		if save_steps is True:	self.save_graph("propagated", folder=save_folder)

		if verbose is True: print("Final consistency check..", end='\r')
		self._skeleton_consistency(Sepset_xy_k)
		if save_steps or save_final is True:	self.save_graph("final", folder=save_folder)

		try:
			self.learned_bn = graph_to_bn(self.graph, self.nb_values)
		except:
			raise RuntimeError("Learning failed, learned BN contains cycles.")

		return {"graph":self.graph}

if __name__ == "__main__":
	bn, learner = generate_bn_and_csv(folder=save_folder).values()
	pc_ccs = PC_ccs_skeleton()

	pc_ccs.learn(bn, learner, save_folder=save_folder)
	_, hamming, skeleton_scores = pc_ccs.compare_learned_to_bn(bn).values()

	print("Hamming: {}\nSkeleton scores: {}\n".format(hamming, skeleton_scores))
	print("\nProportion of failed learnings: {}%".format(round(test_robustness(PC_ccs_orientation) * 100, 3)))
