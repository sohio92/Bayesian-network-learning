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
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)


	def algorithm4(self, learner):
		# Init
		G_null = gum.MixedGraph()
		G_c = self.graph
		G_0, SeptSet_xy  = self.NewStep1(G_c, G_null, learner)
		k = 0
		G_ks = [G_0]
		loop_detected = False
		while not loop_detected:
			k += 1
			G_k, SeptSet_xy = self.NewStep1(G_0, G_ks[k-1], learner)
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
			print(arc)
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
			self.graph.addArc(arc[0], arc[1])

		return {"graph": self.graph, "SeptSet_xy": SeptSet_xy}


	@save_result(save_prefix + "_final.png", save=save_final)
	def learn(self, bn, learner):
		print("Initializing the graph..")
		self._init_graph(bn)

		print("Learning ...")
		self.algorithm4(learner)

		print("Wrapping up the orientations..")
		self._wrap_up_learning()

		try:
			self.learned_bn = graph_to_bn(self.graph)
		except:
			print("Learning failed, learned BN contains cycles.")

		return {"graph":self.graph}





if __name__ == "__main__":
	bn, learner = generate_bn_and_csv(folder=save_folder).values()
	pc_ccs = PC_ccs_skeleton()

	pc_ccs.learn(bn, learner)
	_, hamming, skeleton_scores = pc_ccs.compare_learned_to_bn(bn).values()

	print("Hamming: {}\nSkeleton scores: {}\n".format(hamming, skeleton_scores))
	# print("\nProportion of failed learnings: {}%".format(round(test_robustness(PC_ccs) * 100, 3)))
