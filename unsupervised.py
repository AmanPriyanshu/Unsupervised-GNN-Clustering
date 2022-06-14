import numpy as np
from construct_base_graph import ConstructGraph
from IPython import get_ipython

try:
	shell = get_ipython().__class__.__name__
	if shell == 'ZMQInteractiveShell':
		from tqdm.notebook import tqdm   # Jupyter notebook or qtconsole
	elif shell == 'TerminalInteractiveShell':
		from tqdm import tqdm # Terminal running IPython
	else:
		from tqdm import tqdm  # Other type (?)
except NameError:
	from tqdm import tqdm 

class EncodeGraph:
	def __init__(self, path="./data/HR/HR_edges.csv", min_connections=10, embedding_dim=100, alpha=0.75, synchronic_update=True):
		self.synchronic_update = synchronic_update
		self.alpha = alpha
		self.cg = ConstructGraph(path=path, min_connections=min_connections)
		self.graph = self.cg.data
		self.embedding_dim = embedding_dim
		self.embeddings = np.random.normal(loc=0.0, scale=1.0, size=(len(self.cg.data_keys), self.embedding_dim))

	def embed(self, num_iters):
		progress = []
		for epoch in range(num_iters):
			if self.synchronic_update:
				embeddings = np.zeros((len(self.cg.data_keys), self.embedding_dim))
			bar = tqdm(self.graph.items())
			running_loss = 0.0
			for idx, (key_id,item_ids) in enumerate(bar):
				key = self.cg.data_keys.index(key_id)
				item = [self.cg.data_keys.index(str(item_id)) for item_id in item_ids if str(item_id) in self.cg.data_keys]
				neighbour_embeds = np.mean(self.embeddings[item], axis=0, keepdims=True)
				if self.synchronic_update:
					embeddings[int(key)] = self.alpha*self.embeddings[int(key)] + (1-self.alpha)*neighbour_embeds
					loss = np.mean(np.abs(embeddings[int(key)] - self.embeddings[int(key)]))
				else:
					embeddings = self.alpha*self.embeddings[int(key)] + (1-self.alpha)*neighbour_embeds
					loss = np.mean(np.abs(embeddings - self.embeddings[int(key)]))
					self.embeddings[int(key)] = embeddings
				running_loss += loss.item()
				bar.set_description(str({"iter_no": epoch+1, "loss": round(running_loss/(idx+1), 3)}))
			bar.close()
			progress.append(running_loss/(idx+1))
			if self.synchronic_update:
				self.embeddings = embeddings
		keys_arr = np.array(self.cg.data_keys)
		indices = np.argsort(keys_arr)
		self.embeddings = self.embeddings[indices]
		if self.synchronic_update:
			output_path = 'synchronic_updated_embedding.npy'
		else:
			output_path = 'asynchronic_updated_embedding.npy'
		with open(output_path, 'wb') as f:
			np.save(f, self.embeddings)
		return progress

if __name__ == '__main__':
	eg = EncodeGraph(min_connections=20, synchronic_update=False)
	progress = eg.embed(10)
	print(progress)
	#True = [0.20417472002595274, 0.1495597889762692, 0.11035480219710965, 0.08205113283742081, 0.06149363372346165, 0.04646298484540638, 0.0353981863614869, 0.02719463288748907, 0.021068037717305553, 0.01645958888075756]
	#False = [0.20209164553150366, 0.14904244252956836, 0.11052413719366071, 0.08243816918031266, 0.061867613989980606, 0.046727504912119536, 0.035526625973649945, 0.02719582396213272, 0.020965983489706186, 0.016281603695358346]