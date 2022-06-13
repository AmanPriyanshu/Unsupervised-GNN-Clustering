import numpy as np
from tqdm import tqdm
from .construct_base_graph import ConstructGraph

class EncodeGraph:
	def __init__(self, path="./data/HR/HR_edges.csv", min_connections=10, embedding_dim=100, alpha=0.75, synchronic_update=True):
		self.synchronic_update = synchronic_update
		self.alpha = alpha
		self.cg = ConstructGraph(path=path, min_connections=min_connections)
		self.graph = self.cg.data
		self.embedding_dim = embedding_dim
		self.embeddings = np.random.normal(loc=0.0, scale=1.0, size=(len(self.cg.data_keys), self.embedding_dim))

	def embed(self, num_iters):
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
			if self.synchronic_update:
				self.embeddings = embeddings
		if self.synchronic_update:
			output_path = 'synchronic_updated_embedding.npy'
		else:
			output_path = 'asynchronic_updated_embedding.npy'
		with open(output_path, 'wb') as f:
			np.save(f, self.embeddings)

if __name__ == '__main__':
	eg = EncodeGraph(min_connections=20, synchronic_update=False)
	eg.embed(5)
	#0.2, 0.16, 0.12, 0.08, 0.06