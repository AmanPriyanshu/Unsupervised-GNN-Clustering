import torch
import numpy as np
from construct_base_graph import ConstructGraph
from IPython import get_ipython
import os
import pandas as pd

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

class EdgeModel(torch.nn.Module):
	def __init__(self, embedding_dim, vocab_size):
		super(EdgeModel, self).__init__()
		self.embedding_mat = torch.nn.Embedding(vocab_size, embedding_dim)
		self.linear_1 = torch.nn.Linear(embedding_dim, embedding_dim//2)
		self.linear_2 = torch.nn.Linear(embedding_dim//2, 1)
		self.activation = torch.nn.ReLU()
		self.output_activation = torch.nn.Sigmoid()

	def forward(self, x, x_):
		x = self.embedding_mat(x)
		x_ = self.embedding_mat(x_)
		x = torch.mul(x, x_)
		x = self.linear_1(x)
		x = self.activation(x)
		x = self.linear_2(x)
		x = self.output_activation(x)
		return x

class PredictEdges:
	def __init__(self, path="./data/HR/HR_edges.csv", min_connections=10, synchronic_update=True, embedding_dim=100, m=3):
		self.m = m
		self.cg = ConstructGraph(path=path, min_connections=min_connections)
		self.graph = self.cg.data
		self.synchronic_update = synchronic_update
		if self.synchronic_update:
			self.embeddings = np.load('synchronic_updated_embedding.npy')
		else:
			self.embeddings = np.load('asynchronic_updated_embedding.npy')
		self.embedding_dim = self.embeddings.shape[1]
		self.vocab_size = self.embeddings.shape[0]
		self.vocab = [int(i) for i in sorted(self.cg.data_keys)]
		self.criterion = torch.nn.BCELoss()
		self.data = []
		if not os.path.exists("data.csv"):
			for key, item_s in tqdm(self.graph.items(), desc="reading values"):
				non_config_item_s = np.random.choice([i for i in self.vocab if i not in item_s], 3*len(item_s))
				for idx, item in enumerate(item_s):
					try:
						self.data.append([self.vocab.index(int(key)), self.vocab.index(int(item)), 1])
						for i in range(self.m):
							self.data.append([self.vocab.index(int(key)), self.vocab.index(int(non_config_item_s[idx*self.m+i])), 0])
					except:
						pass
			self.data = np.array(self.data)
			self.data = pd.DataFrame(self.data)
			self.data.columns = ['Node1', 'Node2', 'Edge']
			self.data.to_csv("data.csv", index=False)
		else:
			self.data = pd.read_csv("data.csv")
		self.data = self.data.values
		self.model = EdgeModel(self.embedding_dim, self.vocab_size)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model.to(self.device)

	def single_epoch(self, epoch, train_size, batch_size=16):
		data = self.data[:]
		np.random.shuffle(data)
		data = data[:train_size]
		bar = tqdm([data[i:i+batch_size] for i in range(0, len(data), batch_size)])
		running_loss, running_acc = 0.0, 0.0
		for idx, batch in enumerate(bar):
			batch = torch.from_numpy(batch.T).to(self.device)
			x, x_, y = batch
			out = self.model(x, x_)
			self.optimizer.zero_grad()
			y = y.unsqueeze(1).float()
			loss = self.criterion(out, y)
			loss.backward()
			self.optimizer.step()
			running_loss += loss.item()
			pred = torch.round(out)
			acc = torch.mean((pred==y).float())
			running_acc += acc.item()
			bar.set_description(str({"epoch": epoch+1, "loss": round(running_loss/(idx+1), 3), "acc": round(running_acc/(idx+1), 3)}))
		bar.close()
		return running_loss/(idx+1), running_acc/(idx+1)

	def train(self, epochs, train_size):
		for epoch in range(epochs):
			self.single_epoch(epoch, train_size)
		return self.model

if __name__ == '__main__':
	pe = PredictEdges(min_connections=20)
	model = pe.train(10, train_size=300000)
	torch.save(model, "model.pt")