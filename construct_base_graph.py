import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import os

def read_json(path="./data/HR/HR_genres.json"):
	with open(path, 'r') as f:
		data = json.load(f)
	return data

def extract_vocab(data):
	vocab = []
	for key, item in data.items():
		vocab.extend(item)
		vocab = list(set(vocab))
	return vocab

class ConstructGraph:
	def __init__(self, path, min_connections):
		self.df = pd.read_csv(path)
		self.data_keys = []
		self.data_counts = []
		if not os.path.exists('tmp.json'):
			self.data = {}
			self.construct()
			with open('tmp.json', 'w') as json_file:
				json.dump(self.data, json_file)
		else:
			with open('tmp.json', 'r') as json_file:
				self.data = json.load(json_file)
			for key,item in self.data.items():
				self.data_keys.append(key)
				self.data_counts.append(len(item))
		self.min_connections = min_connections
		for key,count in zip(self.data_keys, self.data_counts):
			if count<self.min_connections:
				del self.data[key]
		self.data_keys = []
		self.data_counts = []
		for key,item in self.data.items():
			self.data_keys.append(key)
			self.data_counts.append(len(item))

	def construct(self):
		if type(self.df)!='pandas.core.frame.DataFrame':
			self.df = self.df.values
		self.df = self.df.tolist()
		for row in tqdm(self.df):
			if row[0] in self.data_keys:
				self.data[row[0]].append(row[1])
				self.data[row[0]] = list(set(self.data[row[0]]))
				self.data_counts[self.data_keys.index(row[0])] += 1
			else:
				self.data[row[0]] = [row[1]]
				self.data_keys.append(row[0])
				self.data_counts.append(1)
			if row[1] in self.data_keys:
				self.data[row[1]].append(row[0])
				self.data[row[1]] = list(set(self.data[row[1]]))
				self.data_counts[self.data_keys.index(row[1])] += 1
			else:
				self.data[row[1]] = [row[0]]
				self.data_keys.append(row[1])
				self.data_counts.append(1)

if __name__ == '__main__':
	data = read_json()
	vocab = extract_vocab(data)
	cg = ConstructGraph("./data/HR/HR_edges.csv", min_connections=10)
	print(np.unique(cg.data_counts, return_counts=True))