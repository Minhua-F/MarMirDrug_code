import os
from sklearn.cluster import KMeans

def K_means_clustering(k, txtpath, datapath):
	entity2idPath = 'entity2id.txt'
	relation2idPath = 'relation2id.txt'
	pretrainedEmbPath = 'entity2vec.bern'
	graphPath = 'graph.txt'

	# read entity IDs
	entity2idPath = os.path.join(txtpath, entity2idPath)

	with open(entity2idPath, "r") as f:
		entity2id = {}
		next(f)
		for line in f.readlines():
			str_list = line.strip("\n").split("\t")
			entity = str_list[0]
			eid = str_list[1]
			# entity, eid = line.split()
			entity2id[entity] = int(eid)

	# read relation IDs
	relation2idPath = os.path.join(txtpath, relation2idPath)

	with open(relation2idPath, "r") as f:
		relation2id = {}
		next(f)
		for line in f.readlines():
			str_list = line.strip("\n").split("\t")
			relation = str_list[0]
			rid = str_list[1]
			# relation, rid = line.split()
			relation2id[relation] = int(rid)

	# read entity embeddings
	pretrained_emb_file = os.path.join(datapath, pretrainedEmbPath)

	entity2emb = []
	with open(pretrained_emb_file, "r") as f:
		for line in f:
			entity2emb.append([float(value) for value in line.split()])

	# entity2emb = np.load(pretrained_emb_file)
	# entity2emb = list(entity2emb)

	# K Means CLustering
	kmeans_entity = KMeans(n_clusters=k, random_state=0).fit(entity2emb)

	# assign cluster label to entities
	entity2cluster = {}

	for idx, label in enumerate(kmeans_entity.labels_):
		entity2cluster[idx] = int(label)

	# print(entity2cluster)

	ent2clusterFile = os.path.join(datapath, 'entity2clusterid.txt')
	with open(ent2clusterFile, 'w') as f:
		for ent in entity2cluster.keys():
			f.write(str(ent) + '\t' + str(entity2cluster[ent]) + '\n')