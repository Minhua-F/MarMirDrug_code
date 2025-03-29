import torch
import resource
emb = torch.nn.Embedding(3, 5)
print(emb(torch.LongTensor([1])))


re = resource.getrusage(resource.RUSAGE_SELF)
print(re)