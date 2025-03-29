# change embedding to torch

EntOrRel = 'Ent'
Method_type = 'TransE'
InVar = embeddings_list

input_ent = torch.LongTensor(range(InVar.num_embeddings))
ent_cur = floatTensor(InVar(input_ent))
with open('./Embed_' + args.dataset + '_' + EntOrRel + '_' + Method_type + '.txt', 'w') as log_file:
    ent_cur_cpu = ent_cur.cpu()
    np.savetxt(log_file, ent_cur_cpu.detach().numpy())

