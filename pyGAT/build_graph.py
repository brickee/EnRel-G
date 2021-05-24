import os
import json
import numpy as np
import scipy.sparse as sp
import torch
import itertools
import string
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from torch.nn.utils.rnn import pad_sequence
import scipy
import pickle

lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()
nltk.download('stopwords')
nltk.download('wordnet')
noises = set(list(string.punctuation) + list(string.digits)+list(nltk.corpus.stopwords.words('english')+['[sep]','[cls]', '[SEP]','[CLS]']))


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def split(word):
    return list(word)


def acronyms(tokens):

    result = []
    for id, token in enumerate(tokens):
        flag = False
        # if token.isupper() and tokens[id - 1] == '(':
        if tokens[id - 1] == '(':
            flag = True
            for i in range(len(token)):
                # TODO: cannot cover all situations,like Day 1 (Dy1)
                # print(tokens[id-i-2], tokens[id-1-1],  token,token[-i-1])
                try:
                    if tokens[id - i - 2][0].lower() != token[-i - 1].lower():
                        flag = False
                except:
                    continue
        if flag:
            result.append(id)

    return result





def get_edges(sentence, pos_id,dep_id, head,longest_token_sequence_in_batch):

    pos_map = {9:'NOUN',13: 'PROPN'}
    # edge a for lexicon clues
    edges_a = []
    # edge b for adjacent tokens
    edges_b = []

    for combs in itertools.combinations(range(len(sentence)),2):

        try:
            # when run with multiple gpus, [CLS] cause error
            lemma1 = lemmatizer.lemmatize(sentence[combs[0]] , pos="n").lower()
        except:
            lemma1 = sentence[combs[0]]
        if (lemma1 not in noises) and (
                    (pos_id[combs[0]] in pos_map) or (pos_id[combs[1]] in pos_map)):
            # Exactly the same
            if (sentence[combs[0]] == sentence[combs[1]]):

                # Index of token starts from 0!
                edge = [combs[0], combs[1]]
                if edge not in edges_a:
                    edges_a.append(edge)

            # Lemma Match
            elif lemma1 == lemmatizer.lemmatize(sentence[combs[1]] , pos="n").lower():

                edge = [combs[0], combs[1]]
                if edge not in edges_a:
                    edges_a.append(edge)

    abbr_idx = acronyms(sentence)
    for i in abbr_idx:
        for j in range(len(sentence[i])):
            # tkn = tokens[i - j - 2]
            #Now the idx starts from 0
            if i-j-2>=0:
                edge = [i , i-j -2]
                if edge not in edges_a:
                    edges_a.append(edge)

    edges_f,edges_g = [],[]

    sentence[-1] = '.'

    tokens_text = [text.replace('.', '$').replace('!','$').replace('?','$') if text not in  ['.','?','!'] else text for text in sentence ]
    sen_list = [list(filter(('').__ne__, (l+'.').split('|*|'))) for l in '|*|'.join(tokens_text).replace('?','.').replace('!','.').split('.')][:-1]

    assert len(tokens_text) == sum([len(sub) for sub in sen_list])
    bias = 0
    biases = []


    subobj_map = {31:'nsubj',22:'dobj',35:'obj', 32:'nsubjpass'}
    root_map = {'ROOT':47}
    prep_map = {42:'prep'}
    probj_map = {39:'pobj'}

    comp_map = {14:'compound'}
    mod_map = {6:'amod', 33:'nummod'}  # this part does not work
    nmod_map = {29: 'nmod',  49: 'nmod:npmod', 50: 'nmod:poss'}
    all_map = comp_map.copy()
    all_map.update(mod_map)
    all_map.update(nmod_map)

    # Only consider the local sentences!
    for m,sen_tkns in enumerate(sen_list):
        for combs in itertools.combinations(range(len(sen_tkns)), 2):
            try:
  

                idx1 = bias+combs[0]
                idx2 = bias+combs[1]
                dep1 = dep_id[bias + combs[0]]
                dep2 = dep_id[bias + combs[1]]
                head1 = head[bias + combs[0]]
                head2 = head[bias + combs[1]]


                ##################################################
                #### Connect all dependency trees ################
                ##################################################

                # all dependency tree


                # if (head1==head2 ) and (head1!=-1):
                #
                #     edge = [idx1, idx2]
                #     if edge not in edges_f:
                #         edges_f.append(edge)
                #
                #     if int(head1 + 1) < longest_token_sequence_in_batch and int(
                #             head1) > 0:  # TODO: bug in data preprocessing!
                #
                #         edge = [idx1, int(
                #             head1) + 1]  # I checked that the idx of head is from idx of token, which starts from 0, but here we add [CLS] at first!, so add back
                #         if edge not in edges_f:
                #             edges_f.append(edge)
                #         edge = [idx2, int(head2) + 1]
                #         if edge not in edges_f:
                #             edges_f.append(edge)
                # all but only Nouns


                ##################################################
                #### Connect sub/obj and heads ##################
                ##################################################

                # connect different dependent obj/sub, and root as predicate
                # if (dep1 in subobj_map) and (dep2 in subobj_map) and (head1==head2 ) and (dep1!=dep2) and (subobj_map[dep1][-3:]!= subobj_map[dep2][-3:]) and dep_head ==root_map['ROOT']:
                # head='ROOT' is new compared to previous flat platform
                if (dep1 in subobj_map) and (dep2 in subobj_map) and (head1 == head2) and (dep1 != dep2) and (
                        subobj_map[dep1][-3:] != subobj_map[dep2][-3:]) and (
                        subobj_map[dep1][:5] != subobj_map[dep2][:5]):

                        # Not for mars
                        # and dep_head == root_map['ROOT']:

                # connect all dependent words
                # if (head1 == head2):

                # conncet all dependent Nouns
                # if (head1 == head2) and (pos1 in pos_list) and (pos2 in pos_list):
                #     print([(tkn, i) for i, tkn in enumerate(sentence)])
                #     print(bias + combs[0], bias + combs[1], dep1, dep2, subobj_map[dep1], subobj_map[dep2], head1, head2)
                #     print(dep_id)
                #     print(head)

                    edge = [idx1, idx2]
                    if edge not in edges_f:
                        edges_f.append(edge)

                    # connect head too!  Head start from 0!
                    if int(head1+1)<longest_token_sequence_in_batch and int(head1)>0: # TODO: bug in data preprocessing!

                        edge = [idx1, int(head1)+1]  # I checked that the idx of head is from idx of token, which starts from 0, but here we add [CLS] at first!, so add back
                        if edge not in edges_f:
                            edges_f.append(edge)
                        edge = [idx2, int(head2)+1]
                        if edge not in edges_f:
                            edges_f.append(edge)

            except:
                # assert sentence[-1] not in ['.', '!', '?']
                continue


        biases.append(bias)
        bias+=len(sen_tkns)

    # add compound, Only compound works
    # add amod/nummod
    # add nmod
    for edge in edges_f:
        for node in edge:
            for i in range(len(sentence)):
                if dep_id[i] in comp_map and (int(head[i])+1) == node:
                # if dep_id[i] in mod_map and (int(head[i]) + 1) == node:
                # if dep_id[i] in nmod_map and (int(head[i]) + 1) == node:
                # if dep_id[i] in all_map and (int(head[i]) + 1) == node:
                    com_edge = [node, i]
                    if com_edge not in edges_f:

                        edges_f.append(com_edge)

    return np.asarray(edges_a),np.asarray(edges_b),np.asarray(edges_f),np.asarray(edges_g)





def edge2adj(edges,longest_token_sequence_in_batch):
    if len(edges):
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(longest_token_sequence_in_batch, longest_token_sequence_in_batch), dtype=np.float32)
    else:
        adj = sp.coo_matrix(
            np.zeros(shape=(longest_token_sequence_in_batch, longest_token_sequence_in_batch), dtype=np.float32))

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # diag matrix sp.eye(adj.shape[0])
    adj = adj + sp.eye(adj.shape[0])
    return adj


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)



def build_graph(sentences,max_len, pos_ids, dep_ids, head):

    # lengths = [len(sentence) for sentence in sentences]
    # longest_token_sequence_in_batch = max(lengths)
    longest_token_sequence_in_batch = max_len

    # assert longest_token_sequence_in_batch <= 1024
    adjs_a, adjs_b, adjs_f, adjs_g = [],[],[],[] # for combine adjs from sentence to doc.
    for i in range(len(sentences)):


        # tokens = [token for token in sent]
        edges_a,edges_b,edges_f,edges_g = get_edges(sentences[i],pos_ids[i],dep_ids[i], head[i],longest_token_sequence_in_batch)
        adj_a = edge2adj(edges_a,longest_token_sequence_in_batch)

        adjs_a.append(adj_a)

        adj_b = edge2adj(edges_b, longest_token_sequence_in_batch)
        adjs_b.append(adj_b)

        adj_f = edge2adj(edges_f, longest_token_sequence_in_batch)

        adjs_f.append(adj_f)

        adj_g = edge2adj(edges_g, longest_token_sequence_in_batch)
        adjs_g.append(adj_g)

        a = adj_a.todense()
        assert check_symmetric(a)
        f = adj_f.todense()
        assert check_symmetric(f)


    adj_matrix_a = torch.tensor(np.asarray(sp.vstack(adjs_a).todense()),dtype=torch.float32).view(-1, longest_token_sequence_in_batch,
                                                                       longest_token_sequence_in_batch)
    adj_matrix_b = torch.tensor(np.asarray(sp.vstack(adjs_b).todense()),dtype=torch.float32).view(-1, longest_token_sequence_in_batch,
                                                                                longest_token_sequence_in_batch)
    adj_matrix_f = torch.tensor(np.asarray(sp.vstack(adjs_f).todense()), dtype=torch.float32).view(-1,
                                                                                                   longest_token_sequence_in_batch,
                                                                                                   longest_token_sequence_in_batch)
    adj_matrix_g = torch.tensor(np.asarray(sp.vstack(adjs_g).todense()), dtype=torch.float32).view(-1,
                                                                                                   longest_token_sequence_in_batch,
                                                                                                   longest_token_sequence_in_batch)

    dist_As = []
    return adj_matrix_a,adj_matrix_b,adj_matrix_f,adj_matrix_g,dist_As





def get_sen_embed(docs, embeds):

    seg_idx_list = []
    sen_embed_batch = []
    for i,sen in enumerate(docs):
        embed = embeds[i]
        seg_idx = [x.idx for  x in sen.tokens if x.text in ['.', '?', '!']]
        seg_idx_list.append(seg_idx)
        sen_embeds = []
        last_i = 0
        for seg_i in seg_idx:
            sen_embed = torch.mean(embed[last_i:seg_i],dim = 0)
            sen_embeds.append(sen_embed)
            last_i = seg_i
        if not sen_embeds:
            sen_embed_batch.append(torch.mean(embed[0:len(sen)],dim=0).unsqueeze(0))
        else:
            sen_embed_batch.append(torch.stack(sen_embeds))
    return sen_embed_batch,seg_idx_list

def _cosine_similarity(vec0:torch.FloatTensor, vec1:torch.FloatTensor):
    assert vec0.dim() == 1
    assert vec1.dim() == 1
    return F.cosine_similarity(vec0.unsqueeze(0), vec1.unsqueeze(0), dim=1)




def build_sim_graph(sen_embed_batch):
    num_sent = max([len(sen_embeds) for sen_embeds in sen_embed_batch])
    cosine_As = []
    for sen_embeds in sen_embed_batch:
        cosine_A = np.zeros((num_sent, num_sent))
        for i, sent_i in enumerate(sen_embeds):
            for j, sent_j in enumerate(sen_embeds):
                if i<=j:
                    cosine_A[i][j] = _cosine_similarity(sent_i, sent_j)
                    cosine_A[j][i] = cosine_A[i][j]

        cosine_As.append(torch.tensor(cosine_A,dtype=torch.float32))
    cosine_As =  torch.stack(cosine_As)

    return cosine_As


def build_dist_graph(sen_embed_batch):
    num_sent = max([len(sen_embeds) for sen_embeds in sen_embed_batch])
    dist_As = []
    for sen_embeds in sen_embed_batch:
        dist_A = np.zeros((num_sent, num_sent))
        for i, sent_i in enumerate(sen_embeds):
            gauss = scipy.stats.norm(i, 0.75)  # 0.75 can be modified
            for j, sent_j in enumerate(sen_embeds):
                dist_A[i, j] = gauss.pdf(j)

        dist_As.append(torch.tensor(dist_A,dtype=torch.float32))
    dist_As =  torch.stack(dist_As)

    return dist_As

def combine_embeds(sentence_tensor,sen_embed_batch_pad,seg_idx_list):
    out_tensor = []
    for i in range(len(sentence_tensor)):
        word_embeds = sentence_tensor[i]
        sen_embeds = sen_embed_batch_pad[i]
        seg_idx = seg_idx_list[i]
        last_idx = 0

        com_embeds = []

        for j in range(len(seg_idx)):
            idx = seg_idx[j]
            sen_embed = sen_embeds[j]
            com_embed = torch.cat((word_embeds[last_idx:idx],sen_embed.unsqueeze(0).repeat(idx-last_idx,1)),dim=1)
            com_embeds.append(com_embed)
            last_idx = idx

        # zero sentence splitting
        if not com_embeds:
            # Only the first sentence embedding is meaningful
            com_embeds = torch.cat((word_embeds, sen_embeds[0].unsqueeze(0).repeat(len(word_embeds), 1)), dim=1)
        else:
            com_embeds = torch.cat(com_embeds, dim=0)


        out_tensor.append(com_embeds)

    out_tensor = pad_sequence(out_tensor,batch_first=True,padding_value=0)
    # TODO: check what really happens here!
    if  out_tensor.size(1) != sentence_tensor.size(1):
        pad_tensor = torch.zeros(out_tensor.size(0),sentence_tensor.size(1)- out_tensor.size(1),out_tensor.size(2)).to(flair.device)
        out_tensor = torch.cat((out_tensor,pad_tensor),dim=1)
    assert out_tensor.size(1) == sentence_tensor.size(1)
    return out_tensor

def store_tkn_dist_graph():
    longest_token_sequence_in_batch = 2000
    dist_A = np.zeros((longest_token_sequence_in_batch, longest_token_sequence_in_batch))
    for i in range(longest_token_sequence_in_batch):
        gauss = scipy.stats.norm(i, 0.75)  # 0.75 can be modified
        for j in range(longest_token_sequence_in_batch):
            dist_A[i, j] = gauss.pdf(j)
    print(dist_A)
    with open('../data/AnatEM-1.0.2/tkn_dist_graph.pickle', 'wb') as handle:
        pickle.dump(dist_A, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_tkn_dist_graph(max_len,lengths):
    batch_size = len(lengths)

    with open('./data/AnatEM-1.0.2/tkn_dist_graph.pickle', 'rb') as handle:
        dist_As = pickle.load(handle)
    # print(dist_As.shape)
    crop_dist_As = torch.tensor(np.tile(dist_As[:max_len,:max_len],(batch_size,1,1)),dtype=torch.float32)
    # print(crop_dist_As.shape)
    mask = torch.zeros_like(crop_dist_As,dtype=torch.bool)
    for i,length in enumerate(lengths):

        mask[i][:length,:length] = True
    # print(mask)
    crop_dist_As = crop_dist_As.masked_fill(mask, 0)
    # print(crop_dist_As)
    return crop_dist_As



    # print(edges_a)





