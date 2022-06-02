import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

torch.autograd.set_detect_anomaly(True)

from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch, DataLoader

from collections import defaultdict


import numpy as np
import matplotlib.pyplot as plt


class TGN(nn.Module):
    """
    Attributes
    ----------
    T : time neighbourhoood
    d_t : dimension of encoded time

    d_v : dimension of node 
    d_e : dimension of edge 
    d_h : dimension of node embedded state
    d_s : dimension of node memory state
    d_rm_n : dimension of raw-message from node-wise event
    d_rm_i : dimension of raw_message from interaction event

    l : number of GAT layers
    k : number of attention heads
    d_q : dimension of query
    d_c : dimension of concatenated neighbours
    n : maximum number of neighbours per node used in GAT (take the last n edge events)

    rm : raw messages
    om : old messages (= m : messages)
    am : aggregated messages
    
    raw_message_store :  v_i, e_ij -- stores node_wise events v_i and interaction events e_ij at every timestep
                         For a directed edge enter only (i,j), for undirected (i,j) and (j,i) in interaction events
    memory :             s_i -- stores memory vector s_i for every node
    message_function_v : msg_v() -- message to node that has a node_wise_event
    message_function_e : msg_e() -- message to both nodes in an interaction_event
    message_aggregator : agg() -- aggregates all past messages before using them to update memory
    memory_updater :     mem() -- use aggregated message to update the memory
    phi :                phi() -- time embedding
    embedder :           h_i -- temporal graph attention to embedd nodes
    decoder :            p_e_ij -- edge probabilities 
    
    Methods
    -------
    train_loss :         BCE based on negative log likelihood of only those edges, that we put in as input for the next timestep.
    predict_links :      Predict all edges (bidirectional, (i,j) and (j,i)) of the graph for the next timestep (not just new edges).
    add_batch_to_graph : Update the graph with new nodes and edges events (add / delete / modify)
    """

    def __init__(self, G, TGN_hyperparams):
        super(TGN, self).__init__()

        l = TGN_hyperparams["l"]
        k = TGN_hyperparams["k"]
        n = TGN_hyperparams["n"]
        d_t = TGN_hyperparams["d_t"]
        d_v = TGN_hyperparams["d_v"]
        d_e = TGN_hyperparams["d_e"]
        d_m = TGN_hyperparams["d_m"]
        d_h = d_v
        d_s = d_v
        d_rm_v = d_s + 1 + d_v
        d_rm_e = 2 * d_s + 1 + d_e
        d_q = d_h + d_t
        d_c = d_h + d_t + d_e

        self.raw_message_store = RawMessageStore(G)
        self.memory = Memory(N_0, d_s)
        self.message_function_v = MessageFunction(d_rm_v, d_m)
        self.message_function_e = MessageFunction(d_rm_e, d_m)
        self.message_aggregator = get_MessageAggregator(reduction="mean")
        self.memory_updater = MemoryUpdater(d_m, d_s)
        self.phi = time2vec(d_t)
        self.embedder = TemporalGraphAttention(l, k, n, d_c, d_h, d_t)
        self.decoder = LinkPrediction()

    def old_messages(self):
        """
        Arguments
        ---------
        -
        
        Returns
        -------
        old_messages : dict( int : {torch.tensor[1] : torch.tensor[d_m]})
                       dict( i : {t : m})
        """
        old_messages = dict()
        v = self.raw_message_store.node_wise_events
        e = self.raw_message_store.interaction_events_individual

        for i in v.keys():
            s_i = self.memory.S[i]
            v_i_t = v[i]
            e_i_t = e[i]

            old_messages_i = defaultdict(list)
            for t, v_i in v_i_t.items():
                m_i = self.message_function_v([s_i, t, v_i])
                old_messages_i[t].append(m_i)
            for j, e_ij_t in e_i_t.items():
                for t, (e_ij, delta_t) in e_ij_t.items():
                    s_j = self.memory.S[j]
                    m_i = self.message_function_e([s_i, s_j, t, e_ij])
                    old_messages_i[t].append(m_i)
            old_messages[i] = old_messages_i
        return old_messages

    def aggregated_messages(self, old_messages):
        """
        Arguments
        ---------
        om : dict( int : {torch.tensor[1] : torch.tensor[d_m]})
             dict( i : {t : m})
             
        Returns
        -------
        am : dict( int : torch.tensor[d_m] )
             dict( i : am_i)
        """
        am = self.message_aggregator(old_messages)
        return am

    def update_memory(self, am):
        """
        Arguments
        ---------
        am : dict( int : torch.tensor[d_m] )
             dict( i : am_i)
             
        Returns
        -------
        -
        """
        S_next = self.memory_updater(am, self.memory.S)
        self.memory(S_next)

    def init_hidden_states(self):
        """
        Arguments
        ---------
        -
        
        Returns
        -------
        h : torch.tensor[N, d_h]
        """
        N = len(self.memory.S)
        h = list()
        for i in range(N):
            s_i = self.memory.S[i]
            t = list(self.raw_message_store.node_wise_events[i].keys())[-1]
            v_i = self.raw_message_store.node_wise_events[i][t]
            h_i = s_i + v_i
            h.append(h_i)
        h = torch.stack(h, dim=0)
        return h

    def embedd(self, t):
        """
        Arguments
        ---------
        t : torch.tensor[1]
        
        Returns
        -------
        h : torch.tensor[N, d_h]
        """
        h = self.init_hidden_states()
        e = self.raw_message_store.interaction_events_individual
        phi_0 = self.phi.t0()
        phi_t = self.phi.t_minus_tj(e, t)
        h = self.embedder(h, phi_t, phi_0, e)
        return h

    def loss_fn(self, p_e_ij, e_ij):
        """
        Arguments
        ---------
        p_e_ij: torch.tensor[E_new, 1]
        e_ij: torch.tensor[E_new, 1]
        
        Returns
        -------
        BCE : torch.tensor[1]
        
        """
        ones = torch.ones_like(p_e_ij)

        BCE = -(
            e_ij * torch.log(p_e_ij) + (ones - e_ij) * torch.log(ones - p_e_ij)
        ).sum(dim=0)
        return BCE

    def forward(self, t):
        """
        Arguments
        ---------
        t : torch.tensor[1]
        
        Returns
        -------
        h : torch.tensor[N, d_h]
        """
        om = self.old_messages()
        am = self.aggregated_messages(om)
        self.update_memory(am)
        h = self.embedd(t)
        return h

    def train_loss(self, interaction_events):
        """
        Arguments
        ---------
        interaction_events : list(  ((int,int), torch.tensor[d_e], torch.tensor[1]), ... )
                             list(  ((j,i),     e_ij,              t             ), ... )
                             
        Returns
        -------
        loss : torch.tensor[1]
        """
        t = interaction_events[0][-1]
        h = self(t)
        e_ids_event = [e_id for (e_id, _, _) in interaction_events]
        e_ij_event = torch.stack(
            [e_ij for (_, e_ij, _) in interaction_events], dim=0
        )
        p_e_ij_event = self.decoder.probs(h, e_ids_event)
        loss = self.loss_fn(p_e_ij_event, e_ij_event)
        return loss

    def predict_links(self, t):
        """
        Arguments
        ---------
        t : torch.tensor[1]
        
        Returns
        -------
        pred_edge_index : torch.tensor[2, E]
            no self connection, but directed connections
            contains predicted edges of entire graph, not just new ones
        """
        h = self(t)
        pred_edge_index = self.decoder.pred_edge_index(h)
        return pred_edge_index

    def add_batch_to_graph(self, node_wise_events=[], interaction_events=[]):
        """process new nodes and edges: (add / modify / remove)
        
        Arguments
        ---------
        node_wise_events   : list(  (int,       torch.tensor[d_v], torch.tensor[1]), ... )
                             list(  (i,         v_i,               t             ), ... )
                             
        interaction_events : list(  ((int,int), torch.tensor[d_e], torch.tensor[1]), ... )
                             list(  ((j,i),     e_ij,              t             ), ... )
                             
        Returns
        -------
        -
        """
        self.raw_message_store.save_new_raw_messages(
            node_wise_events, interaction_events
        )
        for i, v_i, _ in node_wise_events:
            # remove node
            if v_i.nonzero().sum() == 0:
                self.raw_message_store.remove_node(i)
            # add node
            if i > (len(self.memory.S) - 1):
                self.memory.add_node()

        # remove edge


class RawMessageStore(nn.Module):
    """
    Attributes
    ----------
    node_wise_events   : dict(  { int :       {torch.tensor[1] : torch.tensor[d_v]}                   }  )
                         dict(  { i :         {t              : v_i              }                    }  )
                         
    interaction_events : dict(  { (int,int) : {torch.tensor[1] : (torch.tensor[d_e]}, torch.tensor[1]) }  })
                         dict(  { (j,i) :     {t               : (e_ij,               delta_t        ) }  })
                         
    interaction_events_individual :
                         dict(  { int : { int : {torch.tensor[1] : (torch.tensor[d_e]}, torch.tensor[1]) }  })
                         dict(  { i   : { j   : {t :               (e_ij,               delta_t        ) }  })    
    """

    def __init__(self, G):
        super(RawMessageStore, self).__init__()

        self.init_raw_messages(G)

    def init_raw_messages(self, graph):
        """
        Arguments
        ---------
        graph : torch_geometric.data.Data
        
        Returns
        -------
        Initialize RawMessageStore
        """
        self.node_wise_events = defaultdict(dict)
        self.interaction_events = defaultdict(dict)
        self.interaction_events_individual = defaultdict(
            lambda: defaultdict(dict)
        )

        t_0 = torch.tensor([0.0])
        node_wise_events = [(i, v_i, t_0) for (i, v_i) in enumerate(graph["x"])]
        interaction_events = list()
        for (edge_id, e_ij) in enumerate(graph["edge_attr"]):
            j = graph["edge_index"][0, edge_id].item()
            i = graph["edge_index"][1, edge_id].item()
            interaction_events.append(((j, i), e_ij, t_0))
        self.save_new_raw_messages(node_wise_events, interaction_events)

    def save_new_raw_messages(self, node_wise_events=[], interaction_events=[]):
        """
        Arguments
        ---------
        node_wise_events   : list(  (int,       torch.tensor[d_v], torch.tensor[1]), ... )
                             list(  (i,         v_i,               t             ), ... )
                             
        interaction_events : list(  ((int,int), torch.tensor[d_e], torch.tensor[1]), ... )
                             list(  ((j,i),     e_ij,              t             ), ... )
        
        Returns
        -------
        Update RawMessageStore
        """
        for i, v_i, t in node_wise_events:
            self.node_wise_events[i][t] = v_i

        for (j, i), e_ij, t in interaction_events:
            # Interaction event as edge (j,i)
            ###################################################
            if not self.interaction_events[(j, i)]:
                delta_t = torch.tensor(0.0)
            else:
                t_prev = list(self.interaction_events[(j, i)].keys())[-1]
                delta_t = t - t_prev
            self.interaction_events[(j, i)][t] = e_ij, delta_t

            # Interaction event per node, per neigbour [i][j]
            ###################################################
            if not self.interaction_events_individual[i][j]:
                delta_t = torch.tensor(0.0)
            else:
                t_prev = list(self.interaction_events_individual[i][j].keys())[
                    -1
                ]
            self.interaction_events_individual[i][j][t] = e_ij, delta_t

    def remove_node(self, i):
        self.node_wise_events.pop(i, None)
        for j, e_j_t in self.interaction_events_individual.items():
            e_j_t.pop(i, None)
        self.interaction_events_individual.pop(i, None)


class Memory(nn.Module):
    def __init__(self, N_0, d_s):
        super(Memory, self).__init__()
        self.num_nodes = {t_0: N_0}
        self.d_s = d_s
        self.mem = nn.GRU(d_h, d_h)
        self.__init_memory__(N_0, d_s)

    def __init_memory__(self, N_0, d_s):
        # self.register_buffer('S', torch.zeros((N_0, d_s)))
        self.S = nn.Parameter(torch.zeros((N_0, d_s)), requires_grad=False)

    def forward(self, S_next):
        self.S = nn.Parameter(S_next, requires_grad=False)

    def add_node(self):
        self.S = nn.Parameter(
            torch.cat(
                [self.S, nn.Parameter(torch.zeros((1, self.d_s)))], dim=0
            ),
            requires_grad=False,
        )


class MemoryUpdater(nn.Module):
    def __init__(self, d_m, d_s):
        super(MemoryUpdater, self).__init__()

        self.mem = nn.GRUCell(d_m, d_s)

    def forward(self, am, S):
        """
        Arguments
        ---------
        am : list( torch.tensor[d_m] ) 
        S : torch.tensor[N, d_s]
        
        Returns
        -------
        S_next : torch.tensor[N, d_s]
        """
        S_next = torch.ones_like(S)
        for i, am_i in am.items():
            x = am_i.unsqueeze(0)
            h = S[i].unsqueeze(0)
            S_next[i] = self.mem(x, h).squeeze(0)
        return S_next


class MessageFunction(nn.Module):
    def __init__(self, d_rm, d_m):
        super(MessageFunction, self).__init__()

        self.d_rm = d_rm
        self.d_m = d_m

        self.MLP = nn.Sequential(
            nn.Linear(d_rm, d_rm // 2), nn.ReLU(), nn.Linear(d_rm // 2, d_m)
        )

    def forward(self, raw_message):
        """
        Arguments
        ----------
        interaction event
            raw_message = [s_i_prev, s_j_prev, delta_t, e_ij]
            s_i_prev : torch.tensor [d_s]
            s_j_prev : torch.tensor [d_s]
            delta_t : torch.tensor []
            e_ij : torch.Tensor [d_e]
            
        node wise event
            raw_message = [s_i_prev, t, v_i]
            s_i : torch.tensor [d_s]
            t : torch.tensor []
            v_i : torch.Tensor [d_v]

        Returns
        -------
        interaction event
            m_i : torch.tensor [d_s + d_s + 1 + d_e]
            
        node wise event
            m_i : torch.tensor [d_s + 1 + d_v]
        """
        rm_i = torch.cat(raw_message, dim=-1)
        try:
            m_i = self.MLP(rm_i)
        except:
            print(rm_i.shape)
        return m_i


class MessageAggregator(nn.Module):
    def __init__(self):
        super(MessageAggregator, self).__init__()

    def forward(self, om):
        """
        Arguments
        ---------
        om : dict( int : {torch.tensor[1] : torch.tensor[d_m]})
             dict( i : {t : m})
             
        Returns
        -------
        m_agg : dict( int : torch.tensor[d_m] )
                dict( i : m_agg_i)
        """
        m_agg = dict()
        for i, om_i in om.items():
            m_i_agg = list()
            for t, m_i_all in om_i.items():
                for m_i in m_i_all:
                    m_i_agg.append(m_i)
            m_agg[i] = torch.stack(m_i_agg, dim=0).mean(dim=0)
        return m_agg


def get_MessageAggregator(reduction):
    if reduction == "mean":
        return MessageAggregator()


class time2vec(nn.Module):
    def __init__(self, d_t):
        super(time2vec, self).__init__()
        self.lin = nn.Linear(1, d_t)

    def forward(self, t):
        """
        Arguments
        ---------
        t : torch.Tensor [1]
        
        Returns
        -------
        t2v : torch.Tensor [d_t]
        """
        t2v = self.lin(t)
        t2v = torch.cos(t2v)
        return t2v

    def t_minus_tj(self, e, t):
        """
        Arguments
        ---------
        e : interaction_events_individual
        t : torch.tensor[1]

        Returns
        -------
        phi_t : dict( i : {j : phi(t - t_j)})
        """
        phi_t = defaultdict(dict)
        for i, e_ij in e.items():
            for j, e_ij_t in e_ij.items():
                t_j = list(e_ij_t.keys())[-1]
                phi_t[i][j] = self(t - t_j)
        return phi_t

    def t0(self):
        """
        Arguments
        ---------
        -
        
        Returns
        -------
        t0 : torch.tensor [d_t]
        """
        t0 = self(torch.tensor([0.0]))
        return t0


class SingleHeadAttention(nn.Module):
    def __init__(self, d_q, d_k, d_v, d_h):
        super(SingleHeadAttention, self).__init__()

        self.norm = torch.sqrt(torch.tensor(d_h).float())
        self.W_q = nn.Linear(d_q, d_h)
        self.W_k = nn.Linear(d_k, d_h)
        self.W_v = nn.Linear(d_v, d_h)

    def forward(self, Q, K, V):
        """
        Arguments
        ---------
        Q : torch.tensor[n_q, d_q]
        K : torch.tensor[n_k, d_k]
        V : torch.tensor[n_v, d_v]
        
        Returns
        -------
        Y : torch.tensor[n_q, d_h]
        """
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        S = (Q @ K.T) / self.norm
        A = nn.Softmax(-1)(S)
        Y = A @ V
        return Y


class MultiHeadAttention(nn.Module):
    def __init__(self, k, d_q, d_k, d_v, d_h):
        super(MultiHeadAttention, self).__init__()

        self.k = k
        self.heads = nn.ModuleList(
            [SingleHeadAttention(d_q, d_k, d_v, d_h) for _ in range(k)]
        )
        self.lin = nn.Linear(k * d_h, d_h)

    def forward(self, Q, K, V):
        """
        Arguments
        ---------
        Q : torch.tensor[n_q, d_q]
        K : torch.tensor[n_k, d_k]
        V : torch.tensor[n_v, d_v]
        
        Returns
        -------
        Y : torch.tensor[n_q, d_h]
        """
        return self.lin(torch.cat([head(Q, K, V) for head in self.heads], -1))


class TemporalGraphAttention(nn.Module):
    def __init__(self, l, k, n, d_c, d_h, d_t):
        super(TemporalGraphAttention, self).__init__()
        self.layers = nn.ModuleList(
            [TemporalGraphAttentionLayer(k, n, d_c, d_h, d_t) for _ in range(l)]
        )

    def forward(self, h, phi_t, phi_0, e):
        """
        Arguments
        ---------
        h : torch.tensor[N, d_h]
        phi_t : dict( i : {j : phi(t - t_j)})
        phi_0 : torch.tensor[d_t]
        e : interaction_events_individual
        
        Returns
        -------
        h : torch.tensor[N, d_h]
        """
        for layer in self.layers:
            h = layer(h, phi_t, phi_0, e)
        return h


class TemporalGraphAttentionLayer(nn.Module):
    def __init__(self, k, n, d_c, d_h, d_t):
        super(TemporalGraphAttentionLayer, self).__init__()
        self.attention = MultiHeadAttention(
            k=k, d_q=d_h + d_t, d_k=d_c, d_v=d_c, d_h=d_h
        )
        self.n = n
        self.d_c = d_c

    def forward(self, h, phi_t, phi_0, e):
        """
        Arguments
        ---------
        h : torch.tensor[N, d_h]
        phi_t : dict( i : {j : phi(t - t_j)})
        phi_0 : torch.tensor[d_t]
        e : interaction_events_individual
        
        Returns
        -------
        h_next : torch.tensor[N, d_h]
        """
        h_next = list()
        for i in range(len(h)):
            C_i = torch.zeros((self.n, self.d_c))
            C_list = list()
            h_i = h[i]
            e_i_t = e[i]
            for j, e_ij_t in e_i_t.items():
                t_e = list(e[i][j].keys())[-1]
                c_j = torch.cat([h[j], phi_t[i][j], e_ij_t[t_e][0]])
                C_list.append(c_j)

            if len(C_list) > 0:
                C_i[: len(C_list), :] = torch.stack(C_list, dim=0)
            Q_i = torch.cat([h_i, phi_0], dim=0)
            #             print(f'C:{C_i.shape}')
            #             print(f'Q:{Q_i.shape}')
            h_i_next = self.attention(Q_i, K=C_i, V=C_i)
            h_next.append(h_i_next)
        h_next = torch.stack(h_next, dim=0)
        return h_next


class LinkPrediction(nn.Module):
    def __init__(self):
        super(LinkPrediction, self).__init__()

    def pred_adj(self):
        raise NotImplementedError()

    def pred_edge_ids(self):
        raise NotImplementedError()


class LinkBinaryClassification(LinkPrediction):
    def __init__(self):
        super(LinkBinaryClassification, self).__init__()

    def prob_adj(self, h):
        """
        Arguments
        ---------
        h : torch.tensor[N, d_h]
        
        Returns
        -------
        p_e_ij_adj : torch.tensor[N, N]
        """
        p_e_ij_adj = nn.Sigmoid()(h @ h.T)
        p_e_ij_adj -= torch.diag_embed(p_e_ij_adj.diag())
        return p_e_ij_adj

    def prob_edge_index(self, h, edge_ids):
        """
        Arguments
        ---------
        h : torch.tensor[N, d_h]
        edge_ids : list( (j,i), ... )
        
        Returns
        -------
        p_e_ij : torch.tensor[E, 1]
        """
        p_e_ij = torch.stack(
            [nn.Sigmoid()(h[j] @ h[i].T) for j, i in edge_ids], dim=0
        )
        return p_e_ij

    def pred_edge_index(self, h):
        """
        Arguments
        ---------
        h : torch.tensor[N, d_h]
        edge_ids : list( (j,i), ... )
        
        Returns
        -------
        pred_edge_index : torch.tensor[2, E]
        """
        p_e_ij = self.prob_adj(h)
        e_ij_pred_adj = torch.round(p_e_ij).detach().numpy()
        pred_edge_index = torch.from_numpy(np.argwhere(e_ij_pred_adj)).T
        return pred_edge_index


class LinkCategoricalClassification(LinkPrediction):
    def __init__(self):
        super(LinkCategoricalClassification, self).__init__()

    def pred_adj(self, h):
        NotImplemented
        # TODO

    def pred_edge_ids(self, h, edge_ids):
        NotImplemented
        # TODO


class LinkRegression(LinkPrediction):
    def __init__(self):
        super(LinkRegression, self).__init__()

    def pred_adj(self, h):
        NotImplemented
        # TODO

    def pred_edge_ids(self, h, edge_ids):
        NotImplemented
        # TODO
