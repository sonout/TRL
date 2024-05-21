import math
import copy


import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Optional, Tuple


class Transformer(nn.Module):

    def __init__(self, ninput, max_seq_len=200, ff_dim=2048, nhead=4, nlayer=2, attn_dropout=0, pos_droput=0.1):
        super().__init__()

        self.n_head = nhead

        # Adds positional information to source/target token's embedding vector
        # (otherwise we'd lose the positional information which is important in human languages)
        self.src_pos_embedding = PositionalEncoding(ninput, pos_droput)
        
        # some useful precompute for the RoPE relative positional embeddings
        freqs_cos, freqs_sin = precompute_freqs_cis(ninput // nhead, max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # All of these will get deep-copied multiple times internally
        mha = MHA(ninput, nhead, attn_dropout)
        pwn = PositionwiseFeedForwardNet(ninput, attn_dropout, ff_dim)
        encoder_layer = EncoderLayer(ninput, attn_dropout, mha, pwn)
        self.encoder = Encoder(encoder_layer, nlayer)
        self.lstm_agg = nn.LSTM(
            input_size=ninput,
            hidden_size=ninput,
            num_layers=1,
            batch_first=True,
        )
        self.init_params()

    def init_params(self, default_initialization=False):
        # Not mentioned in the paper, but other implementations used xavier.
        # I tested both PyTorch's default initialization and this, and xavier has tremendous impact! I didn't expect
        # a model's perf, with normalization layers, to be so much dependent on the choice of weight initialization.
        if not default_initialization:
            for name, p in self.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, x, lengths):
        # x = x.transpose(0, 1) # before it was (B, T, D) now it's (T, B, D) -> I think we do not need it...
        _bsz, seqlen, _emb_dim = x.shape
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        max_src_len = torch.arange(lengths.max().item()).to(lengths.get_device()) # in essense -- trajs1_len[0]
        src_padding_mask = max_src_len[None, :] < lengths[:, None]
        src_padding_mask = src_padding_mask.view(_bsz, 1, 1, seqlen).expand(-1, self.n_head, -1, -1)
        

        x = self.src_pos_embedding(x)
        rtn = self.encoder(x, src_padding_mask, freqs_cos, freqs_sin)
        # Maybe Norm afterwards?

        # mask = max_src_len[None, :] >= lengths[:, None]
        # mask = 1 - mask.unsqueeze(-1).expand(rtn.shape).float()
        # rtn = torch.sum(mask * rtn, 1) # Sum over all seq_len
        # rtn = rtn / lengths.unsqueeze(-1).expand(rtn.shape)

        ## LSTM 
        self.lstm_agg.flatten_parameters()
        outputs, (hs, cs) = self.lstm_agg(rtn) # Outputs shape: same as rtn: [batch_size, max_seq_len, emb_size]
        rtn = outputs[torch.arange(rtn.shape[0]), lengths-1]

        return rtn



# For Ablation only
class TransformerA(nn.Module):

    def __init__(self, ninput, max_seq_len=200, ff_dim=2048, nhead=4, nlayer=2, attn_dropout=0, pos_droput=0.1):
        super().__init__()

        self.n_head = nhead

        # Adds positional information to source/target token's embedding vector
        # (otherwise we'd lose the positional information which is important in human languages)
        self.src_pos_embedding = PositionalEncoding(ninput, pos_droput)
        
        # some useful precompute for the RoPE relative positional embeddings
        freqs_cos, freqs_sin = precompute_freqs_cis(ninput // nhead, max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # All of these will get deep-copied multiple times internally
        mha = MHA(ninput, nhead, attn_dropout)
        pwn = PositionwiseFeedForwardNet(ninput, attn_dropout, ff_dim)
        encoder_layer = EncoderLayer(ninput, attn_dropout, mha, pwn)
        self.encoder = Encoder(encoder_layer, nlayer)
        self.lstm_agg = nn.LSTM(
            input_size=ninput,
            hidden_size=ninput,
            num_layers=1,
            batch_first=True,
        )
        self.init_params()

    def init_params(self, default_initialization=False):
        # Not mentioned in the paper, but other implementations used xavier.
        # I tested both PyTorch's default initialization and this, and xavier has tremendous impact! I didn't expect
        # a model's perf, with normalization layers, to be so much dependent on the choice of weight initialization.
        if not default_initialization:
            for name, p in self.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, x, lengths):
        # x = x.transpose(0, 1) # before it was (B, T, D) now it's (T, B, D) -> I think we do not need it...
        _bsz, seqlen, _emb_dim = x.shape
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        max_src_len = torch.arange(lengths.max().item()).to(lengths.get_device()) # in essense -- trajs1_len[0]
        src_padding_mask = max_src_len[None, :] < lengths[:, None]
        src_padding_mask = src_padding_mask.view(_bsz, 1, 1, seqlen).expand(-1, self.n_head, -1, -1)
        

        x = self.src_pos_embedding(x)
        rtn = self.encoder(x, src_padding_mask, freqs_cos, freqs_sin)
        # Maybe Norm afterwards?

        mask = max_src_len[None, :] >= lengths[:, None]
        mask = 1 - mask.unsqueeze(-1).expand(rtn.shape).float()
        rtn = torch.sum(mask * rtn, 1) # Sum over all seq_len
        rtn = rtn / lengths.unsqueeze(-1).expand(rtn.shape)

        ## LSTM 
        # self.lstm_agg.flatten_parameters()
        # outputs, (hs, cs) = self.lstm_agg(rtn) # Outputs shape: same as rtn: [batch_size, max_seq_len, emb_size]
        # rtn = outputs[torch.arange(rtn.shape[0]), lengths-1]

        return rtn




#
# Encoder architecture
#


class Encoder(nn.Module):

    def __init__(self, encoder_layer, number_of_layers):
        super().__init__()
        assert isinstance(encoder_layer, EncoderLayer), f'Expected EncoderLayer got {type(encoder_layer)}.'

        self.encoder_layers = get_clones(encoder_layer, number_of_layers)
        self.norm = nn.LayerNorm(encoder_layer.model_dimension)

    def forward(self, src_embeddings_batch, src_mask, freqs_cos = None, freqs_sin = None):
        # Just update the naming so as to reflect the semantics of what this var will become (the initial encoder layer
        # has embedding vectors as input but later layers have richer token representations)
        src_representations_batch = src_embeddings_batch

        # Forward pass through the encoder stack
        for encoder_layer in self.encoder_layers:
            # src_mask's role is to mask/ignore padded token representations in the multi-headed self-attention module
            src_representations_batch = encoder_layer(src_representations_batch, src_mask, freqs_cos, freqs_sin)

        # Not mentioned explicitly in the paper (a consequence of using LayerNorm before instead of after the sublayer
        # check out the SublayerLogic module)
        return self.norm(src_representations_batch)


class EncoderLayer(nn.Module):

    def __init__(self, model_dimension, dropout_probability, multi_headed_attention, pointwise_net):
        super().__init__()
        
        self.multi_headed_attention = multi_headed_attention
        self.pointwise_net = pointwise_net

        self.model_dimension = model_dimension

        self.norm = nn.LayerNorm(model_dimension)
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, x, mask, freqs_cos = None, freqs_sin = None):
        # Define anonymous (lambda) function which only takes src_representations_batch (srb) as input,
        # this way we have a uniform interface for the sublayer logic.
        _, seq_len, _ = x.shape
        attention = lambda x: self.multi_headed_attention(x, x, x, mask, freqs_cos, freqs_sin)

        x = x + self.dropout(attention(self.norm(x)))
        x = x + self.dropout(self.pointwise_net(self.norm(x)))
        return x




#
# Helper modules (designed with modularity in mind) and organized top to bottom.
#



class PositionwiseFeedForwardNet(nn.Module):
    """
        It's position-wise because this feed forward net will be independently applied to every token's representation.

        Representations batch is of the shape (batch size, max token sequence length, model dimension).
        This net will basically be applied independently to every token's representation (you can think of it as if
        there was a nested for-loop going over the batch size and max token sequence length dimensions
        and applied this net to token representations. PyTorch does this auto-magically behind the scenes.

    """
    def __init__(self, model_dimension, dropout_probability, ff_dim=None):
        super().__init__()

        if ff_dim is None:
            ff_dim = 4 * model_dimension

        self.linear1 = nn.Linear(model_dimension, ff_dim)
        self.linear2 = nn.Linear(ff_dim, model_dimension)

        # This dropout layer is not explicitly mentioned in the paper but it's common to use to avoid over-fitting
        self.dropout = nn.Dropout(p=dropout_probability)
        self.relu = nn.ReLU()

    def forward(self, representations_batch):
        return self.linear2(self.dropout(self.relu(self.linear1(representations_batch))))




class MHA(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        assert dim % n_heads == 0, "model dimension must be divisible by number of heads"

        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.dropout = dropout

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        attn_mask: None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,        
    ):
        bsz, seqlen, _ = xq.shape

        # QKV
        xq, xk, xv = self.wq(xq), self.wk(xk), self.wv(xv)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        # RoPE relative positional embeddings
        if freqs_cos is not None and freqs_sin is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # flash implementation
        output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=False)


        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output



#
# Input modules
#



class PositionalEncoding(nn.Module):

    def __init__(self, model_dimension, dropout_probability, expected_max_sequence_length=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_probability)

        # (stated in the paper) Use sine functions whose frequencies form a geometric progression as position encodings,
        # (learning encodings will also work so feel free to change it!). Page 6, Chapter 3.5 "Positional Encoding"
        position_id = torch.arange(0, expected_max_sequence_length).unsqueeze(1)
        frequencies = torch.pow(10000., -torch.arange(0, model_dimension, 2, dtype=torch.float) / model_dimension)

        # Checkout playground.py for visualization of how these look like (it's super simple don't get scared)
        positional_encodings_table = torch.zeros(expected_max_sequence_length, model_dimension)
        positional_encodings_table[:, 0::2] = torch.sin(position_id * frequencies)  # sine on even positions
        positional_encodings_table[:, 1::2] = torch.cos(position_id * frequencies)  # cosine on odd positions

        # Register buffer because we want to save the positional encodings table inside state_dict even though
        # these are not trainable (not model's parameters) so they otherwise would be excluded from the state_dict
        self.register_buffer('positional_encodings_table', positional_encodings_table)

    def forward(self, embeddings_batch):
        assert embeddings_batch.ndim == 3 and embeddings_batch.shape[-1] == self.positional_encodings_table.shape[1], \
            f'Expected (batch size, max token sequence length, model dimension = {self.positional_encodings_table.shape[1]}) got {embeddings_batch.shape}'

        # embedding_batch's shape = (B, S/T, D), where S/T max src/trg token-sequence length, D - model dimension
        # So here we get (S/T, D) shape which will get broad-casted to (B, S/T, D) when we try and add it to embeddings
        positional_encodings = self.positional_encodings_table[:embeddings_batch.shape[1]]

        # (stated in the paper) Applying dropout to the sum of positional encodings and token embeddings
        # Page 7, Chapter 5.4 "Regularization"
        return self.dropout(embeddings_batch + positional_encodings)


#
# Helper model functions
#
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f"{freqs_cis.shape} != {(x.shape[1], x.shape[-1])}"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )



def get_clones(module, num_of_deep_copies):
    # Create deep copies so that we can tweak each module's weights independently
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_of_deep_copies)])


# Count how many trainable weights the model has <- just for having a feeling for how big the model is
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def analyze_state_dict_shapes_and_names(model):
    # This part helped me figure out that I don't have positional encodings saved in the state dict
    print(model.state_dict().keys())

    # This part helped me see that src MHA was missing in the decoder since both it and trg MHA were referencing
    # the same MHA object in memory - stupid mistake, happens all the time, embrace the suck!
    for name, param in model.named_parameters():
        print(name, param.shape)
        if not param.requires_grad:
            raise Exception('Expected all of the params to be trainable - no param freezing used.')