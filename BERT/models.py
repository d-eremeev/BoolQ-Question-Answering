import torch.nn as nn
import torch


class PretrainedBERT(nn.Module):
    """
    - Wrapper for pretrained BERT model.
      https://huggingface.co/transformers/model_doc/bert.html#bertmodel
    - forward extracts embeddings for question/passage pairs.
    """

    def __init__(self,
                 pretrained_bert,
                 use_pooling):

        super().__init__()
        self.pretrained_bert = pretrained_bert
        self.use_pooling = use_pooling

    def forward(self,
                tokens_q,
                tokens_p,
                att_mask_q,
                att_mask_p):

        # (last_hidden_state, pooler_output) named tuples:
        # ([batch_size, seq_len, hid_dim], [batch_size, hid_dim])
        output_q = self.pretrained_bert(tokens_q, att_mask_q)
        output_p = self.pretrained_bert(tokens_p, att_mask_p)

        if self.use_pooling:
            # use output of HF BERT pooling
            pooled_q = output_q[1]  # [batch_size, hid_dim]
            pooled_p = output_p[1]
        else:
            # extract state of <CLS> token.
            pooled_q = output_q[0][:, 0, :]
            pooled_p = output_p[0][:, 0, :]

        # concatenate question, passage embeddings
        pair_embedding = torch.cat([pooled_q, pooled_p], dim=-1)

        return pair_embedding