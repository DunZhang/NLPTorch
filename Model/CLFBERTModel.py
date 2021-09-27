import torch
from transformers import BertModel, BertTokenizer
from NLPTrainPlatform.Model import AModel
from NLPTrainPlatform.Config.ModelConfig import CLFBERTConfig


class CLFBERTModel(AModel):
    def __init__(self, conf: CLFBERTConfig):
        super().__init__()
        self.conf = conf
        self.bert = BertModel.from_pretrained(conf.pretrained_model_dir)
        self.clf = torch.nn.Linear(in_features=self.bert.config.hidden_size, out_features=conf.num_labels)
        self.pooling_mode = conf.pooling_mode

    def _get_mean_embed(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        sen_vec = sum_embeddings / sum_mask
        return sen_vec

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, *args, **kwargs):
        token_embeddings, pooler_output, hidden_states = self.bert(input_ids=input_ids,
                                                                   attention_mask=attention_mask,
                                                                   token_type_ids=token_type_ids,
                                                                   output_hidden_states=True)[0:3]
        pooling_mode = self.pooling_mode
        if pooling_mode == "cls":
            sen_vec = pooler_output
        elif pooling_mode == "mean":
            # get mean token sen vec
            sen_vec = self._get_mean_embed(token_embeddings, attention_mask)
        elif pooling_mode == 'first_last_mean':
            sen_vec = (self._get_mean_embed(hidden_states[-1], attention_mask) + self._get_mean_embed(hidden_states[1],
                                                                                                      attention_mask)) / 2
        elif pooling_mode == 'last2mean':
            sen_vec = (self._get_mean_embed(hidden_states[-1], attention_mask) + self._get_mean_embed(hidden_states[-2],
                                                                                                      attention_mask)) / 2
        elif pooling_mode == "max":
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            sen_vec = torch.max(token_embeddings, 1)[0]

        logits = self.clf(sen_vec)  # bsz * num_labels
        return logits
