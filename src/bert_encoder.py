import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class BertTextEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased', freeze=False):
        """
        Task 1/3: BERT Text Encoder for contextual language representations.
        """
        super(BertTextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        
        # Freeze BERT parameters if required (useful for early stages of Task 3/4)
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
                
        self.hidden_size = self.bert.config.hidden_size

    def forward(self, input_ids, attention_mask):
        """
        Forward pass for BERT.
        Returns:
            last_hidden_state: (batch_size, seq_len, hidden_size) - used for cross-attention
            cls_token: (batch_size, hidden_size) - used for global representation
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        last_hidden_state = outputs.last_hidden_state
        cls_token = outputs.pooler_output
        
        return last_hidden_state, cls_token
