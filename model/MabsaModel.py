
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, BertModel,DebertaV2Model
from model.attention import DotProductAttention


# Construct the joint Caption & Sentiment Classifier Model
class CustomModel(nn.Module):
    def __init__(self, cfg, catr):
        super().__init__()
        self.cfg = cfg
        self.catr = catr
        self.bert = DebertaV2Model.from_pretrained(self.cfg.model)
        self.dropout = nn.Dropout(self.cfg.dropout)
        self.out = nn.Linear(self.bert.config.hidden_size, 3) # 3 classes
        self.att = DotProductAttention(self.cfg.dropout)
        self.mul_att = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True, dropout=self.cfg.dropout)
        self.project = nn.Linear(self.bert.config.hidden_size*2, self.bert.config.hidden_size)
        self._init_weights(self.out)
        self._init_weights(self.project)

        self.hidden_size = self.bert.config.hidden_size

        self.visual_project = nn.Linear(2048,self.hidden_size)

        self.pos_project = nn.Linear(4,self.hidden_size)
        self.region_norm = nn.LayerNorm(self.hidden_size)
        self._init_weights(self.visual_project)
        self._init_weights(self.pos_project)

        self.gate_layer = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.gate_norm = nn.LayerNorm(self.hidden_size)
        self._init_weights(self.gate_layer)


    def _init_weights(self, module):
        # 稳健地获取初始化标准差，DeBERTa 和 RoBERTa 默认通常都是 0.02
        initializer_range = getattr(self.bert.config, "initializer_range", 0.02)
        
        if isinstance(module, nn.Linear):
            # 使用 BERT 系模型的标准正态分布初始化
            module.weight.data.normal_(mean=0.0, std=initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        
        # 修复逻辑：LayerNorm 应该是一个独立的判断分支
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
        elif isinstance(module, nn.AdaptiveAvgPool1d):
            # 注意：AdaptiveAvgPool1d 通常没有 weight 属性，除非你自定义了带权重的池化
            # 如果是标准 nn.AdaptiveAvgPool1d，这行跳过即可
            if hasattr(module, 'weight') and module.weight is not None:
                torch.nn.init.xavier_normal_(module.weight.data)


    def forward(self,samples,caption,c_mask,region_feat,region_box,input_ids,attention_mask,ce_loss,input_ids_tt,attention_mask_tt,input_ids_at,attention_mask_at,is_training=True):
        sample_k = self.cfg.sample_k
        cap_len = self.cfg.max_caption_len
        pad_token_id = self.cfg.pad_token_id
        end_token_id = self.cfg.end_token_id
        bs = caption.shape[0]
        device = caption.device

        origin_len = attention_mask.sum(dim=-1)
        
        embeddings = self.bert.get_input_embeddings()
        end_token_embedding = embeddings(torch.tensor(end_token_id, device=device))

        _, caption_out, _, cap_mask, finished = self.catr(samples,caption,c_mask,cap_len,sample_k,end_token_id,pad_token_id)
        sorted_out_id = torch.argsort(caption_out, dim=-1, descending=True)  # bs*tgt_len*vocab_size
        sample_out_id = sorted_out_id[:, :, :sample_k]
        sample_out = torch.zeros((bs, cap_len, sample_k),dtype=torch.float,device=device)
        for i in range(bs):
            for j in range(cap_len):
                sample_out[i][j] = caption_out[i, j, sample_out_id[i][j]]
        sample_prob = F.softmax(sample_out, dim=-1)
        sample_prob = sample_prob.unsqueeze(3)
        sample_caption_embedding = self.bert.embeddings.word_embeddings(sample_out_id)
        sample_caption_embedding = (sample_prob*sample_caption_embedding).sum(dim=2) # element-wise product

    
        v_feat = self.visual_project(region_feat) 
        p_feat = self.pos_project(region_box)
   
        region_embeds = self.region_norm(v_feat + p_feat) # [BS, 36, 768]

        combined_visual_ctx = torch.cat([sample_caption_embedding, region_embeds], dim=1)
        
       

        # Calculate input embeddings
        for i in range(bs):
            if finished[i]:
                cap_mask[i, finished[i]] = True
        inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)
        inputs_embeds_tt = self.bert.embeddings.word_embeddings(input_ids_tt)
        inputs_embeds_at = self.bert.embeddings.word_embeddings(input_ids_at)
        target_len = attention_mask_at.sum(dim=-1)
        cap_mask = ~cap_mask  # cap_mask(valid: False; else:True) | attention_mask(valid:1; else:0)
        cap_mask = cap_mask.int()
        caption_len = cap_mask.sum(dim=-1)
        
        att_output_at, _ = self.att(inputs_embeds_at, inputs_embeds_tt, inputs_embeds_tt, attention_mask_tt.sum(dim=-1))
       
        new_valid_lens = caption_len + 36
        att_output_ai, attn_weights = self.att(
            inputs_embeds_at,           # Query: Aspect
            combined_visual_ctx,        # Key
            combined_visual_ctx,        # Value
            new_valid_lens                        # 视情况添加 mask
        )

        global_saliency = torch.mean(attn_weights.float(), dim=1, keepdim=True)

        contrastive_weights = torch.relu(attn_weights.float() - global_saliency)

        sum_weights = contrastive_weights.sum(dim=-1, keepdim=True) + 1e-8
        contrastive_weights = contrastive_weights / sum_weights

        refined_att_output = torch.matmul(contrastive_weights, combined_visual_ctx)

        alpha = 0.4
        att_output_ai = alpha * refined_att_output + (1 - alpha) * att_output_ai
        gate_input = torch.cat([inputs_embeds_at,att_output_ai],dim = -1)

        gate = torch.sigmoid(self.gate_layer(gate_input))

        att_output_ai = self.gate_norm(inputs_embeds_at+gate*att_output_ai)

        # concatenation
        for i in range(bs):
            attention_mask[i, origin_len[i]: origin_len[i]+target_len[i]] = 1
            inputs_embeds[i, origin_len[i]: origin_len[i]+target_len[i]] = att_output_ai[i, :target_len[i]]
            inputs_embeds[i, origin_len[i]+target_len[i]] = end_token_embedding

        outputs_1 = self.bert(inputs_embeds=inputs_embeds,attention_mask=attention_mask)
        last_hidden_state_1 = outputs_1.last_hidden_state
    
        f_combined_1 = last_hidden_state_1[:,0,:]
        outputs_1 = self.out(self.dropout(f_combined_1))

        outputs_2 = self.bert(inputs_embeds=inputs_embeds,attention_mask=attention_mask)
        last_hidden_state_2 = outputs_2.last_hidden_state
        f_combined_2 = last_hidden_state_2[:,0,:]
        outputs_2 = self.out(self.dropout(f_combined_2))
        return outputs_1,outputs_2,att_output_at,att_output_ai
