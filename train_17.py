import os
import gc
import random
import time
import json
import math
import warnings
import argparse
warnings.filterwarnings("ignore")

import scipy as sp
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score

from transformers import AutoTokenizer, RobertaModel, BertModel
from transformers import AdamW

import pickle
import argparse

from catr.models import caption
from catr.datasets import coco, utils
from catr.configuration import Config

# from model.MabsaModel import *
from model.MabsaModel import *

from utils.helpers import *
from utils.data_utils import *
from utils.dataset import *

class WassersteinLoss(nn.Module):
    def __init__(self, eps=0.1, max_iter=5):
        super().__init__()
        self.eps = eps
        self.max_iter = max_iter

    def forward(self, x, y):
        # x: [BS, T_len, Dim], y: [BS, R_len, Dim]
        # 计算余弦距离矩阵 C: [BS, T_len, R_len]
        x_norm = F.normalize(x, dim=-1)
        y_norm = F.normalize(y, dim=-1)
        C = 1 - torch.matmul(x_norm, y_norm.transpose(-1, -2))
        
        bs, n, m = C.shape
        mu = torch.ones(bs, n, device=C.device) / n
        nu = torch.ones(bs, m, device=C.device) / m
        
        K = torch.exp(-C / self.eps)
        u = torch.ones_like(mu)
        
        # Sinkhorn 迭代实现可微的最优传输
        for _ in range(self.max_iter):
            v = nu / (torch.matmul(K.transpose(-1, -2), u.unsqueeze(-1)).squeeze(-1) + 1e-8)
            u = mu / (torch.matmul(K, v.unsqueeze(-1)).squeeze(-1) + 1e-8)
        
        T = u.unsqueeze(-1) * K * v.unsqueeze(-2)
        wd_dist = torch.sum(T * C, dim=(-1, -2)) 
        return wd_dist.mean()

def info_nce_loss(text_feat, visual_feat, temperature=0.1):
    """
    计算文本特征和视觉特征之间的对比损失
    text_feat: [BS, hidden_size]
    visual_feat: [BS, hidden_size]
    """
    # 1. 归一化特征
    text_feat =  text_feat.mean(dim=1)
    visual_feat =  visual_feat.mean(dim=1)
    text_feat = F.normalize(text_feat, dim=-1)
    visual_feat = F.normalize(visual_feat, dim=-1)

    # 2. 计算相似度矩阵 [BS, BS]
    logits = torch.matmul(text_feat, visual_feat.T) / temperature
    
    # 3. 生成标签（对角线为正样本）
    batch_size = text_feat.shape[0]
    labels = torch.arange(batch_size).to(text_feat.device)
    
    # 4. 计算对称交叉熵
    loss_t2v = F.cross_entropy(logits, labels)
    loss_v2t = F.cross_entropy(logits.T, labels)
    
    return (loss_t2v + loss_v2t) / 2

# Config
parser = argparse.ArgumentParser()
#att & box
parser.add_argument('--region_box', type=str, default="../region_box/17_twitter/_box")
parser.add_argument('--region_att', type=str, default="../region_box/17_twitter/_att")


parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--data_dir', type=str, default="../absa_data")
parser.add_argument('--catr_ckpt', type=str, default="../catr/checkpoint.pth")
parser.add_argument('--result_dir', type=str, default="result")
parser.add_argument('--log_dir', type=str, default="logs")
parser.add_argument('--dataset', type=str, default="twitter2017") # twitter2015 or twitter2017
parser.add_argument('--model', type=str, default="bertweet-base") # "bert-base-uncased" "vinai/bertweet-base"

parser.add_argument('--num_cycles', type=float, default=0.5)
parser.add_argument('--num_warmup_steps', type=int, default=0)
parser.add_argument('--adamw_correct_bias', type=bool, default=True)
parser.add_argument('--scheduler', type=str, default='linear') # ['linear', 'cosine']
parser.add_argument('--print_freq', type=int, default=30)
parser.add_argument('--num_workers', type=int, default=4)

parser.add_argument('--max_grad_norm', type=int, default=100)
parser.add_argument('--seed', type=int, default=123)

parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--lr_catr', type=float, default=2e-5)
parser.add_argument('--lr_backbone', type=float, default=5e-6)
parser.add_argument('--max_caption_len', type=int, default=12)
parser.add_argument('--max_len', type=int, default=128)
parser.add_argument('--sample_k', type=int, default=10)
parser.add_argument('--epochs', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--wd_beta', type=float, default=0.05)
parser.add_argument('--kl', type=int, default=2)

args = parser.parse_args()

device = torch.device(args.device)
LOGGER = get_logger(os.path.join(args.log_dir, args.dataset+f"_{args.seed}"))
train_df, val_df,  test_df = load_data(args)
wd_loss_fn = WassersteinLoss(eps=0.1, max_iter=5).to(device)
# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)
args.tokenizer = tokenizer
args.pad_token_id = args.tokenizer.pad_token_id
args.end_token_id = args.tokenizer.sep_token_id

## Train fn
def train_fn(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples, epoch):
    model = model.train()
    losses = []
    correct_predictions = 0
    start = end = time.time()
    for step, d in enumerate(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        image = d["image"]
        image_mask = d["image_mask"]
        caption = d["caption"].to(device)
        cap_mask = d["caption_mask"].to(device)
        labels = d["labels"].to(device)
        input_ids_tt = d["input_ids_tt"].to(device)
        attention_mask_tt = d["attention_mask_tt"].to(device)
        input_ids_at = d["input_ids_at"].to(device)
        attention_mask_at = d["attention_mask_at"].to(device)
        samples = utils.NestedTensor(image, image_mask).to(device)
        region_feat = d["region_feat"].to(device) # 形状 (BS, 36, 2048)
        region_box = d["region_box"].to(device)   # 形状 (BS, 36, 4)
        if not isinstance(samples, utils.NestedTensor):
            raise TypeError("Unsupported type:", type(samples))
        outputs,outputs_2,text_at,visual_at = model(samples, caption, cap_mask,region_feat,region_box,
                input_ids, attention_mask, loss_fn,
                input_ids_tt, attention_mask_tt, input_ids_at, attention_mask_at)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)
        # log_probs = F.log_softmax(outputs, dim=1)       # 对 outputs 应用 log_softmax
        # target_probs = F.softmax(outputs_2, dim=1)
        
        # 2. 计算 CE Loss (对两个输出都做监督)
        loss_ce1 = loss_fn(outputs, labels)
        loss_ce2 = loss_fn(outputs_2, labels)
        ce_loss = 0.5 * (loss_ce1 + loss_ce2)

        # 3. 计算 KL 散度 (R-Drop 核心)
        # KL 散度是非对称的，标准做法是计算双向 KL 并取平均
        p_loss = F.kl_div(F.log_softmax(outputs, dim=-1), F.softmax(outputs_2, dim=-1), reduction='batchmean')
        q_loss = F.kl_div(F.log_softmax(outputs_2, dim=-1), F.softmax(outputs, dim=-1), reduction='batchmean')
        kl_loss = (p_loss + q_loss) / 2

        # 4. 总损失
        # total_loss = ce_loss + 2 * kl_loss + 0.001 * cl_loss
        wd_loss = wd_loss_fn(text_at, visual_at)
        # 4. 总损失
        # total_loss = ce_loss + 5 * kl_loss + 0.1 * cl_loss + 0.05 * wd_loss
        total_loss = ce_loss + args.kl * kl_loss  + args.wd_beta * wd_loss
        
        correct_predictions += torch.sum(preds == labels).item()
        losses.append(total_loss.item())
        total_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm) # clip grad
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        end = time.time()
        if step % args.print_freq == 0 or step == (len(data_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                'Elapsed {remain:s} '
                        'Loss: {loss:.4f} '
                        'Grad: {grad_norm:.4f}  '
                        'LR: {lr:.8f}  '
                        .format(epoch+1, step, len(data_loader),
                                remain=timeSince(start, float(step+1)/len(data_loader)),
                                loss=loss.item(),
                                grad_norm=grad_norm,
                                lr=scheduler.get_lr()[0]))

    return correct_predictions / n_examples, np.mean(losses)

# eval fn
def format_eval_output(rows):
    tweets, targets, labels, predictions = zip(*rows)
    tweets = np.vstack(tweets)
    targets = np.vstack(targets)
    labels = np.vstack(labels)
    predictions = np.vstack(predictions)
    results_df = pd.DataFrame()
    results_df["tweet"] = tweets.reshape(-1).tolist()
    results_df["target"] = targets.reshape(-1).tolist()
    results_df["label"] = labels
    results_df["prediction"] = predictions
    return results_df


def eval_model(model, data_loader, loss_fn, device, n_examples, detailed_results=False):
    model = model.eval()

    losses = []
    correct_predictions = 0
    rows = []

    with torch.no_grad():
      for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        image = d["image"]
        image_mask = d["image_mask"]
        caption = d["caption"].to(device)
        cap_mask = d["caption_mask"].to(device)
        labels = d["labels"].to(device)
        samples = utils.NestedTensor(image, image_mask).to(device)
        input_ids_tt = d["input_ids_tt"].to(device)
        attention_mask_tt = d["attention_mask_tt"].to(device)
        input_ids_at = d["input_ids_at"].to(device)
        attention_mask_at = d["attention_mask_at"].to(device)
        region_feat = d["region_feat"].to(device) # 形状 (BS, 36, 2048)
        region_box = d["region_box"].to(device)   # 形状 (BS, 36, 4)
        outputs,outputs_2,text_at,visual_at = model(samples, caption, cap_mask,region_feat,region_box,
                input_ids, attention_mask, loss_fn,
                input_ids_tt, attention_mask_tt, input_ids_at, attention_mask_at)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)
       
        loss_ce1 = loss_fn(outputs, labels)
        loss_ce2 = loss_fn(outputs_2, labels)
        ce_loss = 0.5 * (loss_ce1 + loss_ce2)

        p_loss = F.kl_div(F.log_softmax(outputs, dim=-1), F.softmax(outputs_2, dim=-1), reduction='batchmean')
        q_loss = F.kl_div(F.log_softmax(outputs_2, dim=-1), F.softmax(outputs, dim=-1), reduction='batchmean')
        kl_loss = (p_loss + q_loss) / 2

        wd_loss = wd_loss_fn(text_at, visual_at)
   
        total_loss = ce_loss + args.kl * kl_loss  + args.wd_beta * wd_loss
        correct_predictions += torch.sum(preds == labels).item()
        losses.append(total_loss.item())
        rows.extend(
          zip(d["review_text"],
            d["targets"],
            d["labels"].numpy(),
            preds.cpu().numpy(),
          )
        )

      if detailed_results:
          return (correct_predictions / n_examples,
                np.mean(losses),
                format_eval_output(rows),
            )

    return correct_predictions / n_examples, np.mean(losses)

def train_loop():
    LOGGER.info(f"========== Start Training ==========")
    # Create DataLoader
    image_captions = None
    train_dataset = TwitterDataset(args, train_df, image_captions, coco.val_transform, utils.nested_tensor_from_tensor_list)
    test_dataset = TwitterDataset(args, test_df, image_captions, coco.val_transform, utils.nested_tensor_from_tensor_list)
    val_dataset = TwitterDataset(args, val_df, image_captions, coco.val_transform, utils.nested_tensor_from_tensor_list)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                num_workers=args.num_workers, pin_memory=True, drop_last=False)
    # Model
    config = Config()
    config.vocab_size = 30522
    config.pad_token_id = args.pad_token_id
    catr, _ = caption.build_model(config)
    checkpoint = torch.load(args.catr_ckpt)
    catr.to(device)
    catr.load_state_dict(checkpoint['model'])
    model = CustomModel(args, catr)
    model.to(device)
    # Configure the optimizer and scheduler.
    param_dicts = [{"params": [p for n, p in model.named_parameters() if "catr" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
        "lr": args.lr_backbone,},
        {"params": [p for n, p in model.named_parameters() if "catr" in n and 'backbone' not in n and p.requires_grad],
        "lr": args.lr_catr,},
    ]
    optimizer = AdamW(param_dicts, lr=args.lr, correct_bias=args.adamw_correct_bias)

    num_train_steps = int(len(train_df)/args.batch_size*args.epochs)
    scheduler = get_scheduler(args, optimizer, num_train_steps)
    loss_fn = nn.CrossEntropyLoss().to(device)

    for epoch in range(args.epochs):
        print(f"===============Epoch {epoch + 1}/{args.epochs}==============")
        start_time = time.time()

        train_acc, train_loss = train_fn(
        model, train_loader, loss_fn, optimizer, device, scheduler, len(train_df), epoch
        )

        test_acc, test_loss, dr = eval_model(model, test_loader, loss_fn, device, len(test_df), detailed_results=True)
        macro_f1_test = f1_score(dr.label, dr.prediction, average="macro")
        elapsed = time.time() - start_time
        LOGGER.info(f'Epoch {epoch+1} - Train loss {train_loss} accuracy {train_acc}" time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Test loss {test_loss} accuracy {test_acc} macro f1 {macro_f1_test}'
        )
    

    LOGGER.info(f"TEST ACC = {test_acc}\nMACRO F1 = {macro_f1_test}")
    return dr

if __name__ == "__main__":
    seed_everything(seed = args.seed)
    x = train_loop()




