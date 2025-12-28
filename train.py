# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np
import time
import os

from model import Seq2Seq, Encoder, Decoder, Attention
from utils import build_vocab, TranslationDataset, collate_fn

# 固定随机种子
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# 路径配置
DATA_DIR = 'dataset-master/data/task1/tok'
TRAIN_SRC = os.path.join(DATA_DIR, 'train.lc.norm.tok.en')
TRAIN_TGT = os.path.join(DATA_DIR, 'train.lc.norm.tok.de')
VAL_SRC = os.path.join(DATA_DIR, 'val.lc.norm.tok.en')
VAL_TGT = os.path.join(DATA_DIR, 'val.lc.norm.tok.de')

# 超参数
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
CLIP = 1.0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, iterator, optimizer, criterion, clip=1, src_vocab=None, tgt_vocab=None):
    model.train()
    epoch_loss = 0
    for i, (src, trg) in enumerate(tqdm(iterator, desc="Training", leave=False)):
        src = src.to(DEVICE)
        trg = trg.to(DEVICE)

        # === 调试：打印第一个 batch 的第一对样本 ===
        if i == 0 and src_vocab is not None and tgt_vocab is not None:
            # 获取反向映射：id -> word
            if '_itos' in src_vocab:
                src_itos = src_vocab['_itos']  # list: index = id
                tgt_itos = tgt_vocab['_itos']
            else:
                # 兼容旧版：构建 dict
                src_itos = {v: k for k, v in src_vocab.items() if isinstance(k, str)}
                tgt_itos = {v: k for k, v in tgt_vocab.items() if isinstance(k, str)}

            src_ids = src[:, 0].cpu().tolist()
            trg_ids = trg[:, 0].cpu().tolist()

            # 解码 source
            if isinstance(src_itos, list):
                src_words = [src_itos[idx] for idx in src_ids if idx != 0]
            else:
                src_words = [src_itos.get(idx, '<unk>') for idx in src_ids if idx != 0]

            # 解码 target（跳过 <sos>=1, <eos>=2, <pad>=0）
            trg_words = []
            for idx in trg_ids:
                if idx in [0, 1, 2]:
                    continue
                if isinstance(tgt_itos, list):
                    trg_words.append(tgt_itos[idx])
                else:
                    trg_words.append(tgt_itos.get(idx, '<unk>'))

            print("\n" + "="*60)
            print("[DEBUG] First training sample in first batch:")
            print("Source (EN):", " ".join(src_words))
            print("Target (DE):", " ".join(trg_words))
            print("="*60)

        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, trg in tqdm(iterator, desc="Validation", leave=False):
            src = src.to(DEVICE)
            trg = trg.to(DEVICE)
            output = model(src, trg, 0)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    print("正在构建词表...")
    src_vocab = build_vocab(TRAIN_SRC)
    tgt_vocab = build_vocab(TRAIN_TGT)
    print(f"Source vocab size: {len(src_vocab)}")
    print(f"Target vocab size: {len(tgt_vocab)}")

    print("正在加载数据集...")
    train_dataset = TranslationDataset(TRAIN_SRC, TRAIN_TGT, src_vocab, tgt_vocab)
    val_dataset = TranslationDataset(VAL_SRC, VAL_TGT, src_vocab, tgt_vocab)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    INPUT_DIM = len(src_vocab)
    OUTPUT_DIM = len(tgt_vocab)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    attn = Attention(HID_DIM, HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, attn)
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)

    print(f"Using device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"Model is on GPU: {next(model.parameters()).is_cuda}")

    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    PAD_IDX = 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    best_valid_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, CLIP, src_vocab, tgt_vocab)
        valid_loss = evaluate(model, val_loader, criterion)
        end_time = time.time()

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_model.pt')

        print(f'Epoch: {epoch:02} | Train Loss: {train_loss:.3f} | Val Loss: {valid_loss:.3f}')

    print("Training finished. Model saved as 'best_model.pt'")