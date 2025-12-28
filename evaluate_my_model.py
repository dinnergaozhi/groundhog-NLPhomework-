import torch
import re
import pickle
from tqdm import tqdm
import os
from model import Seq2Seq, Encoder, Decoder, Attention
from utils import TranslationDataset, collate_fn
import sacrebleu

# === 路径配置 ===
DATA_DIR = 'dataset-master/data/task1/tok'
VAL_SRC = os.path.join(DATA_DIR, 'val.lc.norm.tok.en')
VAL_TGT = os.path.join(DATA_DIR, 'val.lc.norm.tok.de')
MODEL_PATH = 'best_model.pt'

# === 超参数（必须与 train.py 完全一致）===
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def tokenize_like_reference(text):
    """将模型输出转为与 reference 相同的 tokenized 格式"""
    text = text.lower()
    text = re.sub(r"([.,!?;:])", r" \1 ", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def ids_to_words(ids, itos_list):
    """将 ID 列表转换为字符串（跳过 <pad>=0, <sos>=1, <eos>=2）"""
    words = []
    for idx in ids:
        if idx in [0, 1, 2]:
            continue
        word = itos_list[idx]  # itos_list 是 list，index = id
        words.append(word)
    return ' '.join(words)

def main():
    print("正在加载训练时保存的词表...")
    with open('src_vocab.pkl', 'rb') as f:
        src_vocab = pickle.load(f)
    with open('tgt_vocab.pkl', 'rb') as f:
        tgt_vocab = pickle.load(f)

    # 从 vocab 中提取 _itos 列表（你的 utils.py 保证了这一点）
    src_itos = src_vocab['_itos']  # list: index == word_id
    tgt_itos = tgt_vocab['_itos']

    print(f"Source vocab size: {len(src_vocab)}")
    print(f"Target vocab size: {len(tgt_vocab)}")

    # === 构建模型（使用训练词表大小！）===
    attn = Attention(HID_DIM, HID_DIM)
    enc = Encoder(len(src_vocab), ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(len(tgt_vocab), DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, attn)
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)

    # === 加载训练好的权重 ===
    print(f"正在加载模型权重: {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # === 读取验证集原始文本（用于 BLEU）===
    with open(VAL_SRC, 'r', encoding='utf-8') as f:
        val_src_lines_raw = [line.strip() for line in f.readlines()]
    with open(VAL_TGT, 'r', encoding='utf-8') as f:
        val_tgt_lines_raw = [line.strip() for line in f.readlines()]

    # === 创建 Dataset 和 DataLoader（使用训练词表进行 numericalize）===
    val_dataset = TranslationDataset(VAL_SRC, VAL_TGT, src_vocab, tgt_vocab)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
    )

    hypotheses = []
    print(f"正在翻译 {len(val_src_lines_raw)} 个句子...")
    with torch.no_grad():
        for i, (src, _) in enumerate(tqdm(val_loader)):
            src = src.to(DEVICE)  # [src_len, 1]

            # 初始化 target 输入为 <sos> (ID=1)
            trg_indexes = [1]

            # Greedy decoding
            for _ in range(100):  # 最大长度限制
                trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(1).to(DEVICE)  # [trg_len, 1]
                output = model(src, trg_tensor, 0)  # teacher_forcing_ratio=0
                pred_token = output.argmax(2)[-1].item()
                trg_indexes.append(pred_token)
                if pred_token == 2:  # <eos>
                    break

            # 将 ID 转回 words
            hyp_text = ids_to_words(trg_indexes, tgt_itos)
            hyp_text = tokenize_like_reference(hyp_text)
            hypotheses.append(hyp_text)

    # === 计算 BLEU（与 evaluate_hf.py 完全一致）===
    bleu = sacrebleu.corpus_bleu(
        hypotheses,
        [val_tgt_lines_raw],  # reference 是原始 tokenized 行
        tokenize='none',
        force=True  # 忽略 sacrebleu 的 detokenize 警告
    )

    print("\n" + "=" * 50)
    print(f"Your trained Seq2Seq model BLEU 分数: {bleu.score:.2f}")
    print("=" * 50)

    # 保存翻译结果（可选）
    with open('my_model_hypotheses.txt', 'w', encoding='utf-8') as f:
        for hyp in hypotheses:
            f.write(hyp + '\n')

if __name__ == '__main__':
    main()