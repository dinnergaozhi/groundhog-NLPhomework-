from collections import Counter
import torch
from torch.nn.utils.rnn import pad_sequence

# 特殊 token 必须按此顺序，确保 <pad>=0
SPECIALS = ['<pad>', '<sos>', '<eos>', '<unk>']


def build_vocab(file_path, min_freq=1):
    """
    构建词表，保证 SPECIALS 在最前面，且 <pad>=0。
    所有未登录词将映射到 <unk>。
    """
    counter = Counter()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            counter.update(tokens)

    # 过滤低频词，但保留 SPECIALS
    words = [w for w, c in counter.items() if c >= min_freq]

    # 合并：先 specials，再普通词
    vocab_list = SPECIALS + words

    # 构建 word → index 映射
    vocab = {word: idx for idx, word in enumerate(vocab_list)}

    # 额外：保存反向映射（用于 debug）
    vocab['_itos'] = vocab_list

    return vocab


class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, src_file, tgt_file, src_vocab, tgt_vocab):
        with open(src_file, 'r', encoding='utf-8') as f:
            self.src_lines = f.readlines()
        with open(tgt_file, 'r', encoding='utf-8') as f:
            self.tgt_lines = f.readlines()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src_toks = self.src_lines[idx].strip().split()
        tgt_toks = self.tgt_lines[idx].strip().split()

        # 源语言：OOV → <unk>
        src_ids = [
            self.src_vocab.get(w, self.src_vocab['<unk>'])
            for w in src_toks
        ]

        # 目标语言：加 <sos> 和 <eos>
        tgt_ids = (
                [self.tgt_vocab['<sos>']] +
                [self.tgt_vocab.get(w, self.tgt_vocab['<unk>']) for w in tgt_toks] +
                [self.tgt_vocab['<eos>']]
        )

        return torch.tensor(src_ids), torch.tensor(tgt_ids)


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    # padding_value=0 对应 <pad>
    src_batch = pad_sequence(src_batch, padding_value=0, batch_first=False)
    tgt_batch = pad_sequence(tgt_batch, padding_value=0, batch_first=False)
    return src_batch, tgt_batch