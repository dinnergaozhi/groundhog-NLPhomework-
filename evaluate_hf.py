import torch
import re
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer
import sacrebleu

# === 路径配置 ===
DATA_DIR = 'D:/Desktop/GroundHog-master/dataset-master/data/task1/tok'
VAL_SRC = f'{DATA_DIR}/val.lc.norm.tok.en'
VAL_TGT = f'{DATA_DIR}/val.lc.norm.tok.de'


def tokenize_like_reference(text):
    """将 HF 输出转为与 reference 相同的 tokenized 格式"""
    text = text.lower()
    text = re.sub(r"([.,!?;:])", r" \1 ", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def main():
    print("正在加载 Hugging Face 翻译模型...")
    model_name = "Helsinki-NLP/opus-mt-en-de"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name, use_safetensors=True)

    device = 0 if torch.cuda.is_available() else -1
    if device >= 0:
        model = model.to(f'cuda:{device}')
    print(f"Device set to use cuda:{device}" if device >= 0 else "Using CPU")

    # === 读取验证集 ===
    with open(VAL_SRC, 'r', encoding='utf-8') as f:
        val_src_lines = [line.strip() for line in f.readlines()]
    with open(VAL_TGT, 'r', encoding='utf-8') as f:
        val_tgt_lines = [line.strip() for line in f.readlines()]  # 这就是 reference！

    hypotheses = []
    batch_size = 16

    print(f"正在翻译 {len(val_src_lines)} 个句子...")
    for i in tqdm(range(0, len(val_src_lines), batch_size)):
        batch = val_src_lines[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        if device >= 0:
            inputs = {k: v.to(f'cuda:{device}') for k, v in inputs.items()}

        with torch.no_grad():
            translated = model.generate(**inputs)
        decoded = tokenizer.batch_decode(translated, skip_special_tokens=True)

        for hyp in decoded:
            hyp = tokenize_like_reference(hyp)  # 关键：格式对齐
            hypotheses.append(hyp)

    # === 计算 BLEU ===
    bleu = sacrebleu.corpus_bleu(
        hypotheses,
        [val_tgt_lines],  # 注意：这里用 val_tgt_lines
        tokenize='none'
    )

    print("\n" + "=" * 50)
    print(f"Hugging Face (opus-mt-en-de) BLEU 分数: {bleu.score:.2f}")
    print("=" * 50)


if __name__ == '__main__':
    main()