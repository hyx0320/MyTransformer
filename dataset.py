import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import os

# 特殊标记
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'

class TranslationDataset(Dataset):
    """机器翻译数据集"""
    def __init__(self, src_path, tgt_path, src_vocab, tgt_vocab, max_len=5000):
        """
        src_path: 源语言文件路径
        tgt_path: 目标语言文件路径
        src_vocab: 源语言词表 (word -> idx)
        tgt_vocab: 目标语言词表 (word -> idx)
        max_len: 句子最大长度（截断用）
        """
        self.src_sentences = self._read_file(src_path)
        self.tgt_sentences = self._read_file(tgt_path)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

        # 过滤掉过长的句子
        filtered = [(s, t) for s, t in zip(self.src_sentences, self.tgt_sentences)
                    if len(s) <= max_len - 2 and len(t) <= max_len - 2]  # 留出<sos>和<eos>的位置
        self.src_sentences, self.tgt_sentences = zip(*filtered) if filtered else ([], [])

    def _read_file(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip().split() for line in f]

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src = self.src_sentences[idx]
        tgt = self.tgt_sentences[idx]

        # 单词转索引，加上 <sos> 和 <eos>
        src_ids = [self.src_vocab.get(SOS_TOKEN, 1)] + \
                  [self.src_vocab.get(w, self.src_vocab[UNK_TOKEN]) for w in src] + \
                  [self.src_vocab.get(EOS_TOKEN, 2)]
        tgt_ids = [self.tgt_vocab.get(SOS_TOKEN, 1)] + \
                  [self.tgt_vocab.get(w, self.tgt_vocab[UNK_TOKEN]) for w in tgt] + \
                  [self.tgt_vocab.get(EOS_TOKEN, 2)]

        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def build_vocab(file_path, max_size=8000):
    """从文件中构建词表，保留最常见的 max_size 个词"""
    counter = Counter()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            counter.update(line.strip().split())

    # 特殊标记优先
    vocab = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}
    for word, _ in counter.most_common(max_size - 4):
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def collate_fn(batch, pad_idx):
    """自定义 batch 处理：动态填充"""
    src_batch, tgt_batch = zip(*batch)
    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    return src_padded, tgt_padded


def get_data_loader(config, src_path, tgt_path, src_vocab, tgt_vocab, shuffle=True):
    """返回 DataLoader"""
    dataset = TranslationDataset(src_path, tgt_path, src_vocab, tgt_vocab, max_len=config.max_len)
    loader = DataLoader(dataset,
                        batch_size=config.batch_size,
                        shuffle=shuffle,
                        collate_fn=lambda b: collate_fn(b, config.pad_idx))
    return loader