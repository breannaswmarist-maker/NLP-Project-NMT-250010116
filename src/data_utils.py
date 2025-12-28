import json
import re
import jieba
import logging
from collections import Counter
from typing import List, Tuple, Dict

# === 新增的 Import ===
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 定义特殊 Token
PAD_TOKEN = '<pad>'  #用于填充
UNK_TOKEN = '<unk>'  #用于未知词
SOS_TOKEN = '<sos>'  #句子开始
EOS_TOKEN = '<eos>'  #句子结束

class TextProcessor:
    def __init__(self, use_bpe=False):
        self.use_bpe = use_bpe
        # 这里可以初始化 BPE tokenizer (如 HuggingFace tokenizers)，如果需要进阶探索
        if not use_bpe:
            logging.info("Initializing basic tokenizer: Jieba for Chinese, Space/Rule for English.")
    
    def clean_text(self, text: str) -> str:
    # 仅保留中英文、数字、标点符号，过滤掉其他非法符号
        text = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fa5\.,\?!\'\" ]+", "", text) 
        text = re.sub(r'\s+', ' ', text) # 规范化空格
        return text.strip()

    def tokenize_cn(self, text: str) -> List[str]:
        """
        对应要求：Jieba for Chinese
        """
        return list(jieba.cut(text))

    def tokenize_en(self, text: str) -> List[str]:
        """
        对应要求：Space-based / NLTK for English
        这里实现一个简单的规则分词，将标点分开。
        """
        # 简单的正则分词，将单词和标点分开
        return re.findall(r"[\w]+|[^\s\w]", text)

    def process_pair(self, zh_text: str, en_text: str, max_len=100) -> Tuple[List[str], List[str]]:
        """
        处理单对句子：清洗 -> 分词 -> 长度过滤
        对应要求：filter or truncate excessively long sentences
        """
        zh_clean = self.clean_text(zh_text)
        en_clean = self.clean_text(en_text)

        zh_tokens = self.tokenize_cn(zh_clean)
        en_tokens = self.tokenize_en(en_clean.lower()) # 英文通常转小写

        # 简单长度过滤策略：如果任一句子过长，返回 None (后续过滤掉)
        # 或者你也可以选择 truncate (截断)
        if len(zh_tokens) > max_len or len(en_tokens) > max_len:
            return None
        
        return zh_tokens, en_tokens

class Vocabulary:
    def __init__(self, name, freq_threshold=2):
        self.name = name
        self.word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1, SOS_TOKEN: 2, EOS_TOKEN: 3}
        self.idx2word = {0: PAD_TOKEN, 1: UNK_TOKEN, 2: SOS_TOKEN, 3: EOS_TOKEN}
        self.word_counts = Counter()
        self.freq_threshold = freq_threshold # 对应要求：filter out low-frequency words
        self.frozen = False

    def add_sentence(self, tokens: List[str]):
        if self.frozen: return
        self.word_counts.update(tokens)

    def build_vocab(self):
        """
        构建最终词表，过滤低频词
        """
        logging.info(f"Building vocabulary for {self.name}...")
        # 按照频率排序
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        
        for word, freq in sorted_words:
            if freq >= self.freq_threshold:
                if word not in self.word2idx:
                    idx = len(self.word2idx)
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
        
        self.frozen = True
        logging.info(f"Vocabulary {self.name} built. Size: {len(self.word2idx)}. (Ignored words freq < {self.freq_threshold})")

    def numericalize(self, tokens: List[str]) -> List[int]:
        """
        将 token 列表转换为索引列表
        """
        return [self.word2idx.get(token, self.word2idx[UNK_TOKEN]) for token in tokens]

def load_and_process_data(filepath, processor, zh_vocab=None, en_vocab=None, is_train=True):
    """
    加载 JSONL 数据并处理
    如果是训练集，会构建词表；如果是验证/测试集，则使用已有词表。
    """
    pairs = []
    
    # 如果是训练阶段且未提供词表，则初始化新词表
    if is_train and (zh_vocab is None or en_vocab is None):
        zh_vocab = Vocabulary("Chinese")
        en_vocab = Vocabulary("English")

    logging.info(f"Processing file: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                # 假设 JSONL 的键名为 'zh' 和 'en'，请根据实际数据调整 key
                # 如果数据没有 key 只有 values，请相应调整
                zh_text = data.get('zh', '') or data.get('chinese', '')
                en_text = data.get('en', '') or data.get('english', '')
                
                if not zh_text or not en_text: continue

                processed = processor.process_pair(zh_text, en_text)
                if processed:
                    zh_tokens, en_tokens = processed
                    pairs.append((zh_tokens, en_tokens))
                    
                    if is_train:
                        zh_vocab.add_sentence(zh_tokens)
                        en_vocab.add_sentence(en_tokens)
            except Exception as e:
                continue

    if is_train:
        zh_vocab.build_vocab()
        en_vocab.build_vocab()
        
    logging.info(f"Loaded {len(pairs)} sentence pairs from {filepath}")
    return pairs, zh_vocab, en_vocab

# ... (上面是 load_and_process_data 函数)

# ==========================================
# Step 2: Dataset & DataLoader Logic
# ==========================================

# 假设 PAD_TOKEN 对应的 index 是 0 (对应 Vocabulary 类中的定义)
PAD_IDX = 0 

class TranslationDataset(Dataset):
    def __init__(self, pairs, zh_vocab, en_vocab):
        self.pairs = pairs
        self.zh_vocab = zh_vocab
        self.en_vocab = en_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        zh_tokens, en_tokens = self.pairs[idx]
        
        # 将 token 转换为 index
        # 添加 <sos> 和 <eos>
        zh_indices = [self.zh_vocab.word2idx['<sos>']] + \
                     self.zh_vocab.numericalize(zh_tokens) + \
                     [self.zh_vocab.word2idx['<eos>']]
                     
        en_indices = [self.en_vocab.word2idx['<sos>']] + \
                     self.en_vocab.numericalize(en_tokens) + \
                     [self.en_vocab.word2idx['<eos>']]
                     
        return torch.tensor(zh_indices, dtype=torch.long), torch.tensor(en_indices, dtype=torch.long)

def collate_fn(batch):
    zh_batch, en_batch = [], []
    for zh_item, en_item in batch:
        zh_batch.append(zh_item)
        en_batch.append(en_item)
    
    # 填充操作
    zh_batch = pad_sequence(zh_batch, batch_first=True, padding_value=PAD_IDX)
    en_batch = pad_sequence(en_batch, batch_first=True, padding_value=PAD_IDX)
    
    return zh_batch, en_batch

def get_dataloader(pairs, zh_vocab, en_vocab, batch_size=32, shuffle=True):
    dataset = TranslationDataset(pairs, zh_vocab, en_vocab)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_fn
    )
    return dataloader
# ==========================================
# Main Execution (Test All Steps)
# ==========================================
if __name__ == "__main__":
    processor = TextProcessor()
    
    # 1. 加载数据 (请确保路径正确)
    train_file = '/mnt/afs/250010116/nlp/AP0004_Midterm&Final_translation_dataset_zh_en/train_10k.jsonl'
    val_file = '/mnt/afs/250010116/nlp/AP0004_Midterm&Final_translation_dataset_zh_en/valid.jsonl'
    
    train_data, zh_vocab, en_vocab = load_and_process_data(train_file, processor, is_train=True)
    val_data, _, _ = load_and_process_data(val_file, processor, zh_vocab=zh_vocab, en_vocab=en_vocab, is_train=False)
    
    # 2. 测试 DataLoader
    print("-" * 30)
    print("Initializing DataLoader...")
    
    # Batch Size 设为 4 方便观察
    train_loader = get_dataloader(train_data, zh_vocab, en_vocab, batch_size=4)
    
    for i, (src, trg) in enumerate(train_loader):
        print(f"Batch {i} Info:")
        print(f"  Source Shape: {src.shape}") # 期望: [4, max_len]
        print(f"  Target Shape: {trg.shape}")
        print("  Source Tensor Sample (First row):", src[0])
        break # 只打印第一个 batch 验证即可