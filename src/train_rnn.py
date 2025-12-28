import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import time
import logging
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# 引入之前的模块
from data_utils import load_and_process_data, TextProcessor, get_dataloader, PAD_IDX
from rnn_model import Encoder, Decoder, Attention, Seq2Seq

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# ==========================================
# 1. 配置参数 (Hyperparameters)
# ==========================================
CONFIG = {
    'train_path': '/mnt/afs/250010116/nlp/AP0004_Midterm&Final_translation_dataset_zh_en/train_10k.jsonl',
    'val_path': '/mnt/afs/250010116/nlp/AP0004_Midterm&Final_translation_dataset_zh_en/valid.jsonl',
    # 预训练向量路径 (如果没有，设为 None)
    'pretrained_emb_path': None, # e.g., '/path/to/glove.6B.100d.txt'
    
    # 模型参数
    'input_dim': None, # 稍后从词表获取
    'output_dim': None,
    'enc_emb_dim': 256,
    'dec_emb_dim': 256,
    'hid_dim': 512,
    'n_layers': 2,
    'enc_dropout': 0.5,
    'dec_dropout': 0.5,
    'attn_method': 'dot', # 'dot', 'general', 'concat'
    'rnn_type': 'GRU',    # 'GRU', 'LSTM'
    
    # 训练参数
    'batch_size': 128,
    'learning_rate': 0.001, # Adam 默认
    'n_epochs': 15,
    'clip': 1.0, # 梯度裁剪阈值
    'teacher_forcing_ratio': 0.5, # 1.0=完全Teacher Forcing, 0.0=Free Running
    
    # 设备
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

# ==========================================
# 2. 工具函数 (Helper Functions)
# ==========================================

def init_weights(m):
    """
    初始化权重，打破对称性
    """
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

def load_pretrained_embeddings(model, vocab, emb_file, is_encoder=True):
    """
    加载预训练词向量 (符合 Word Embedding Initialization 要求)
    """
    if emb_file is None:
        return
    
    logging.info(f"Loading pretrained embeddings for {'Encoder' if is_encoder else 'Decoder'} from {emb_file}...")
    emb_dim = model.encoder.embedding.embedding_dim if is_encoder else model.decoder.embedding.embedding_dim
    
    # 构建词向量矩阵
    hits = 0
    # 这里是一个简单的 Glove 加载示例，具体格式视文件而定
    # 假设文件格式: word val1 val2 ...
    with open(emb_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            if word in vocab.word2idx:
                vec = torch.tensor([float(x) for x in parts[1:]])
                if vec.shape[0] == emb_dim:
                    idx = vocab.word2idx[word]
                    if is_encoder:
                        model.encoder.embedding.weight.data[idx] = vec
                    else:
                        model.decoder.embedding.weight.data[idx] = vec
                    hits += 1
    
    logging.info(f"Loaded {hits} vectors. Fine-tuning enabled.")

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# ==========================================
# 3. 训练与评估循环 (Train & Eval Loops)
# ==========================================

def train(model, iterator, optimizer, criterion, clip, tf_ratio):
    model.train()
    epoch_loss = 0
    
    for i, (src, trg) in enumerate(iterator):
        src, trg = src.to(CONFIG['device']), trg.to(CONFIG['device'])
        
        optimizer.zero_grad()
        
        # forward pass
        # output: [batch size, trg len, output dim]
        output = model(src, trg, teacher_forcing_ratio=tf_ratio)
        
        # 计算 Loss 时需要忽略 <sos>，并展平维度
        # trg: [batch size, trg len] -> 剔除第一列 <sos>
        # output: [batch size, trg len, vocab] -> 剔除最后一列 (因为长度对应) 
        # 但通常 PyTorch 处理 seq2seq loss 的标准写法是：
        output_dim = output.shape[-1]
        
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        # 梯度裁剪 (防止梯度爆炸)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src, trg = src.to(CONFIG['device']), trg.to(CONFIG['device'])

            # 验证时不使用 teacher forcing (ratio = 0)
            output = model(src, trg, teacher_forcing_ratio=0) 

            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()
            
    return epoch_loss / len(iterator)

def calculate_bleu(model, iterator, en_vocab):
    """
    计算 BLEU Score (使用 NLTK)
    这对应作业 Metric 要求     """
    model.eval()
    targets = []
    outputs = []
    
    with torch.no_grad():
        for src, trg in iterator:
            src = src.to(CONFIG['device'])
            trg = trg.to(CONFIG['device'])
            
            # 使用 Greedy Decoding 生成翻译
            # 注意: 如果要对比 Beam Search，这里可以调用 model.beam_search (但速度较慢，通常仅在 Test 阶段做)
            # 这里为了训练速度，使用 batch greedy
            output = model(src, trg, teacher_forcing_ratio=0) 
            # output: [batch, len, vocab]
            
            # 获取预测的 token id
            preds = output.argmax(2) # [batch, len]
            
            # 转换为单词列表
            for i in range(src.shape[0]):
                # Target (Reference)
                trg_indices = trg[i].cpu().numpy()
                # 过滤特殊 token (PAD, SOS, EOS)
                trg_tokens = [en_vocab.idx2word[idx] for idx in trg_indices if idx not in [0, 2, 3]]
                targets.append([trg_tokens]) # corpus_bleu 需要 list of list of refs
                
                # Prediction (Hypothesis)
                pred_indices = preds[i].cpu().numpy()
                pred_tokens = []
                for idx in pred_indices:
                    if idx == 3: # EOS
                        break
                    if idx not in [0, 2, 3]:
                        pred_tokens.append(en_vocab.idx2word[idx])
                outputs.append(pred_tokens)
                
    # 计算 BLEU-4
    score = corpus_bleu(targets, outputs, smoothing_function=SmoothingFunction().method1)
    return score * 100

# ==========================================
# 4. 主程序 (Main Execution)
# ==========================================
if __name__ == "__main__":
    # A. 数据准备
    processor = TextProcessor() # 这里未来可以扩展 BPE
    
    # 1. 加载并处理数据
    train_data, zh_vocab, en_vocab = load_and_process_data(CONFIG['train_path'], processor, is_train=True)
    val_data, _, _ = load_and_process_data(CONFIG['val_path'], processor, zh_vocab=zh_vocab, en_vocab=en_vocab, is_train=False)
    
    # 2. 构建 DataLoader
    train_loader = get_dataloader(train_data, zh_vocab, en_vocab, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = get_dataloader(val_data, zh_vocab, en_vocab, batch_size=CONFIG['batch_size'], shuffle=False)
    
    CONFIG['input_dim'] = len(zh_vocab.word2idx)
    CONFIG['output_dim'] = len(en_vocab.word2idx)
    
    logging.info(f"Data Loaded. Vocab: ZH={CONFIG['input_dim']}, EN={CONFIG['output_dim']}")

    # B. 模型构建
    attn = Attention(CONFIG['hid_dim'], method=CONFIG['attn_method'])
    enc = Encoder(CONFIG['input_dim'], CONFIG['enc_emb_dim'], CONFIG['hid_dim'], CONFIG['n_layers'], CONFIG['enc_dropout'], rnn_type=CONFIG['rnn_type'])
    dec = Decoder(CONFIG['output_dim'], CONFIG['dec_emb_dim'], CONFIG['hid_dim'], CONFIG['n_layers'], CONFIG['dec_dropout'], attn, rnn_type=CONFIG['rnn_type'])
    
    model = Seq2Seq(enc, dec, CONFIG['device']).to(CONFIG['device'])
    
    # 初始化权重
    model.apply(init_weights)
    
    # C. 加载预训练向量 (如果配置了路径)
    # 这满足了 "Word Embedding Initialization" 的要求
    if CONFIG['pretrained_emb_path']:
        load_pretrained_embeddings(model, zh_vocab, CONFIG['pretrained_emb_path'], is_encoder=True)
        # 如果有英文向量，也可以加载给 Decoder
        
    logging.info(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

    # D. 优化器与损失函数
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # E. 训练循环
    best_valid_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_bleu': []}
    
    logging.info("Start Training...")
    
    for epoch in range(CONFIG['n_epochs']):
        start_time = time.time()
        
        train_loss = train(model, train_loader, optimizer, criterion, CONFIG['clip'], CONFIG['teacher_forcing_ratio'])
        valid_loss = evaluate(model, val_loader, criterion)
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        # 保存最佳模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_rnn_model.pt')
        
        # 计算 BLEU (可选：每隔几个 epoch 算一次以节省时间)
        bleu_score = calculate_bleu(model, val_loader, en_vocab)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(valid_loss)
        history['val_bleu'].append(bleu_score)
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f} | BLEU: {bleu_score:.2f}')

    # F. 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss Curve')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_bleu'], label='BLEU Score', color='orange')
    plt.legend()
    plt.title('Validation BLEU')
    
    plt.savefig('training_curves.png')
    logging.info("Training finished. Best model saved as 'best_rnn_model.pt'. Plots saved.")