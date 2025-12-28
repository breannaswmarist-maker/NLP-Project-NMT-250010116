import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
import logging
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from data_utils import load_and_process_data, TextProcessor, get_dataloader, PAD_IDX
from transformer_model import TransformerNMT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# ==========================================
# 配置参数 (支持 Ablation)
# ==========================================
CONFIG = {
    'train_path': '/mnt/afs/250010116/nlp/AP0004_Midterm&Final_translation_dataset_zh_en/train_10k.jsonl',
    'val_path': '/mnt/afs/250010116/nlp/AP0004_Midterm&Final_translation_dataset_zh_en/valid.jsonl',
    'batch_size': 64,  # Transformer 可以尝试更大的 batch
    'lr': 0.0001,
    'n_epochs': 20,
    
    # === 模型架构消融参数 ===
    'emb_dim': 256,
    'n_heads': 4,
    'n_layers': 3,
    'ffn_dim': 512,
    'norm_type': 'layernorm', # 'layernorm' or 'rmsnorm'
    'pos_type': 'absolute',   # 'absolute' or 'learnable'
    # ========================
    
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

def train(model, iterator, optimizer, criterion, scheduler):
    model.train()
    epoch_loss = 0
    
    for i, (src, trg) in enumerate(iterator):
        src, trg = src.to(CONFIG['device']), trg.to(CONFIG['device'])
        
        # Transformer 的输入通常是 target 的前 t-1 个词，预测 t 个词
        # trg_input: <sos> ... <last_word>
        # trg_label: <first_word> ... <eos>
        trg_input = trg[:, :-1]
        trg_label = trg[:, 1:]
        
        optimizer.zero_grad()
        
        output = model(src, trg_input)
        # output: [batch, trg_len-1, output_dim]
        
        output_dim = output.shape[-1]
        output = output.reshape(-1, output_dim)
        trg_label = trg_label.reshape(-1)
        
        loss = criterion(output, trg_label)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # Step Scheduler
        if scheduler is not None:
            scheduler.step()
            
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src, trg = src.to(CONFIG['device']), trg.to(CONFIG['device'])
            trg_input = trg[:, :-1]
            trg_label = trg[:, 1:]
            
            output = model(src, trg_input)
            output_dim = output.shape[-1]
            output = output.reshape(-1, output_dim)
            trg_label = trg_label.reshape(-1)
            
            loss = criterion(output, trg_label)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def calculate_bleu_transformer(model, iterator, en_vocab):
    # 简化的 Greedy Decoding 用于 BLEU 计算
    model.eval()
    targets, outputs = [], []
    
    with torch.no_grad():
        for src, trg in iterator:
            src = src.to(CONFIG['device'])
            batch_size = src.shape[0]
            max_len = trg.shape[1] + 5
            
            # Start token
            sos_idx = en_vocab.word2idx['<sos>']
            eos_idx = en_vocab.word2idx['<eos>']
            
            # 初始化 decoder input: [batch, 1] filled with <sos>
            trg_input = torch.tensor([[sos_idx]], device=CONFIG['device']).repeat(batch_size, 1)
            
            finished = torch.zeros(batch_size, dtype=torch.bool).to(CONFIG['device'])
            
            # Autoregressive generation
            for _ in range(max_len):
                pred = model(src, trg_input) # [batch, len, vocab]
                next_token = pred[:, -1, :].argmax(dim=-1).unsqueeze(1) # [batch, 1]
                
                trg_input = torch.cat([trg_input, next_token], dim=1)
                
                # 标记已经生成 <eos> 的句子
                is_eos = (next_token.squeeze(1) == eos_idx)
                finished |= is_eos
                if finished.all():
                    break
            
            # 处理结果用于 BLEU
            preds = trg_input[:, 1:] # remove <sos>
            for i in range(batch_size):
                # Target
                trg_list = [en_vocab.idx2word[idx.item()] for idx in trg[i] if idx.item() not in [0, 2, 3]]
                targets.append([trg_list])
                
                # Pred
                pred_list = []
                for idx in preds[i]:
                    idx_val = idx.item()
                    if idx_val == eos_idx: break
                    if idx_val not in [0, 2, 3]:
                        pred_list.append(en_vocab.idx2word[idx_val])
                outputs.append(pred_list)
                
    score = corpus_bleu(targets, outputs, smoothing_function=SmoothingFunction().method1)
    return score * 100

if __name__ == "__main__":
    processor = TextProcessor()
    train_data, zh_vocab, en_vocab = load_and_process_data(CONFIG['train_path'], processor, is_train=True)
    val_data, _, _ = load_and_process_data(CONFIG['val_path'], processor, zh_vocab=zh_vocab, en_vocab=en_vocab, is_train=False)
    
    train_loader = get_dataloader(train_data, zh_vocab, en_vocab, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = get_dataloader(val_data, zh_vocab, en_vocab, batch_size=CONFIG['batch_size'], shuffle=False)
    
    src_vocab_size = len(zh_vocab.word2idx)
    trg_vocab_size = len(en_vocab.word2idx)
    
    logging.info(f"Building Transformer. Norm: {CONFIG['norm_type']}, Pos: {CONFIG['pos_type']}")
    
    model = TransformerNMT(
        src_vocab_size, trg_vocab_size, 
        src_pad_idx=PAD_IDX, trg_pad_idx=PAD_IDX,
        emb_dim=CONFIG['emb_dim'], 
        n_heads=CONFIG['n_heads'], 
        n_layers=CONFIG['n_layers'],
        ffn_dim=CONFIG['ffn_dim'],
        norm_type=CONFIG['norm_type'],
        pos_type=CONFIG['pos_type']
    ).to(CONFIG['device'])
    
    optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    # 简单的 Scheduler，避免 NoamOpt 的复杂性，使用 ReduceLROnPlateau 或 StepLR
    # 这里为了作业的超参敏感性实验，也可以设为 None
    scheduler = None 

    logging.info(f'Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    
    best_loss = float('inf')
    
    for epoch in range(CONFIG['n_epochs']):
        start_time = time.time()
        
        train_loss = train(model, train_loader, optimizer, criterion, scheduler)
        valid_loss = evaluate(model, val_loader, criterion)
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'best_transformer.pt')
            
        # BLEU 计算比较耗时，可以每 5 个 epoch 算一次
        bleu = 0.0
        if (epoch + 1) % 5 == 0:
            bleu = calculate_bleu_transformer(model, val_loader, en_vocab)
            
        print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val Loss: {valid_loss:.3f} | BLEU: {bleu:.2f}')