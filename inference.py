import torch
import argparse
import logging
import sys
import os

# 引入你的项目模块
from data_utils import load_and_process_data, TextProcessor, PAD_IDX
from rnn_model import Encoder, Decoder, Attention, Seq2Seq
from transformer_model import TransformerNMT

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 默认配置 (请确保这里与你训练时的参数一致)
# ==========================================
RNN_CONFIG = {
    'enc_emb_dim': 256, 'dec_emb_dim': 256, 'hid_dim': 512, 
    'n_layers': 2, 'dropout': 0.5, 'rnn_type': 'GRU', 'attn_method': 'dot'
}

TRANSFORMER_CONFIG = {
    'emb_dim': 256, 'n_heads': 4, 'n_layers': 3, 'ffn_dim': 512,
    'norm_type': 'layernorm', 'pos_type': 'absolute'
}

def load_vocab(train_path):
    """
    为了保证推理时的 token 索引与训练时一致，我们需要重新构建词表。
    注意：在工业级应用中，通常会 pickle 保存 vocab 对象，这里为了简化依赖，
    我们通过重新加载小部分训练数据来重建词表。
    """
    if not os.path.exists(train_path):
        logging.error(f"Error: Training data not found at {train_path}. Cannot build vocabulary.")
        sys.exit(1)
        
    logging.info("Building vocabulary from training data...")
    processor = TextProcessor()
    # 只需要加载数据来构建词表，不需要返回 pairs
    _, zh_vocab, en_vocab = load_and_process_data(train_path, processor, is_train=True)
    return zh_vocab, en_vocab, processor

def load_model(model_type, checkpoint_path, zh_vocab, en_vocab):
    """
    加载模型架构并读取权重
    """
    input_dim = len(zh_vocab.word2idx)
    output_dim = len(en_vocab.word2idx)
    
    logging.info(f"Loading {model_type} model from {checkpoint_path}...")
    
    if model_type == 'rnn':
        config = RNN_CONFIG
        attn = Attention(config['hid_dim'], method=config['attn_method'])
        enc = Encoder(input_dim, config['enc_emb_dim'], config['hid_dim'], 
                      config['n_layers'], config['dropout'], rnn_type=config['rnn_type'])
        dec = Decoder(output_dim, config['dec_emb_dim'], config['hid_dim'], 
                      config['n_layers'], config['dropout'], attn, rnn_type=config['rnn_type'])
        model = Seq2Seq(enc, dec, device).to(device)
        
    elif model_type == 'transformer':
        config = TRANSFORMER_CONFIG
        model = TransformerNMT(
            input_dim, output_dim, 
            src_pad_idx=PAD_IDX, trg_pad_idx=PAD_IDX,
            emb_dim=config['emb_dim'], 
            n_heads=config['n_heads'], 
            n_layers=config['n_layers'],
            ffn_dim=config['ffn_dim'],
            norm_type=config['norm_type'],
            pos_type=config['pos_type']
        ).to(device)
    else:
        raise ValueError("Invalid model_type. Choose 'rnn' or 'transformer'.")

    # 加载权重
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except FileNotFoundError:
        logging.error(f"Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)
        
    model.eval()
    return model

def predict(model, sentence, processor, zh_vocab, en_vocab, model_type):
    """
    对单个句子进行推理
    """
    model.eval()
    
    # 1. 预处理输入
    # 清洗 + 分词
    tokens = processor.tokenize_cn(processor.clean_text(sentence))
    # 转化为索引 + 添加 <sos>/<eos>
    indices = [zh_vocab.word2idx['<sos>']] + zh_vocab.numericalize(tokens) + [zh_vocab.word2idx['<eos>']]
    src_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device) # [1, len]
    
    # 2. 生成翻译 (使用 Greedy Search)
    max_len = 50
    sos_idx = en_vocab.word2idx['<sos>']
    eos_idx = en_vocab.word2idx['<eos>']
    
    # 初始化 Decoder 输入
    trg_indices = torch.tensor([[sos_idx]], device=device)
    
    with torch.no_grad():
        for _ in range(max_len):
            if model_type == 'rnn':
                output = model(src_tensor, trg_indices, teacher_forcing_ratio=0)
                next_token_logits = output[:, -1, :]
            else:
                output = model(src_tensor, trg_indices)
                next_token_logits = output[:, -1, :]
            
            next_token = next_token_logits.argmax(1).unsqueeze(1)
            trg_indices = torch.cat([trg_indices, next_token], dim=1)
            
            if next_token.item() == eos_idx:
                break
    
    # 3. 将索引转回文本
    trg_tokens = []
    for idx in trg_indices.squeeze(0):
        idx_val = idx.item()
        if idx_val in [sos_idx, eos_idx, PAD_IDX]:
            continue
        trg_tokens.append(en_vocab.idx2word[idx_val])
        
    return " ".join(trg_tokens)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Chinese-English Translation Inference')
    parser.add_argument('--model_type', type=str, default='transformer', choices=['rnn', 'transformer'], help='Model type: rnn or transformer')
    parser.add_argument('--checkpoint', type=str, default='best_transformer.pt', help='Path to model checkpoint')
    # 注意：这里需要指向你的训练数据来构建词表
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data (jsonl) to build vocabulary')
    
    args = parser.parse_args()

    print(f"Initializing {args.model_type.upper()} Translation Model...")
    
    # 1. 加载词表
    zh_vocab, en_vocab, processor = load_vocab(args.data_path)
    
    # 2. 加载模型
    model = load_model(args.model_type, args.checkpoint, zh_vocab, en_vocab)
    
    print("-" * 50)
    print(f"Model loaded successfully! (Vocab size: ZH={len(zh_vocab.word2idx)}, EN={len(en_vocab.word2idx)})")
    print("Enter a Chinese sentence to translate (or type 'q' to quit).")
    print("-" * 50)
    
    # 3. 交互式循环
    while True:
        try:
            src_text = input("中文输入 (Chinese): ")
            if src_text.lower() in ['q', 'quit', 'exit']:
                break
            
            if not src_text.strip():
                continue
                
            translation = predict(model, src_text, processor, zh_vocab, en_vocab, args.model_type)
            print(f"英文翻译 (English): {translation}")
            print("-" * 20)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error during inference: {e}")