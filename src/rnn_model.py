import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import logging

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, rnn_type='GRU'):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.rnn_type = rnn_type
        
        # 词向量层
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        # RNN层：支持 GRU 和 LSTM
        # bidirectional=False (默认即为单向), batch_first=False (默认 inputs: seq_len, batch, feature)
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        else:
            self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout)
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src: [batch size, src len] -> [src len, batch size]
        src = src.permute(1, 0)
        
        embedded = self.dropout(self.embedding(src))
        # embedded: [src len, batch size, emb dim]
        
        outputs, hidden = self.rnn(embedded)
        
        # outputs: [src len, batch size, hid dim]
        # hidden: [n layers, batch size, hid dim] (GRU)
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hid_dim, method='dot'):
        super().__init__()
        self.method = method # 'dot', 'general' (multiplicative), 'concat' (additive)
        
        if method == 'general':
            self.W = nn.Linear(hid_dim, hid_dim)
        elif method == 'concat':
            self.W = nn.Linear(hid_dim * 2, hid_dim)
            self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch size, hid dim] (Decoder上一时刻的hidden state)
        # encoder_outputs: [src len, batch size, hid dim]
        
        # 调整维度以便广播计算
        src_len = encoder_outputs.shape[0]
        
        # hidden -> [batch size, src len, hid dim]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # encoder_outputs -> [batch size, src len, hid dim]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        # --- Attention Score Calculation ---
        if self.method == 'dot':
            # score = hidden * encoder_output
            energy = torch.sum(hidden * encoder_outputs, dim=2) 
            
        elif self.method == 'general':
            # score = hidden * W * encoder_output
            energy = self.W(encoder_outputs)
            energy = torch.sum(hidden * energy, dim=2)
            
        elif self.method == 'concat':
            # score = v * tanh(W * [hidden; encoder_output])
            combined = torch.cat((hidden, encoder_outputs), dim=2)
            energy = torch.tanh(self.W(combined))
            energy = self.v(energy).squeeze(2)
        
        # energy: [batch size, src len]
        return F.softmax(energy, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention, rnn_type='GRU'):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.rnn_type = rnn_type
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(emb_dim + hid_dim, hid_dim, n_layers, dropout=dropout)
        else:
            self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim, n_layers, dropout=dropout)
            
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        # input: [batch size] (当前时间步的输入词索引)
        input = input.unsqueeze(0) # [1, batch size]
        
        embedded = self.dropout(self.embedding(input)) # [1, batch size, emb dim]
        
        # 取 Decoder 最后一层的 hidden state 用于计算 Attention         # hidden shape: [n layers, batch, hid]
        if self.rnn_type == 'LSTM':
            last_hidden = hidden[0][-1] 
        else:
            last_hidden = hidden[-1]
            
        # 计算 Attention Weights
        a = self.attention(last_hidden, encoder_outputs) # [batch, src len]
        a = a.unsqueeze(1) # [batch, 1, src len]
        
        # 计算 Context Vector
        # encoder_outputs: [src len, batch, hid] -> [batch, src len, hid]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs) # [batch, 1, hid]
        weighted = weighted.permute(1, 0, 2) # [1, batch, hid]
        
        # RNN 输入: Embedding + Context Vector
        rnn_input = torch.cat((embedded, weighted), dim=2)
        
        output, hidden = self.rnn(rnn_input, hidden)
        
        # 预测层: [Embedding; RNN Output; Context Vector] -> Vocabulary
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        训练时的前向传播：支持 Teacher Forcing 和 Free Running
        """
        # src: [batch size, src len]
        # trg: [batch size, trg len]
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        encoder_outputs, hidden = self.encoder(src)
        
        # 第一个输入是 <sos>
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t, :] = output
            
            # Teacher Forcing 策略: 随机决定使用真实标签还是模型预测作为下一步输入
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1) 
            input = trg[:, t] if teacher_force else top1
            
        return outputs

    def beam_search(self, src_tensor, beam_width=3, max_len=50, sos_idx=2, eos_idx=3):
        """
        【新增功能】Beam Search 解码策略
        用于推理阶段 (Inference)，仅支持 batch_size=1
        """
        self.eval() # 切换到评估模式
        with torch.no_grad():
            # src_tensor: [1, src len]
            encoder_outputs, hidden = self.encoder(src_tensor)
            
            # Beam 初始化: [(cumulative_score, [token_history], decoder_hidden)]
            # score 使用 log probability (初始为0)
            beams = [(0, [sos_idx], hidden)] 
            
            for step in range(max_len):
                candidates = []
                
                # 对当前 beam 中的每个路径进行扩展
                for score, seq, h in beams:
                    # 如果该路径已经结束 (<eos>), 直接保留
                    if seq[-1] == eos_idx:
                        candidates.append((score, seq, h))
                        continue
                    
                    # 准备 Decoder 输入
                    input_token = torch.tensor([seq[-1]], device=self.device)
                    
                    # Decoder 单步前向
                    output, new_hidden = self.decoder(input_token, h, encoder_outputs)
                    
                    # 获取 log_softmax 概率 (取 log 方便加法运算)
                    probs = F.log_softmax(output, dim=1) # [1, vocab_size]
                    
                    # 选出 top-k 个候选词
                    topk_probs, topk_ids = probs.topk(beam_width)
                    
                    for i in range(beam_width):
                        token = topk_ids[0][i].item()
                        p = topk_probs[0][i].item()
                        # 更新分数和路径
                        candidates.append((score + p, seq + [token], new_hidden))
                
                # 排序并剪枝 (Pruning)，只保留分数最高的 beam_width 个
                beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]
                
                # 如果所有 beam 都已经生成了 <eos>，则提前退出
                if all(b[1][-1] == eos_idx for b in beams):
                    break
            
            # 返回分数最高的一条路径 (去掉 <sos>)
            best_seq = beams[0][1]
            return best_seq[1:] 

# ==========================================
# 简单的冒烟测试 (Smoke Test)
# ==========================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模拟作业要求的参数
    INPUT_DIM = 100
    OUTPUT_DIM = 100
    ENC_EMB_DIM = 32
    DEC_EMB_DIM = 32
    HID_DIM = 64
    N_LAYERS = 2  # <--- 符合双层要求
    DROPOUT = 0.5
    
    attn = Attention(HID_DIM, method='dot') 
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, DROPOUT, rnn_type='GRU')
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DROPOUT, attn, rnn_type='GRU')
    model = Seq2Seq(enc, dec, device).to(device)
    
    print(f"Model initialized. Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 测试 Greedy (Forward)
    src = torch.randint(0, 100, (4, 10)).to(device) # Batch=4
    trg = torch.randint(0, 100, (4, 12)).to(device)
    out = model(src, trg)
    print(f"Greedy Forward Output Shape: {out.shape}") # [4, 12, 100]
    
    # 测试 Beam Search (Inference)
    print("Testing Beam Search...")
    src_single = torch.randint(0, 100, (1, 10)).to(device) # Batch=1
    beam_out = model.beam_search(src_single, beam_width=3, max_len=15)
    print(f"Beam Search Result (Indices): {beam_out}")