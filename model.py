# model.py
import torch
import torch.nn as nn
import random

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, dec_hid_dim] ← 注意：这是 decoder 最后一层的 hidden
        # encoder_outputs: [src_len, batch_size, enc_hid_dim * 2]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # Repeat hidden across source sequence length
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # [bsz, src_len, dec_hid_dim]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # [bsz, src_len, enc_hid_dim * 2]

        # Concatenate and compute energy
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy: [bsz, src_len, dec_hid_dim]

        attention = self.v(energy).squeeze(2)  # [bsz, src_len]
        return torch.softmax(attention, dim=1)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)  # Merge bidirectional states
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [src_len, batch_size]
        embedded = self.dropout(self.embedding(src))  # [src_len, bsz, emb_dim]
        outputs, hidden = self.rnn(embedded)
        # outputs: [src_len, bsz, hid_dim * 2]
        # hidden: [n_layers * 2, bsz, hid_dim]

        # Split into forward and backward
        n_layers = hidden.shape[0] // 2
        hidden_forward = hidden[:n_layers]    # [n_layers, bsz, hid_dim]
        hidden_backward = hidden[n_layers:]   # [n_layers, bsz, hid_dim]

        # Concatenate and compress to single direction
        merged_hidden = torch.tanh(
            self.fc(torch.cat([hidden_forward, hidden_backward], dim=2))
        )  # [n_layers, bsz, hid_dim]

        return outputs, merged_hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim + (hid_dim * 2), hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(emb_dim + hid_dim + (hid_dim * 2), output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input: [batch_size]
        # hidden: [n_layers, batch_size, hid_dim]
        # encoder_outputs: [src_len, batch_size, hid_dim * 2]

        input = input.unsqueeze(0)  # [1, batch_size]
        embedded = self.dropout(self.embedding(input))  # [1, bsz, emb_dim]

        # Use last layer of decoder hidden for attention
        attn_hidden = hidden[-1]  # [batch_size, hid_dim]
        a = self.attention(attn_hidden, encoder_outputs)  # [bsz, src_len]
        a = a.unsqueeze(1)  # [bsz, 1, src_len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # [bsz, src_len, hid_dim*2]
        weighted = torch.bmm(a, encoder_outputs)  # [bsz, 1, hid_dim*2]
        weighted = weighted.permute(1, 0, 2)  # [1, bsz, hid_dim*2]

        rnn_input = torch.cat((embedded, weighted), dim=2)  # [1, bsz, emb_dim + hid_dim*2]

        # hidden is already [n_layers, bsz, hid_dim] → pass directly
        output, hidden = self.rnn(rnn_input, hidden)

        # Prepare for final prediction
        embedded = embedded.squeeze(0)      # [bsz, emb_dim]
        output = output.squeeze(0)          # [bsz, hid_dim]
        weighted = weighted.squeeze(0)      # [bsz, hid_dim*2]

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        # [bsz, output_dim]

        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [src_len, batch_size]
        # trg: [trg_len, batch_size]
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(src)  # hidden: [n_layers, bsz, hid_dim]

        input = trg[0, :]  # <sos> tokens: [batch_size]

        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs