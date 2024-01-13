from torch import nn
import torch.nn.init as init
import torch
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, nhead, num_layers, dim_feedforward, dropout):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.pos_encoder = PositionalEncoding(emb_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(emb_size, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.emb_size = emb_size

    def forward(self, src, src_mask, src_padding_mask):
        src_emb = self.embedding(src) * math.sqrt(self.emb_size)
        #print("nan count in embedding: ", torch.isnan(src_emb).sum().item())
        src_emb = self.pos_encoder(src_emb)
        #print(f"Max value in pos encoding: {src_emb.max().item()}, Min value pos encoding: {src_emb.min().item()}")
        #print(f"stddev value pos encoding: {src_emb.std().item()}, mean value pos encoding: {src_emb.mean().item()}")
        #print("nan count in pos encoder: ", torch.isnan(src_emb).sum().item())
        #print("nan count in src: ", torch.isnan(src).sum().item())
        #print("nan count in src_mask: ", torch.isnan(src_mask).sum().item())
        #print("nan count in src_padding_mask: ", torch.isnan(src_padding_mask).sum().item())
        output = self.transformer_encoder(src_emb, mask=src_mask, src_key_padding_mask=src_padding_mask)
        # assert no nan values
        #assert torch.isnan(output).sum() == 0, f"output contains {torch.isnan(output).sum()} nan values"
        #print("nan count in encoder: ", torch.isnan(output).sum().item())
        return output

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, emb_size, nhead, num_layers, dim_feedforward, dropout):
        super(TransformerDecoder, self).__init__()
        self.emb_size = emb_size
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.pos_encoder = PositionalEncoding(emb_size, dropout)
        decoder_layers = nn.TransformerDecoderLayer(emb_size, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)

    def forward(self, tgt, memory, tgt_mask, tgt_padding_mask, memory_key_padding_mask):
        tgt_emb = self.embedding(tgt) * math.sqrt(self.emb_size)
        tgt_emb = self.pos_encoder(tgt_emb)
        output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        return output


class Seq2SeqTransformer(nn.Module):
    def __init__(self, encoder, decoder, vocab_size=None, special_symbols=None, device=None):
        super(Seq2SeqTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.fc = nn.Linear(decoder.emb_size, vocab_size)
        self.apply(self.init_weights)
        self.special_symbols = special_symbols
        self.device = device

    def forward(self, x):
        print("forward")
        src = x
        tgt=self.shift_right(src)
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(src, tgt)
        self.check_transformer_input_dimensions(src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        memory = self.encoder(src, src_mask, src_padding_mask)
        outs = self.decoder(tgt, memory, tgt_mask, tgt_padding_mask, src_padding_mask)
        outs= self.fc(outs)
        return outs, tgt
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Embedding):
            nn.init.uniform_(m.weight, -0.05, 0.05)

    def shift_right(self, target_sequences):
        batch_size, seq_len = target_sequences.size()
        sos_tensor = torch.full((batch_size, 1), self.special_symbols["SOS"], dtype=target_sequences.dtype, device=target_sequences.device)    
        shifted_sequences = torch.cat([sos_tensor, target_sequences[:, :-1]], dim=1)
        return shifted_sequences
    
    def create_mask(self, src, tgt):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        mask = (torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device=self.device)) == 1).transpose(0, 1)
        tgt_mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        src_mask = torch.zeros((src_seq_len, src_seq_len),device=self.device).type(torch.bool)

        src_padding_mask = (src == self.special_symbols["PAD"]).transpose(0, 1)
        tgt_padding_mask = (tgt == self.special_symbols["PAD"]).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    
    def check_transformer_input_dimensions(self, src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        batch_size = src.size(1)
        src_seq_len = src.size(0)
        tgt_seq_len = tgt_input.size(0)

        assert src.dim() == 2, f"src should have 2 dimensions, but has {src.dim()}"
        assert src.size(1) == batch_size, f"The second dimension of src should be batch_size, but is {src.size(1)}"

        # assert no nan values
        assert torch.isnan(src).sum() == 0, f"src contains {torch.isnan(src).sum()} nan values"
        assert torch.isnan(tgt_input).sum() == 0, f"tgt_input contains {torch.isnan(tgt_input).sum()} nan values"
        assert torch.isnan(src_mask).sum() == 0, f"src_mask contains {torch.isnan(src_mask).sum()} nan values"
        assert torch.isnan(tgt_mask).sum() == 0, f"tgt_mask contains {torch.isnan(tgt_mask).sum()} nan values"
        assert torch.isnan(src_padding_mask).sum() == 0, f"src_padding_mask contains {torch.isnan(src_padding_mask).sum()} nan values"
        assert torch.isnan(tgt_padding_mask).sum() == 0, f"tgt_padding_mask contains {torch.isnan(tgt_padding_mask).sum()} nan values"
        assert torch.isnan(memory_key_padding_mask).sum() == 0, f"memory_key_padding_mask contains {torch.isnan(memory_key_padding_mask).sum()} nan values"
            
        assert tgt_input.dim() == 2, f"tgt_input should have 2 dimensions, but has {tgt_input.dim()}"
        assert tgt_input.size(1) == batch_size, f"The second dimension of tgt_input should be batch_size, but is {tgt_input.size(1)}"
        
        assert src_mask.dim() == 2, f"src_mask should have 2 dimensions, but has {src_mask.dim()}"
        assert src_mask.size() == (src_seq_len, src_seq_len), f"src_mask should have shape (src_seq_len, src_seq_len), but has {src_mask.size()}"
        
        assert tgt_mask.dim() == 2, f"tgt_mask should have 2 dimensions, but has {tgt_mask.dim()}"
        assert tgt_mask.size() == (tgt_seq_len, tgt_seq_len), f"tgt_mask should have shape (tgt_seq_len, tgt_seq_len), but has {tgt_mask.size()}"
        
        for mask, name in zip([src_padding_mask, memory_key_padding_mask], ["src_padding_mask", "memory_key_padding_mask"]):
            assert mask.dim() == 2, f"{name} should have 2 dimensions, but has {mask.dim()}"
            assert mask.size() == (batch_size, src_seq_len), f"{name} should have shape (batch_size, src_seq_len), but has {mask.size()}"
        
        assert tgt_padding_mask.dim() == 2, f"tgt_padding_mask should have 2 dimensions, but has {tgt_padding_mask.dim()}"
        assert tgt_padding_mask.size() == (batch_size, tgt_seq_len), f"tgt_padding_mask should have shape (batch_size, tgt_seq_len), but has {tgt_padding_mask.size()}"



    
    
def get_text_from_model_output(output, idx_to_word, PAD_IDX=1):
    output = output.argmax(dim=2)
    output_text=[]
    for i in range(output.size(0)):
        for word_idx in output[i]:
            output_token=""
            if word_idx.item() == PAD_IDX:
                break
            elif word_idx.item() == 0:
                output_token="<unk>"
            else:
                output_token=idx_to_word[word_idx.item()]
            output_text.append(output_token)
    return " ".join(output_text[:30])
