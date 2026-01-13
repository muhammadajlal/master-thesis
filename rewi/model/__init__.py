import torch
import torch.nn as nn

from .ablation import AblaDec, AblaEnc
from .conv import BLConv
from .lstm import LSTM
from .others.convnext import ConvNeXt
from .others.mlp_mixer import MLPMixer
from .others.resnet import ResNet
from .others.swin import SwinTransformerV2
from .others.vit import ViT
from .previous.cldnn import CLDNNDec, CLDNNEnc
from .previous.ott import OttBiLSTM, OttCNN
from .transformer import Transformer
from .conformer_en import ConformerEncoder
from .ARDecoder import ARDecoder


def build_encoder(in_chan: int, arch: str, len_seq: int = 0) -> nn.Module:
    match arch:
        case 'blconv_b':
            return BLConv(in_chan)
        case 'blconv_s':
            return BLConv(in_chan, [1, 1, 1], [64, 128, 256])
        case 'cldnn':
            return CLDNNEnc(in_chan)
        case 'ott':
            return OttCNN(in_chan)
        case 'convnext':
            return ConvNeXt(in_chan)
        case 'mlp_mixer':
            return MLPMixer(in_chan, len_seq)
        case 'resnet':
            return ResNet(in_chan)
        case 'swinv2':
            return SwinTransformerV2(in_chan, len_seq)
        case 'vit':
            return ViT(in_chan, len_seq)
        case 'abla':
            return AblaEnc(in_chan, True, True, True, True, True, True)
        case 'conformer_b':
            return ConformerEncoder(in_channels=in_chan)
        case _:
            raise ValueError(f"Unknown encoder arch: {arch}")


def build_decoder(dim_in: int, num_cls: int, arch: str, len_seq: int = 0, 
                  use_gated_attention: bool = False, 
                  gating_type: str = "elementwise") -> nn.Module:
    match arch:
        case 'bilstm_b':
            return LSTM(dim_in, num_cls)
        case 'bilstm_wide':  # width-only, ~Transformer-S size on char setup
            return LSTM(dim_in, num_cls, hidden_size=164, num_layers=3, r_drop=0.2)
        case 'bilstm_s':
            return LSTM(dim_in, num_cls, 64, 2)
        case 'cldnn':
            return CLDNNDec(dim_in, num_cls)
        case 'ott':
            return OttBiLSTM(dim_in, num_cls)
        case 'abla':
            return AblaDec(dim_in, num_cls)

        # CTC per-timestep Transformers
        case 'transformer_s':
            return Transformer(size_in=dim_in, num_cls=num_cls,
                               d_model=256, nhead=4, num_layers=4, dim_ff=1024,
                               p_drop=0.1, apply_softmax=True)
        case 'transformer_m':
            return Transformer(size_in=dim_in, num_cls=num_cls,
                               d_model=384, nhead=6, num_layers=6, dim_ff=1536,
                               p_drop=0.12, apply_softmax=True)
        case 'transformer_l':
            return Transformer(size_in=dim_in, num_cls=num_cls,
                               d_model=512, nhead=8, num_layers=8, dim_ff=2048,
                               p_drop=0.15, apply_softmax=True)
        case 'transformer_xl':
            return Transformer(size_in=dim_in, num_cls=num_cls,
                               d_model=512, nhead=8, num_layers=12, dim_ff=2048,
                               p_drop=0.2, apply_softmax=True)

        # AR Transformer decoders (cross-attention)
        case 'ar_transformer_s':
            return ARDecoder(vocab_size=num_cls, d_model=256, nhead=4, layers=4, 
                             dim_ff=1024, pdrop=0.1, use_gated_attention=use_gated_attention,
                             gating_type=gating_type,)
        case 'ar_transformer_m':
            return ARDecoder(vocab_size=num_cls, d_model=384, nhead=6, layers=6, 
                             dim_ff=1536, pdrop=0.12, use_gated_attention=use_gated_attention,
                             gating_type=gating_type,)
        case 'ar_transformer_l':
            return ARDecoder(vocab_size=num_cls, d_model=512, nhead=8, layers=8, 
                             dim_ff=2048, pdrop=0.15, use_gated_attention=use_gated_attention,
                             gating_type=gating_type,)
        case _:
            raise ValueError(f"Unknown decoder arch: {arch}")


class BaseModel(nn.Module):
    """
    Supports both:
      - CTC pipeline (BLConv + per-timestep decoder)
      - AR pipeline (BLConv + ARDecoder with cross-attention)
    """

    def __init__(self, arch_en: str, arch_de: str, in_chan: int, 
                 num_cls: int, len_seq: int = 0, 
                 use_gated_attention: bool = False,
                 gating_type: str = "elementwise") -> None:
        
        super().__init__()
        self.arch_en = arch_en
        self.arch_de = arch_de
        self.in_chan = in_chan
        self.num_cls = num_cls
        self.len_seq = len_seq

        self.encoder = build_encoder(in_chan, arch_en, len_seq)
        self.decoder = build_decoder(self.encoder.dim_out, num_cls, arch_de,
                                     len_seq // self.encoder.ratio_ds if arch_en != 'trans' else 0,
                                     use_gated_attention=use_gated_attention, gating_type=gating_type,)

        # If AR decoder d_model != encoder dim, add a projection
        self.mem_proj = None
        if isinstance(self.decoder, ARDecoder):
            dec_dim = self.decoder.d_model
            enc_dim = self.encoder.dim_out
            if enc_dim != dec_dim:
                self.mem_proj = nn.Linear(enc_dim, dec_dim)

        

    def _encode_with_mask(self, x: torch.Tensor, in_lengths: torch.Tensor | None):
        # infer raw lengths if not provided (before encoder)
        if in_lengths is None:
            with torch.no_grad():
                valid = (x.abs().sum(dim=1) > 1e-6)   # (B, T) bool, same device as x
                in_lengths = valid.sum(dim=1)         # (B,)
        else:
            # make sure lengths are on the same device as x before encoding
            in_lengths = in_lengths.to(device=x.device)

        feats = self.encoder(x)                        # (B, Tm, Cenc) on CUDA
        Tm = feats.size(1)

        # downsample + clamp
        enc_lengths = torch.div(in_lengths, self.encoder.ratio_ds, rounding_mode='floor')
        enc_lengths = enc_lengths.clamp(min=1, max=Tm)

        # ensure device/dtype match feats for mask construction
        enc_lengths = enc_lengths.to(device=feats.device, dtype=torch.long)

        # key-padding mask: True at PAD
        enc_pad = torch.arange(Tm, device=feats.device).unsqueeze(0) >= enc_lengths.unsqueeze(1)
        return feats, enc_pad


    def forward(self, x: torch.Tensor, in_lengths: torch.Tensor | None = None, y_inp: torch.Tensor | None = None):
        # AR path (teacher forcing or dummy path for profiling)
        if isinstance(self.decoder, ARDecoder):
            mem, enc_pad = self._encode_with_mask(x, in_lengths)     # (B, Tm, Cenc), (B, Tm)
            if self.mem_proj is not None:
                mem = self.mem_proj(mem)                              # (B, Tm, d_model)

            # If no y_inp was provided (e.g., thop/profile calls model(x)), use a 1-token dummy.
            if y_inp is None:
                B = mem.size(0)
                # any token id < vocab_size works for MACs; use 0
                y_inp = torch.zeros(B, 1, dtype=torch.long, device=mem.device)

            return self.decoder(y_inp, mem, enc_pad)                  # (B, N, V)

        # CTC path (per-timestep decoder)
        feats = self.encoder(x)                                       # (B, T', C')
        return self.decoder(feats)


    def infer(self) -> None:
        if hasattr(self.encoder, 'fuse'):
            self.encoder.fuse()
        if hasattr(self.decoder, 'recurrent'):
            self.decoder.recurrent = True

    @property
    def ratio_ds(self) -> int:
        return self.encoder.ratio_ds
