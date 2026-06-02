import torch.nn as nn

#from .ablation import AblaDec, AblaEnc
from .ARDecoder import ARDecoder
from .conv import BLConv
from .conformer_en import ConformerEncoder
from .lstm import LSTM
#from .others.convnext import ConvNeXt
#from .others.mlp_mixer import MLPMixer
#from .others.resnet import ResNet
#from .others.swin import SwinTransformerV2
#from .others.vit import ViT
#from .previous.cldnn import CLDNNDec, CLDNNEnc
#from .previous.ott import OttBiLSTM, OttCNN
from .transformer import Transformer


def build_encoder(in_chan: int, arch: str, len_seq: int = 0) -> nn.Module:
    match arch:
        case 'blconv_l':
            return BLConv(in_chan, [3, 3, 3, 3], [64, 128,256,512])
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


def build_decoder(
    dim_in: int,
    num_cls: int,
    arch: str,
    len_seq: int = 0,
    *,
    use_gated_attention: bool = False,
    gating_type: str = "elementwise",
) -> nn.Module:
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
        case 'transformer_xs':
            # Param-matched parallel to ar_transformer_xs (REWI's CTC budget,
            # ~4.64 M with blconv_b). Same d_model / depth / heads as the AR
            # sibling; the FFN is wider (1472 vs 896) to absorb the parameter
            # budget that the AR decoder spends on cross-attention. Combined
            # with the param-free sinusoidal PE in `Transformer`, total params
            # land at 4.649 M (+0.27 % vs REWI 4.637 M).
            return Transformer(size_in=dim_in, num_cls=num_cls,
                               d_model=256, nhead=4, num_layers=2, dim_ff=1472,
                               p_drop=0.1, apply_softmax=False)
        case 'transformer_s':
            return Transformer(size_in=dim_in, num_cls=num_cls,
                               d_model=256, nhead=4, num_layers=4, dim_ff=1024,
                               p_drop=0.1, apply_softmax=False)
        case 'transformer_m':
            return Transformer(size_in=dim_in, num_cls=num_cls,
                               d_model=384, nhead=6, num_layers=6, dim_ff=1536,
                               p_drop=0.12, apply_softmax=False)
        case 'transformer_l':
            return Transformer(size_in=dim_in, num_cls=num_cls,
                               d_model=512, nhead=8, num_layers=8, dim_ff=2048,
                               p_drop=0.15, apply_softmax=False)
        case 'transformer_xl':
            return Transformer(size_in=dim_in, num_cls=num_cls,
                               d_model=512, nhead=8, num_layers=12, dim_ff=2048,
                               p_drop=0.2, apply_softmax=False)

        # AR Transformer decoders (cross-attention)
        case 'ar_transformer_xs':
            # Param-matched to REWI's CTC baseline (~4.64M total with blconv_b enc).
            # Half the depth of `s` and slightly narrower FFN to land on budget.
            return ARDecoder(vocab_size=num_cls, d_model=256, nhead=4, layers=2,
                             dim_ff=896, pdrop=0.1, use_gated_attention=use_gated_attention,
                             gating_type=gating_type)
        case 'ar_transformer_s':
            return ARDecoder(vocab_size=num_cls, d_model=256, nhead=4, layers=4,
                             dim_ff=1024, pdrop=0.1, use_gated_attention=use_gated_attention,
                             gating_type=gating_type)
        case 'ar_transformer_m':
            return ARDecoder(vocab_size=num_cls, d_model=384, nhead=6, layers=6,
                             dim_ff=1536, pdrop=0.12, use_gated_attention=use_gated_attention,
                             gating_type=gating_type)
        case 'ar_transformer_l':
            return ARDecoder(vocab_size=num_cls, d_model=512, nhead=8, layers=8,
                             dim_ff=2048, pdrop=0.15, use_gated_attention=use_gated_attention,
                             gating_type=gating_type)
        case _:
            raise ValueError(f"Unknown decoder arch: {arch}")
