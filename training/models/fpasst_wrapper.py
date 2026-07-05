from models.fpasst import get_model
from models.seq_models import BidirectionalLSTM, BidirectionalGRU
from abc import ABC, abstractmethod
import torch.nn as nn
import torch
import torchaudio
import os

import config

sz_float = 4  # size of a float
epsilon = 10e-8  # fudge factor for normalization

# From https://github.com/fschmid56/PretrainedSED


class AugmentMelSTFT(nn.Module):
    def __init__(
            self,
            n_mels=128,
            sr=32000,
            win_length=None,
            hopsize=320,
            n_fft=1024,
            freqm=0,
            timem=0,
            htk=False,
            fmin=0.0,
            fmax=None,
            norm=1,
            fmin_aug_range=1,
            fmax_aug_range=1,
            fast_norm=False,
            preamp=True,
            padding="center",
            periodic_window=True,
    ):
        torch.nn.Module.__init__(self)
        # adapted from: https://github.com/CPJKU/kagglebirds2020/commit/70f8308b39011b09d41eb0f4ace5aa7d2b0e806e

        if win_length is None:
            win_length = n_fft

        if isinstance(win_length, list) or isinstance(win_length, tuple):
            assert isinstance(n_fft, list) or isinstance(n_fft, tuple)
            assert len(win_length) == len(n_fft)
        else:
            win_length = [win_length]
            n_fft = [n_fft]

        self.win_length = win_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sr = sr
        self.htk = htk
        self.fmin = fmin
        if fmax is None:
            fmax = sr // 2 - fmax_aug_range // 2
        self.fmax = fmax
        self.norm = norm
        self.hopsize = hopsize
        self.preamp = preamp
        for win_l in self.win_length:
            self.register_buffer(
                f"window_{win_l}",
                torch.hann_window(win_l, periodic=periodic_window),
                persistent=False,
            )
        assert (
                fmin_aug_range >= 1
        ), f"fmin_aug_range={fmin_aug_range} should be >=1; 1 means no augmentation"
        assert (
                fmin_aug_range >= 1
        ), f"fmax_aug_range={fmax_aug_range} should be >=1; 1 means no augmentation"
        self.fmin_aug_range = fmin_aug_range
        self.fmax_aug_range = fmax_aug_range

        self.register_buffer(
            "preemphasis_coefficient", torch.as_tensor([[[-0.97, 1]]]), persistent=False
        )
        if freqm == 0:
            self.freqm = torch.nn.Identity()
        else:
            self.freqm = torchaudio.transforms.FrequencyMasking(freqm, iid_masks=False)
        if timem == 0:
            self.timem = torch.nn.Identity()
        else:
            self.timem = torchaudio.transforms.TimeMasking(timem, iid_masks=False)
        self.fast_norm = fast_norm
        self.padding = padding
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.iden = nn.Identity()

    def forward(self, x):
        if self.preamp:
            x = nn.functional.conv1d(x.unsqueeze(1), self.preemphasis_coefficient)
        x = x.squeeze(1)

        fmin = self.fmin + torch.randint(self.fmin_aug_range, (1,)).item()
        fmax = self.fmax + self.fmax_aug_range // 2 - torch.randint(self.fmax_aug_range, (1,)).item()

        # don't augment eval data
        if not self.training:
            fmin = self.fmin
            fmax = self.fmax

        mels = []
        for n_fft, win_length in zip(self.n_fft, self.win_length):
            x_temp = x
            if self.padding == "same":
                pad = win_length - self.hopsize
                self.iden(x_temp)  # printing
                x_temp = torch.nn.functional.pad(x_temp, (pad // 2, pad // 2), mode="reflect")
                self.iden(x_temp)  # printing

            x_temp = torch.stft(
                x_temp,
                n_fft,
                hop_length=self.hopsize,
                win_length=win_length,
                center=self.padding == "center",
                normalized=False,
                window=getattr(self, f"window_{win_length}"),
                return_complex=True
            )
            x_temp = torch.view_as_real(x_temp)
            x_temp = (x_temp ** 2).sum(dim=-1)  # power mag

            mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(self.n_mels, n_fft, self.sr,
                                                                     fmin, fmax, vtln_low=100.0, vtln_high=-500.,
                                                                     vtln_warp_factor=1.0)
            mel_basis = torch.as_tensor(torch.nn.functional.pad(mel_basis, (0, 1), mode='constant', value=0),
                                        device=x.device)

            with torch.amp.autocast('cuda', enabled=False):
                x_temp = torch.matmul(mel_basis, x_temp)

            x_temp = torch.log(torch.clip(x_temp, min=1e-7))

            mels.append(x_temp)

        mels = torch.stack(mels, dim=1)

        if self.training:
            mels = self.freqm(mels)
            mels = self.timem(mels)
        if self.fast_norm:
            mels = (mels + 4.5) / 5.0  # fast normalization

        return mels

    def extra_repr(self):
        return "winsize={}, hopsize={}".format(self.win_length, self.hopsize)


class BaseModelWrapper(ABC, nn.Module):
    @abstractmethod
    def mel_forward(self, x):
        """Process input waveform to mel spectrogram."""
        pass

    @abstractmethod
    def forward(self, x):
        """Extract embedding sequence from mel spectrogram."""
        pass

    @abstractmethod
    def separate_params(self):
        """Separate model parameters into predefined groups for layer-wise learning rate decay."""
        pass


class FPaSSTWrapper(BaseModelWrapper):
    def __init__(self):
        super().__init__()
        self.mel = AugmentMelSTFT(
            n_mels=128,
            sr=16_000,
            win_length=400,
            hopsize=160,
            n_fft=512,
            freqm=0,
            timem=0,
            htk=False,
            fmin=0.0,
            fmax=None,
            norm=1,
            fmin_aug_range=10,
            fmax_aug_range=2000,
            fast_norm=True,
            preamp=True,
        )
        self.fpasst = get_model(
            arch="passt_deit_bd_p16_384",
            n_classes=527,
            pos_embed_length=250,
            frame_patchout=0,
            in_channels=16
        )

    def mel_forward(self, x):
        return self.mel(x)

    def forward(self, x):
        return self.fpasst(x)

    def separate_params(self):
        pt_params = [[], [], [], [], [], [], [], [], [], [], [], []]
        for k, p in self.fpasst.named_parameters():
            if k in ['cls_token',
                     'dist_token',
                     'new_pos_embed',
                     'freq_new_pos_embed',
                     'time_new_pos_embed',
                     'conv_in_1.weight',
                     'conv_in_1.bias',
                     'conv_in_2.weight',
                     'conv_in_2.bias',
                     'conv_in_3.weight',
                     'conv_in_3.bias',
                     'patch_embed.proj.weight',
                     'patch_embed.proj.bias',
                     ]:
                pt_params[0].append(p)
            elif 'blocks.0.' in k:
                pt_params[0].append(p)
            elif 'blocks.1.' in k:
                pt_params[1].append(p)
            elif 'blocks.2.' in k:
                pt_params[2].append(p)
            elif 'blocks.3.' in k:
                pt_params[3].append(p)
            elif 'blocks.4.' in k:
                pt_params[4].append(p)
            elif 'blocks.5.' in k:
                pt_params[5].append(p)
            elif 'blocks.6.' in k:
                pt_params[6].append(p)
            elif 'blocks.7.' in k:
                pt_params[7].append(p)
            elif 'blocks.8.' in k:
                pt_params[8].append(p)
            elif 'blocks.9.' in k:
                pt_params[9].append(p)
            elif 'blocks.10.' in k:
                pt_params[10].append(p)
            elif 'blocks.11.' in k:
                pt_params[11].append(p)
            elif k in ['norm.weight', 'norm.bias']:
                pt_params[11].append(p)
            else:
                raise ValueError(f"Check separate params for frame-passt! Unexpected key: {k}")
        return list(reversed(pt_params))


class PredictionsWrapper(nn.Module):
    """
        A wrapper module that adds an optional sequence model and classification heads on top of a transformer.
        It implements equations (1), (2), and (3) in the paper.

        Args:
            base_model (BaseModelWrapper): The base model (transformer) providing sequence embeddings
            checkpoint (str, optional): checkpoint name for loading pre-trained weights. Default is None.
            n_classes_strong (int): Number of classes for strong predictions. Default is 447.
            n_classes_weak (int, optional): Number of classes for weak predictions. Default is None,
                                            which sets it equal to n_classes_strong.
            embed_dim (int, optional): Embedding dimension of the base model output. Default is 768.
            seq_len (int, optional): Desired sequence length. Default is 250 (40 ms resolution).
            seq_model_type (str, optional): Type of sequence model to use.
                                            Default is None, which means no additional sequence model is used.
            head_type (str, optional): Type of classification head. Choices are ["linear", "attention", "None"].
                                       Default is "linear". "None" means that sequence embeddings are returned.
            rnn_layers (int, optional): Number of RNN layers if seq_model_type is "rnn". Default is 2.
            rnn_type (str, optional): Type of RNN to use. Choices are ["BiGRU", "BiLSTM"]. Default is "BiGRU".
            rnn_dim (int, optional): Dimension of RNN hidden state if seq_model_type is "rnn". Default is 256.
            rnn_dropout (float, optional): Dropout rate for RNN layers. Default is 0.0.
        """

    def __init__(self,
                 base_model,
                 ckpt_path=None,
                 n_classes_strong=1,
                 n_classes_weak=None,
                 embed_dim=768,
                 seq_len=250,
                 seq_model_type=None,
                 head_type="linear",
                 rnn_layers=2,
                 rnn_type="BiGRU",
                 rnn_dim=2048,
                 rnn_dropout=0.0
                 ):
        super(PredictionsWrapper, self).__init__()
        self.model = base_model
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.n_classes_strong = n_classes_strong
        self.n_classes_weak = n_classes_weak if n_classes_weak is not None else n_classes_strong
        self.seq_model_type = seq_model_type
        self.head_type = head_type

        if self.seq_model_type == "rnn":
            if rnn_type == "BiGRU":
                self.seq_model = BidirectionalGRU(
                    n_in=self.embed_dim,
                    n_hidden=rnn_dim,
                    dropout=rnn_dropout,
                    num_layers=rnn_layers
                )
            elif rnn_type == "BiLSTM":
                self.seq_model = BidirectionalLSTM(
                    nIn=self.embed_dim,
                    nHidden=rnn_dim,
                    nOut=rnn_dim * 2,
                    dropout=rnn_dropout,
                    num_layers=rnn_layers
                )
            num_features = rnn_dim * 2
        elif self.seq_model_type is None:
            self.seq_model = nn.Identity()
            # no additional sequence model
            num_features = self.embed_dim
        else:
            raise ValueError(f"Unknown seq_model_type: {self.seq_model_type}")

        if self.head_type == "attention":
            assert self.n_classes_strong == self.n_classes_weak, "head_type=='attention' requires number of strong and " \
                                                                 "weak classes to be the same!"

        if self.head_type is not None:
            self.strong_head = nn.Linear(num_features, self.n_classes_strong)
            self.weak_head = nn.Linear(num_features, self.n_classes_weak)
        if ckpt_path is not None:
            print("Loading pretrained checkpoint: ", ckpt_path)
            self.load_checkpoint(ckpt_path)

    def _normalize(self, k):
        # strip leading 'model.' / 'model.fpasst.'
        while True:
            if k.startswith("model.fpasst."):
                k = k[len("model.fpasst."):]
            elif k.startswith("model."):
                k = k[len("model."):]
            else:
                break
        return k
    
    def _add_prefix(self, k):
        if k.startswith("strong_head.") or k.startswith("weak_head."):
            return k
        if k.startswith("mel."):
            return "model." + k
        return "model.fpasst." + k
    
    def load_checkpoint(self, ckpt_path):
        # ckpt_file = os.path.join(config.checkpoints_folder, checkpoint + ".pt")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        # TODO: band-aid fix check later
        state_dict = {self._normalize(k): v for k, v in state_dict.items()}
        state_dict = {self._add_prefix(k): v for k, v in state_dict.items()}

        model_keys = set(self.state_dict().keys())
        ckpt_keys = set(state_dict.keys())

        # compatibility with uniform wrapper structure we introduced for the public repo
        # if 'fpasst' in ckpt_path:
        #     state_dict = {("model.fpasst." + k[len("model."):] if k.startswith("model.")
        #                    else k): v for k, v in state_dict.items()}
        # elif 'M2D' in ckpt_path:
        #     state_dict = {("model.m2d." + k[len("model."):] if not k.startswith("model.m2d.") and k.startswith("model.")
        #                    else k): v for k, v in state_dict.items()}
        # elif 'BEATs' in ckpt_path:
        #     state_dict = {("model.beats." + k[len("model.model."):] if k.startswith("model.model")
        #                    else k): v for k, v in state_dict.items()}
        # elif 'ASIT' in ckpt_path:
        #     state_dict = {("model.asit." + k[len("model."):] if k.startswith("model.")
        #                    else k): v for k, v in state_dict.items()}

        if (not any(k.startswith("model.fpasst.") for k in state_dict)):
            state_dict = {("model.fpasst." + k[len("model."):] if k.startswith("model.")
                            else k): v for k, v in state_dict.items()}

        n_classes_weak_in_sd = state_dict['weak_head.bias'].shape[0] if 'weak_head.bias' in state_dict else -1
        n_classes_strong_in_sd = state_dict['strong_head.bias'].shape[0] if 'strong_head.bias' in state_dict else -1
        seq_model_in_sd = any(['seq_model.' in key for key in state_dict.keys()])
        keys_to_remove = []
        strict = True
        expected_missing = 0
        if self.head_type is None:
            # remove all keys related to head
            keys_to_remove.append('weak_head.bias')
            keys_to_remove.append('weak_head.weight')
            keys_to_remove.append('strong_head.bias')
            keys_to_remove.append('strong_head.weight')
        elif self.seq_model_type is not None and not seq_model_in_sd:
            # we want to train a sequence model (e.g., rnn) on top of a
            #   pre-trained transformer (e.g., AS weak pretrained)
            keys_to_remove.append('weak_head.bias')
            keys_to_remove.append('weak_head.weight')
            keys_to_remove.append('strong_head.bias')
            keys_to_remove.append('strong_head.weight')
            num_seq_model_keys = len([key for key in self.seq_model.state_dict()])
            expected_missing = len(keys_to_remove) + num_seq_model_keys
            strict = False
        else:
            # head type is not None
            if n_classes_weak_in_sd != self.n_classes_weak:
                # remove weak head from sd
                keys_to_remove.append('weak_head.bias')
                keys_to_remove.append('weak_head.weight')
                strict = False
            if n_classes_strong_in_sd != self.n_classes_strong:
                # remove strong head from sd
                keys_to_remove.append('strong_head.bias')
                keys_to_remove.append('strong_head.weight')
                strict = False

            # TODO: Check if this ever becomes a problem
            # It became a problem
            # keys_to_remove += ['strong_head.bias', 'strong_head.weight']
            # strict = False
            # expected_missing = len(keys_to_remove)
            # expected_missing = len(keys_to_remove)

        # allow missing mel parameters for compatibility
        num_mel_keys = len([key for key in self.state_dict() if 'mel_transform' in key])
        if num_mel_keys > 0:
            expected_missing += num_mel_keys
            strict = False

        state_dict = {k: v for k, v in state_dict.items() if k not in keys_to_remove}
        missing, unexpected = self.load_state_dict(state_dict, strict=strict)

        assert len(missing) == expected_missing
        assert len(unexpected) == 0

    def separate_params(self):
        if hasattr(self.model, "separate_params"):
            return self.model.separate_params()
        else:
            raise NotImplementedError("The base model has no 'separate_params' method!'")

    def has_separate_params(self):
        return hasattr(self.model, "separate_params")

    def mel_forward(self, x):
        return self.model.mel_forward(x)

    def forward(self, x):
        # base model is expected to output a sequence (see Eq. (1) in paper)
        # (batch size x sequence length x embedding dimension)
        x = self.model(x)

        # ATST: x.shape: batch size x 250 x 768
        # PaSST: x.shape: batch size x 250 x 768
        # ASiT: x.shape: batch size x 497 x 768
        # M2D: x.shape: batch size x 62 x 3840
        # BEATs: x.shape: batch size x 496 x 768

        assert len(x.shape) == 3

        if x.size(-2) > self.seq_len:
            x = torch.nn.functional.adaptive_avg_pool1d(x.transpose(1, 2), self.seq_len).transpose(1, 2)
        elif x.size(-2) < self.seq_len:
            x = torch.nn.functional.interpolate(x.transpose(1, 2), size=self.seq_len,
                                                mode='linear').transpose(1, 2)

        # Eq. (3) in the paper
        # for teachers this is an RNN, for students it is nn.Identity
        x = self.seq_model(x)

        if self.head_type == "attention":
            # attention head to obtain weak from strong predictions
            # this is typically used for the DESED task, which requires both
            # weak and strong predictions
            strong = torch.sigmoid(self.strong_head(x))
            sof = torch.softmax(self.weak_head(x), dim=-1)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1)
            return strong.transpose(1, 2), weak
        elif self.head_type == "linear":
            # simple linear layers as head (see Eq. (3) in the paper)
            # on AudioSet strong, only strong predictions are used
            # on AudioSet weak, only weak predictions are used
            # why both? because we tried to simultaneously train on AudioSet weak and strong (less successful)
            strong = self.strong_head(x)
            weak = self.weak_head(x.mean(dim=1))
            return strong.transpose(1, 2), weak
        else:
            # no head means the sequence is returned instead of strong and weak predictions
            return x
