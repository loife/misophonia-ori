import config

import argparse
import torch
import transformers
from data.datasets import get_strong_training_ds, get_strong_eval_ds, get_weak_eval_ds, get_weak_training_ds
from data.encode import ManyHotEncoder
from pathlib import Path
from typing import Dict
from datetime import datetime
import random
from einops import rearrange

from torch.utils.data import DataLoader
import pandas as pd
import torch.nn as nn
import pytorch_lightning as pl

from models.fpasst_wrapper import FPaSSTWrapper, PredictionsWrapper
from utils.scores import EventBasedScore, SegmentBasedScore, combine_target_events, get_events_for_all_files
from models.augment import RandomResizeCrop, mixstyle, mixup, frame_shift, filter_augmentation, time_mask

import wandb
import numpy as np
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

class StrongModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        fpasst = FPaSSTWrapper()
        self.model = PredictionsWrapper(fpasst, ckpt_path=args.checkpoint_path,
                                    seq_model_type=args.seq_model_type,
                                    n_classes_strong=len(config.labels))

        self.strong_loss = nn.BCEWithLogitsLoss()

        self.freq_warp = RandomResizeCrop((1, 1.0), time_scale=(1.0, 1.0))

        ## TODO: Take from vocab when there's more labels
        self.label_to_idx = {}
        for i, label in enumerate(config.labels):
            self.label_to_idx[label] = i

        self.idx_to_label: Dict[int, str] = {
            idx: label for (label, idx) in self.label_to_idx.items()
        }

        # Main, permissive window
        self.event_onset_500ms_fms = EventBasedScore(
            label_to_idx=self.label_to_idx,
            name="event_onset_500ms_fms",
            scores=("f_measure", "precision", "recall"),
            params={"evaluate_onset": True, "evaluate_offset": False, "t_collar": 0.5}
        )

        self.event_onset_50ms_fms = EventBasedScore(
            label_to_idx=self.label_to_idx,
            name="event_onset_50ms_fms",
            scores=("f_measure", "precision", "recall"),
            params={"evaluate_onset": True, "evaluate_offset": False, "t_collar": 0.05}
        )

        self.segment_1s_er = SegmentBasedScore(
            label_to_idx=self.label_to_idx,
            name="segment_1s_er",
            scores=("error_rate",),
            params={"time_resolution": 1.0},
            maximize=False,
        )

        self.postprocessing_grid = {
            "median_filter_ms": [
                250
            ],
            "min_duration": [
                125
            ]
        }

        self.preds, self.tgts, self.fnames, self.timestamps = [], [], [], []

    def forward(self, audio):
        mel = self.model.mel_forward(audio)
        y_strong, _ = self.model(mel)
        return y_strong

    def separate_params(self):
        pt_params = []
        seq_params = []
        head_params = []

        for name, p in self.named_parameters():
            name = name[len("model."):]
            if name.startswith('model'):
                # the transformer
                pt_params.append(p)
            elif name.startswith('seq_model'):
                # the optional sequence model
                seq_params.append(p)
            elif name.startswith('strong_head') or name.startswith('weak_head'):
                # the prediction head
                head_params.append(p)
            else:
                raise ValueError(f"Unexpected key in model: {name}")

        if self.model.has_separate_params():
            # split parameters into groups according to their depth in the network
            # based on this, we can apply layer-wise learning rate decay
            pt_params = self.model.separate_params()
        else:
            if self.args.lr_decay != 1.0:
                raise ValueError(f"Model has no separate_params function. Can't apply layer-wise lr decay, but "
                                 f"learning rate decay is set to {self.args.lr_decay}.")

        return pt_params, seq_params, head_params

    def get_optimizer(
            self,
            lr,
            lr_decay=1.0,
            transformer_lr=None,
            transformer_frozen=False,
            adamw=False,
            weight_decay=0.01,
            betas=(0.9, 0.999)
    ):
        pt_params, seq_params, head_params = self.separate_params()

        param_groups = [
            {'params': head_params, 'lr': lr},  # model head (besides base model and seq model)
        ]

        if transformer_frozen:
            for p in pt_params + seq_params:
                if isinstance(p, list):
                    for p_i in p:
                        p_i.detach_()
                else:
                    p.detach_()
        else:
            if transformer_lr is None:
                transformer_lr = lr
            if isinstance(pt_params, list) and isinstance(pt_params[0], list):
                # apply lr decay
                scale_lrs = [transformer_lr * (lr_decay ** i) for i in range(1, len(pt_params) + 1)]
                param_groups = param_groups + [{"params": pt_params[i], "lr": scale_lrs[i]} for i in
                                               range(len(pt_params))]
            else:
                param_groups.append(
                    {'params': pt_params, 'lr': transformer_lr},  # pretrained model
                )
            param_groups.append(
                {'params': seq_params, 'lr': lr},  # pretrained model
            )

        # do not apply weight decay to biases and batch norms
        param_groups_split = []
        for param_group in param_groups:
            params_1D, params_2D = [], []
            lr = param_group['lr']
            for param in param_group['params']:
                if param.ndimension() >= 2:
                    params_2D.append(param)
                elif param.ndimension() <= 1:
                    params_1D.append(param)
            param_groups_split += [{'params': params_2D, 'lr': lr, 'weight_decay': weight_decay},
                                   {'params': params_1D, 'lr': lr}]
        if weight_decay > 0:
            assert adamw
        if adamw:
            print(f"\nUsing adamw weight_decay={weight_decay}!\n")
            return torch.optim.AdamW(param_groups_split, lr=lr, weight_decay=weight_decay, betas=betas)
        return torch.optim.Adam(param_groups_split, lr=lr, betas=betas)

    def get_lr_scheduler(
            self,
            optimizer,
            num_training_steps,
            schedule_mode="cos",
            gamma: float = 0.999996,
            num_warmup_steps=100,
            lr_end=1e-7,
    ):
        if schedule_mode in {"exp"}:
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
        if schedule_mode in {"cosine", "cos"}:
            return transformers.get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        if schedule_mode in {"linear"}:
            print("Linear schedule!")
            return transformers.get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                power=1.0,
                lr_end=lr_end,
            )
        raise RuntimeError(f"schedule_mode={schedule_mode} Unknown.")

    def configure_optimizers(self):
        """
        This is the way pytorch lightening requires optimizers and learning rate schedulers to be defined.
        The specified items are used automatically in the optimization loop (no need to call optimizer.step() yourself).
        :return: dict containing optimizer and learning rate scheduler
        """
        optimizer = self.get_optimizer(self.args.max_lr_strong,
                                       lr_decay=self.args.lr_decay,
                                       transformer_lr=self.args.transformer_lr,
                                       transformer_frozen=self.args.transformer_frozen,
                                       adamw=False if self.args.no_adamw else True,
                                       weight_decay=self.args.weight_decay)

        num_training_steps = self.trainer.estimated_stepping_batches

        scheduler = self.get_lr_scheduler(optimizer, num_training_steps,
                                          schedule_mode=self.args.schedule_mode,
                                          lr_end=self.args.lr_end)
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return [optimizer], [lr_scheduler_config]

    def training_step(self, train_batch, batch_idx):
        """
        :param train_batch: contains one batch from train dataloader
        :param batch_idx
        :return: a dict containing at least loss that is used to update model parameters, can also contain
                    other items that can be processed in 'training_epoch_end' to log other metrics than loss
        """

        audios = train_batch["audio"]
        labels = train_batch["strong"]
        fnames = train_batch["filename"]
        timestamps = train_batch["timestamps"]

        if self.args.transformer_frozen:
            self.model.model.eval()
            self.model.seq_model.eval()
        mel = self.model.mel_forward(audios)

        # time rolling
        if self.args.frame_shift_range > 0:
            mel, labels = frame_shift(
                mel,
                labels,
                shift_range=self.args.frame_shift_range
            )

        # mixup
        if self.args.mixup_p_strong > random.random():
            mel, labels = mixup(
                mel,
                targets=labels
            )

        # mixstyle
        if self.args.mixstyle_p_strong > random.random():
            mel = mixstyle(
                mel
            )

        # time masking
        if self.args.max_time_mask_size > 0:
            mel, labels = time_mask(
                mel,
                labels,
                max_mask_ratio=self.args.max_time_mask_size
            )

        # frequency masking
        if self.args.filter_augment_p > random.random():
            mel, _ = filter_augmentation(
                mel
            )

        # frequency warping
        if self.args.freq_warp_p > random.random():
            mel = mel.squeeze(1)
            mel = self.freq_warp(mel)
            mel = mel.unsqueeze(1)

        # forward through network; use strong head
        y_hat_strong, _ = self.model(mel)

        loss = self.strong_loss(y_hat_strong, labels)

        # logging
        self.log('epoch', self.current_epoch)
        for i, param_group in enumerate(self.trainer.optimizers[0].param_groups):
            self.log(f'trainer/lr_optimizer_{i}', param_group['lr'])
        self.log("train/loss", loss.detach().cpu(), prog_bar=True)

        return loss

    def _score_step(self, batch):
        audios = batch["audio"]
        labels = batch["strong"]
        fnames = batch["filename"]
        timestamps = batch["timestamps"]

        strong_preds = self.forward(audios)

        self.preds.append(strong_preds)
        self.tgts.append(labels)
        self.fnames.append(fnames)
        self.timestamps.append(timestamps)

    def _score_epoch_end(self, name="val"):
        preds = torch.cat(self.preds)
        tgts = torch.cat(self.tgts)
        fnames = [item for sublist in self.fnames for item in sublist]
        timestamps = torch.cat(self.timestamps)
        val_loss = self.strong_loss(preds, tgts)
        self.log(f"{name}/loss", val_loss, prog_bar=True)

        # AUROC scoring
        p = torch.sigmoid(preds).to(torch.float32).transpose(1, 2).reshape(-1, preds.size(1)).cpu().numpy()
        t = tgts.transpose(1, 2).reshape(-1, tgts.size(1)).cpu().numpy()

        # the following function expects one prediction per timestamp (sequence dimension must be flattened)
        seq_len = preds.size(-1)
        preds = rearrange(preds, 'bs c t -> (bs t) c').float()
        timestamps = rearrange(timestamps, 'bs t -> (bs t)').float()
        fnames = [fname for fname in fnames for _ in range(seq_len)]

        predicted_events_by_postprocessing = get_events_for_all_files(
            preds,
            fnames,
            timestamps,
            self.idx_to_label,
            self.postprocessing_grid
        )

        # we only have one postprocessing configurations (aligned with HEAR challenge)
        key = list(predicted_events_by_postprocessing.keys())[0]
        predicted_events = predicted_events_by_postprocessing[key]


        for c in range(t.shape[1]):
            if t[:, c].min() != t[:, c].max():
                self.log(f"{name}/auroc_c{c}", roc_auc_score(t[:, c], p[:, c]))
                self.log(f"{name}/pauroc_c{c}", roc_auc_score(t[:, c], p[:, c], max_fpr=0.1))
                self.log(f"{name}/ap_c{c}", average_precision_score(t[:, c], p[:, c]))

        # load ground truth for test fold
        task_path = Path(self.args.strong_ds_path)
        test_target_events = combine_target_events(["valid" if name == "val" else "test"], task_path)
        onset_fms = self.event_onset_500ms_fms(predicted_events, test_target_events)
        onset_fms_50 = self.event_onset_50ms_fms(predicted_events, test_target_events)
        segment_1s_er = self.segment_1s_er(predicted_events, test_target_events)

        self.log(f"{name}/onset_fms", onset_fms[0][1])
        self.log(f"{name}/onset_fms_50", onset_fms_50[0][1])
        self.log(f"{name}/segment_1s_er", segment_1s_er[0][1])

        # free buffers
        self.preds, self.tgts, self.fnames, self.timestamps = [], [], [], []

    def validation_step(self, batch, batch_idx):
        self._score_step(batch)

    def on_validation_epoch_end(self):
        self._score_epoch_end(name="val")

    def test_step(self, batch, batch_idx):
        self._score_step(batch)

    def on_test_epoch_end(self):
        self._score_epoch_end(name="test")


class WeakModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        fpasst = FPaSSTWrapper()
        self.model = PredictionsWrapper(fpasst, ckpt_path=args.checkpoint_path,
                                    seq_model_type=args.seq_model_type,
                                    n_classes_strong=len(config.labels),
                                    n_classes_weak=len(config.labels))

        self.weak_loss = nn.BCEWithLogitsLoss()

        self.freq_warp = RandomResizeCrop((1, 1.0), time_scale=(1.0, 1.0))
        self.val_threshold = args.weak_threshold

        ## TODO: Take from vocab when there's more labels
        self.label_to_idx = {}
        for i, label in enumerate(config.labels):
            self.label_to_idx[label] = i

        self.idx_to_label: Dict[int, str] = {
            idx: label for (label, idx) in self.label_to_idx.items()
        }

        self.preds, self.tgts  = [], []

    def forward(self, audio):
        mel = self.model.mel_forward(audio)
        _, y_weak = self.model(mel)
        return y_weak

    def separate_params(self):
        pt_params = []
        seq_params = []
        head_params = []

        for name, p in self.named_parameters():
            name = name[len("model."):]
            if name.startswith('model'):
                # the transformer
                pt_params.append(p)
            elif name.startswith('seq_model'):
                # the optional sequence model
                seq_params.append(p)
            elif name.startswith('strong_head') or name.startswith('weak_head'):
                # the prediction head
                head_params.append(p)
            else:
                raise ValueError(f"Unexpected key in model: {name}")

        if self.model.has_separate_params():
            # split parameters into groups according to their depth in the network
            # based on this, we can apply layer-wise learning rate decay
            pt_params = self.model.separate_params()
        else:
            if self.args.lr_decay != 1.0:
                raise ValueError(f"Model has no separate_params function. Can't apply layer-wise lr decay, but "
                                 f"learning rate decay is set to {self.args.lr_decay}.")

        return pt_params, seq_params, head_params

    def get_optimizer(
            self,
            lr,
            lr_decay=1.0,
            transformer_lr=None,
            transformer_frozen=False,
            adamw=False,
            weight_decay=0.01,
            betas=(0.9, 0.999)
    ):
        pt_params, seq_params, head_params = self.separate_params()

        param_groups = [
            {'params': head_params, 'lr': lr},  # model head (besides base model and seq model)
        ]

        if transformer_frozen:
            for p in pt_params + seq_params:
                if isinstance(p, list):
                    for p_i in p:
                        p_i.detach_()
                else:
                    p.detach_()
        else:
            if transformer_lr is None:
                transformer_lr = lr
            if isinstance(pt_params, list) and isinstance(pt_params[0], list):
                # apply lr decay
                scale_lrs = [transformer_lr * (lr_decay ** i) for i in range(1, len(pt_params) + 1)]
                param_groups = param_groups + [{"params": pt_params[i], "lr": scale_lrs[i]} for i in
                                               range(len(pt_params))]
            else:
                param_groups.append(
                    {'params': pt_params, 'lr': transformer_lr},  # pretrained model
                )
            param_groups.append(
                {'params': seq_params, 'lr': lr},  # pretrained model
            )

        # do not apply weight decay to biases and batch norms
        param_groups_split = []
        for param_group in param_groups:
            params_1D, params_2D = [], []
            lr = param_group['lr']
            for param in param_group['params']:
                if param.ndimension() >= 2:
                    params_2D.append(param)
                elif param.ndimension() <= 1:
                    params_1D.append(param)
            param_groups_split += [{'params': params_2D, 'lr': lr, 'weight_decay': weight_decay},
                                   {'params': params_1D, 'lr': lr}]
        if weight_decay > 0:
            assert adamw
        if adamw:
            print(f"\nUsing adamw weight_decay={weight_decay}!\n")
            return torch.optim.AdamW(param_groups_split, lr=lr, weight_decay=weight_decay, betas=betas)
        return torch.optim.Adam(param_groups_split, lr=lr, betas=betas)

    def get_lr_scheduler(
            self,
            optimizer,
            num_training_steps,
            schedule_mode="cos",
            gamma: float = 0.999996,
            num_warmup_steps=100,
            lr_end=1e-7,
    ):
        if schedule_mode in {"exp"}:
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
        if schedule_mode in {"cosine", "cos"}:
            return transformers.get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        if schedule_mode in {"linear"}:
            print("Linear schedule!")
            return transformers.get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                power=1.0,
                lr_end=lr_end,
            )
        raise RuntimeError(f"schedule_mode={schedule_mode} Unknown.")

    def configure_optimizers(self):
        """
        This is the way pytorch lightening requires optimizers and learning rate schedulers to be defined.
        The specified items are used automatically in the optimization loop (no need to call optimizer.step() yourself).
        :return: dict containing optimizer and learning rate scheduler
        """
        optimizer = self.get_optimizer(self.args.max_lr_weak,
                                       lr_decay=self.args.lr_decay,
                                       transformer_lr=self.args.transformer_lr,
                                       transformer_frozen=self.args.transformer_frozen,
                                       adamw=False if self.args.no_adamw else True,
                                       weight_decay=self.args.weight_decay)

        num_training_steps = self.trainer.estimated_stepping_batches

        scheduler = self.get_lr_scheduler(optimizer, num_training_steps,
                                          schedule_mode=self.args.schedule_mode,
                                          lr_end=self.args.lr_end)
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return [optimizer], [lr_scheduler_config]

    def training_step(self, train_batch, batch_idx):
        """
        :param train_batch: contains one batch from train dataloader
        :param batch_idx
        :return: a dict containing at least loss that is used to update model parameters, can also contain
                    other items that can be processed in 'training_epoch_end' to log other metrics than loss
        """

        audios = train_batch["audio"]
        labels = train_batch["weak"]
        fnames = train_batch["filename"]

        if self.args.transformer_frozen:
            self.model.model.eval()
            self.model.seq_model.eval()
        mel = self.model.mel_forward(audios)

        # mixup
        if self.args.mixup_p_weak > random.random():
            mel, labels = mixup(
                mel,
                targets=labels
            )

        # mixstyle
        if self.args.mixstyle_p_weak > random.random():
            mel = mixstyle(
                mel
            )

        # frequency masking
        if self.args.filter_augment_p > random.random():
            mel, _ = filter_augmentation(
                mel
            )

        # frequency warping
        if self.args.freq_warp_p > random.random():
            mel = mel.squeeze(1)
            mel = self.freq_warp(mel)
            mel = mel.unsqueeze(1)

        # forward through network; use strong head
        _, y_hat_weak = self.model(mel)

        loss = self.weak_loss(y_hat_weak, labels)

        # logging
        self.log('epoch', self.current_epoch)
        for i, param_group in enumerate(self.trainer.optimizers[0].param_groups):
            self.log(f'trainer/lr_optimizer_{i}', param_group['lr'])
        self.log("train/loss", loss.detach().cpu(), prog_bar=True)

        return loss

    def _score_step(self, batch):
        audios = batch["audio"]
        labels = batch["weak"]
        fnames = batch["filename"]
        y_weak = self.forward(audios)
        self.preds.append(y_weak)
        self.tgts.append(labels)

    def _score_epoch_end(self, name="val"):
        preds = torch.cat(self.preds) # (N, n_classes) logits
        tgts  = torch.cat(self.tgts) # (N, n_classes) 0/1

        loss = self.weak_loss(preds, tgts)
        self.log(f"{name}/loss", loss, prog_bar=True)

        probs = torch.sigmoid(preds).to(torch.float32).cpu().numpy()
        y_true = tgts.cpu().numpy()
        y_pred = (probs >= self.val_threshold).astype(int)

        f1s, aps, aucs = [], [], []
        for c in range(y_true.shape[1]):
            f1s.append(f1_score(y_true[:, c], y_pred[:, c], zero_division=0))
            if y_true[:, c].min() != y_true[:, c].max():
                aps.append(average_precision_score(y_true[:, c], probs[:, c]))
                aucs.append(roc_auc_score(y_true[:, c], probs[:, c]))

        self.log(f"{name}/f1", float(np.mean(f1s)), prog_bar=True)
        if aps:
            self.log(f"{name}/ap", float(np.mean(aps)))
            self.log(f"{name}/auroc", float(np.mean(aucs)))

        self.preds, self.tgts = [], []

    def validation_step(self, batch, batch_idx):
        self._score_step(batch)

    def on_validation_epoch_end(self):
        self._score_epoch_end(name="val")

    def test_step(self, batch, batch_idx):
        self._score_step(batch)

    def on_test_epoch_end(self):
        self._score_epoch_end(name="test")


def train(args):
    wandb_logger = WandbLogger(
        project="Misophonia AI",
        config=args,
        name=args.experiment_name
    )
        
    encoder = ManyHotEncoder(config.labels)
    run_id = wandb_logger.experiment.id

    weak_init_ckpt = None

    if(args.training_type == "weak" or args.training_type == "weak-strong"):
        weak_train_dl = DataLoader(get_weak_training_ds(args.weak_ds_path, encoder),
                                   batch_size=args.batch_size, shuffle=True)
        weak_eval_dl  = DataLoader(get_weak_eval_ds(args.weak_ds_path, encoder),
                                   batch_size=args.batch_size, shuffle=False)
        
        weak_module = WeakModule(args)

        weak_ckpt_cb = ModelCheckpoint(
            dirpath=f"checkpoints/misophonia/weak/{run_id}",
            filename="{epoch:02d}-{val/ap:.3f}",
            monitor="val/ap",
            mode="max",
            save_top_k=3, 
            save_last=True,
            )
        
        trainer = pl.Trainer(max_epochs=args.n_epochs_weak,
                    logger=wandb_logger,
                    accelerator='auto',
                    precision=args.precision,
                    num_sanity_val_steps=2,
                    callbacks=[weak_ckpt_cb],
                    check_val_every_n_epoch=args.check_val_every_n_epoch
                    )

        trainer.fit(
            weak_module,
            train_dataloaders=weak_train_dl,
            val_dataloaders=weak_eval_dl,
        )

        best_path = weak_ckpt_cb.best_model_path
        print(f"Best weak checkpoint: {best_path}")

        if(args.training_type == "weak-strong"):
            best_weak = WeakModule.load_from_checkpoint(best_path, args=args)

            weak_init_ckpt = f"checkpoints/misophonia/weak/{run_id}/weak_backbone.pt"
            torch.save(best_weak.model.state_dict(), weak_init_ckpt) 

    if(args.training_type == "strong" or args.training_type == "weak-strong"):
        if weak_init_ckpt is not None:
            args.checkpoint_path = weak_init_ckpt

        strong_train_dl = DataLoader(get_strong_training_ds(args.strong_ds_path, encoder),
                                     batch_size=args.batch_size, shuffle=True)
        strong_eval_dl  = DataLoader(get_strong_eval_ds(args.strong_ds_path, encoder),
                                     batch_size=args.batch_size, shuffle=False)

        
        pl_module = StrongModule(args)

        checkpoint_cb = ModelCheckpoint(
            dirpath=f"checkpoints/misophonia/strong/{run_id}",
            filename="{epoch:02d}-{val/onset_fms:.3f}",
            monitor="val/onset_fms",
            mode="max",
            save_top_k=3, 
            save_last=True,
            )
        
        trainer = pl.Trainer(max_epochs=args.n_epochs_strong,
                            logger=wandb_logger,
                            accelerator='auto',
                            precision=args.precision,
                            num_sanity_val_steps=2,
                            callbacks=[checkpoint_cb],
                            check_val_every_n_epoch=args.check_val_every_n_epoch
                            )
        

        trainer.fit(
            pl_module,
            train_dataloaders=strong_train_dl,
            val_dataloaders=strong_eval_dl,
        )


        
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_model_type', type=str, choices=["rnn"],
                    default=None)

    # general
    parser.add_argument('--experiment_name', type=str, default="Misophonia_Strong")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--precision', type=int, default=16)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=5)

    # training
    parser.add_argument('--n_epochs_strong', type=int, default=30)
    parser.add_argument('--n_epochs_weak',  type=int, default=30)

    parser.add_argument('--use_balanced_sampler', action='store_true', default=False)
    parser.add_argument('--distillation_loss_weight', type=float, default=0.0)
    parser.add_argument('--median_window', type=int, default=9)
    parser.add_argument('--weak_threshold', type=float, default=0.5)

    parser.add_argument('--training_type', type=str, choices=["strong", "weak-strong", "weak"],
                    default="weak-strong")

    # augmentation
    parser.add_argument('--wavmix_p', type=float, default=0.0)
    parser.add_argument('--freq_warp_p', type=float, default=0.0)
    parser.add_argument('--filter_augment_p', type=float, default=0.0)
    parser.add_argument('--frame_shift_range', type=float, default=0.125)  # in seconds
    parser.add_argument('--mixup_p_strong', type=float, default=0.2)
    parser.add_argument('--mixstyle_p_strong', type=float, default=0.2)

    parser.add_argument('--mixup_p_weak', type=float, default=0.5)
    parser.add_argument('--mixstyle_p_weak', type=float, default=0.3)

    parser.add_argument('--max_time_mask_size', type=float, default=0.1)

    # optimizer
    parser.add_argument('--no_adamw', action='store_true', default=False)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--transformer_frozen', action='store_true', dest='transformer_frozen',
                        default=False)

    # lr schedule
    parser.add_argument('--schedule_mode', type=str, default="cos")
    parser.add_argument('--max_lr_strong', type=float, default=3.06e-5)
    parser.add_argument('--max_lr_weak', type=float, default=3.06e-5)


    parser.add_argument('--transformer_lr', type=float, default=None)
    parser.add_argument('--lr_decay', type=float, default=0.6)
    parser.add_argument('--lr_end', type=float, default=2e-7)

    # paths
    parser.add_argument('--strong_ds_path', type=str, default=None)
    parser.add_argument('--weak_ds_path', type=str, default=None)
    parser.add_argument('--checkpoint_path', type=str, default=None)


    args = parser.parse_args()

    train(args)