#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import traceback
import numpy as np
import itertools

import torch
from torch.utils.data import DataLoader

from TTS.speaker_encoder.dataset import SpeakerEncoderDataset

from TTS.speaker_encoder.losses import AngleProtoLoss, GE2ELoss, SoftmaxAngleProtoLoss
from TTS.speaker_encoder.utils.generic_utils import save_best_model, setup_model
from TTS.speaker_encoder.utils.reversal_classifier import ReversalClassifier

from TTS.speaker_encoder.utils.visual import plot_embeddings
from TTS.tts.datasets.preprocess import load_meta_data
from TTS.utils.arguments import init_training
from TTS.utils.audio import AudioProcessor
from TTS.utils.generic_utils import count_parameters, remove_experiment_folder, set_init_dict
from TTS.utils.radam import RAdam
from TTS.utils.training import NoamLR, check_update

from torch.utils.data.sampler import WeightedRandomSampler

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(54321)
use_cuda = torch.cuda.is_available()
num_gpus = torch.cuda.device_count()
print(" > Using CUDA: ", use_cuda)
print(" > Number of GPUs: ", num_gpus)


def setup_loader(ap: AudioProcessor, is_val: bool = False, verbose: bool = False):
    if is_val:
        loader = None
    else:
        dataset = SpeakerEncoderDataset(
            ap,
            meta_data_eval if is_val else meta_data_train,
            voice_len=c.voice_len,
            num_utter_per_speaker=c.num_utters_per_speaker,
            num_speakers_in_batch=c.num_speakers_in_batch,
            skip_speakers=c.skip_speakers,
            storage_size=c.storage["storage_size"],
            sample_from_storage_p=c.storage["sample_from_storage_p"],
            verbose=verbose,
            augmentation_config=c.audio_augmentation
        )
        # balance language considering the number of speakers 
        sampler = None
        language_speaker_freq = {}
        dataset_valid_speakers = dataset.speakers
        speaker_language_map = dataset.speaker_language_map
        language_speaker_freq = dataset.speakers_per_language_freq
        unique_language_names = language_speaker_freq.keys()
        # ignore this for single language train
        if len(unique_language_names)> 1:
            print(f"> Balancing the batch with {len(unique_language_names)} languages !")
            print("> Languages:", list(unique_language_names))
            print("> Num of Speakers by Language: ")
            for lang in language_speaker_freq:
                print('>    ', lang, ':', language_speaker_freq[lang])
            # print(len(speaker_language_map), len(dataset_valid_speakers))
            samples_weight = []
            for spk in dataset_valid_speakers:
                l = speaker_language_map[spk]
                samples_weight.append(1./language_speaker_freq[l])

            dataset_samples_weight = torch.from_numpy(np.array(samples_weight)).double()
            # create sampler
            sampler = WeightedRandomSampler(dataset_samples_weight, len(dataset_samples_weight))

        loader = DataLoader(
            dataset,
            batch_size=c.num_speakers_in_batch,
            shuffle=False,
            drop_last=True,
            num_workers=c.num_loader_workers,
            collate_fn=dataset.collate_fn,
            sampler=sampler
        )
    return loader, dataset.get_num_speakers(), len(unique_language_names)


def train(model, optimizer, scheduler, criterion, data_loader, global_step, reversal_classifier=None):
    model.train()
    epoch_time = 0
    best_loss = float("inf")
    avg_loss = 0
    avg_spk_loss = 0
    avg_lang_loss = 0
    avg_loss_epoch = 0
    avg_loader_time = 0
    end_time = time.time()
    for epoch in range(0, c.epochs):
        print(" EPOCH: ", epoch)
        for _, data in enumerate(data_loader):
            start_time = time.time()
            # setup input data
            inputs, labels, target_lang = data
            loader_time = time.time() - end_time
            global_step += 1

            # setup lr
            if c.lr_decay:
                scheduler.step()
            optimizer.zero_grad()

            # dispatch data to GPU
            if use_cuda:
                inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

            # forward pass model
            outputs = model(inputs)

            # loss computation
            spk_loss = criterion(outputs.view(c.num_speakers_in_batch, outputs.shape[0] // c.num_speakers_in_batch, -1), labels)
            avg_spk_loss = 0.01 * spk_loss.item() + 0.99 * avg_spk_loss if avg_spk_loss != 0 else spk_loss.item()
            if getattr(c, "use_reversal_language_classifier", False):
                if use_cuda:
                    target_lang = target_lang.cuda(non_blocking=True)
                target_lang = target_lang.reshape(-1)
                language_prediciton = reversal_classifier(outputs)
                lang_loss = torch.nn.functional.cross_entropy(language_prediciton, target_lang)
                lang_loss = lang_loss * getattr(c, "lang_loss_alpha", 1)
                loss = spk_loss + lang_loss 
                avg_lang_loss = 0.01 * lang_loss.item() + 0.99 * avg_lang_loss if avg_lang_loss != 0 else lang_loss.item()
            else:
                loss = spk_loss

            loss.backward()
            grad_norm, _ = check_update(model, c.grad_clip)
            optimizer.step()

            step_time = time.time() - start_time
            epoch_time += step_time

            # Averaged Loss and Averaged Loader Time
            avg_loss = 0.01 * loss.item() + 0.99 * avg_loss if avg_loss != 0 else loss.item()
            num_loader_workers = c.num_loader_workers if c.num_loader_workers > 0 else 1
            avg_loader_time = (
                1 / num_loader_workers * loader_time + (num_loader_workers - 1) / num_loader_workers * avg_loader_time
                if avg_loader_time != 0
                else loader_time
            )
            current_lr = optimizer.param_groups[0]["lr"]

            if global_step % c.steps_plot_stats == 0:
                # Plot Training Epoch Stats
                train_stats = {
                    "loss": avg_loss,
                    "lang_loss":avg_lang_loss,
                    "spk_loss": avg_spk_loss,
                    "lr": current_lr,
                    "grad_norm": grad_norm,
                    "step_time": step_time,
                    "avg_loader_time": avg_loader_time,
                }
                tb_logger.tb_train_epoch_stats(global_step, train_stats)
                figures = {
                    # FIXME: not constant
                    "UMAP Plot": plot_embeddings(outputs.detach().cpu().numpy(), 10),
                }
                tb_logger.tb_train_figures(global_step, figures)

            if global_step % c.print_step == 0:
                print(
                    "   | > Step:{}  Loss:{:.5f}  AvgLoss:{:.5f} AvgLangLoss: {:.5f} AvgSpeakerLoss: {:.5f} GradNorm:{:.5f}  "
                    "StepTime:{:.2f}  LoaderTime:{:.2f}  AvGLoaderTime:{:.2f}  LR:{:.6f}".format(
                        global_step, loss.item(), avg_loss, avg_lang_loss, avg_spk_loss, grad_norm, step_time, loader_time, avg_loader_time, current_lr
                    ),
                    flush=True,
                )
            avg_loss_epoch += avg_loss
            end_time = time.time()

        # save best model only
        best_loss = save_best_model(model, optimizer, criterion, avg_loss, best_loss, OUT_PATH, global_step)
        avg_loss_epoch = 0

    return avg_loss, global_step


def main(args):  # pylint: disable=redefined-outer-name
    # pylint: disable=global-variable-undefined
    global meta_data_train
    global meta_data_eval

    ap = AudioProcessor(**c.audio)
    model = setup_model(c)

    # pylint: disable=redefined-outer-name
    meta_data_train, meta_data_eval = load_meta_data(c.datasets, eval_split=False)

    data_loader, num_speakers, num_languages = setup_loader(ap, is_val=False, verbose=True)

    if getattr(c, "use_reversal_language_classifier", False):
        reversal_classifier = ReversalClassifier(c.model_params["proj_dim"], hidden_dim=256, output_dim=num_languages, gradient_clipping_bounds=0.25)         
        optimizer = RAdam(itertools.chain(model.parameters(), reversal_classifier.parameters()), lr=c.lr)
    else: 
        reversal_classifier = None
        optimizer = RAdam(model.parameters(), lr=c.lr)
    
    
    if c.loss == "ge2e":
        criterion = GE2ELoss(loss_method="softmax")
    elif c.loss == "angleproto":
        criterion = AngleProtoLoss()
    elif c.loss == "softmaxproto":
        print(c.model, num_speakers)
        criterion = SoftmaxAngleProtoLoss(c.model["proj_dim"], num_speakers)
    else:
        raise Exception("The %s  not is a loss supported" % c.loss)


    if args.restore_path:
        checkpoint = torch.load(args.restore_path)
        try:
            model.load_state_dict(checkpoint["model"])

            if 'criterion' in checkpoint:
                criterion.load_state_dict(checkpoint["criterion"])

        except (KeyError, RuntimeError):
            print(" > Partial model initialization.")
            model_dict = model.state_dict()
            model_dict = set_init_dict(model_dict, checkpoint['model'], c)
            model.load_state_dict(model_dict)
            del model_dict
        for group in optimizer.param_groups:
            group["lr"] = c.lr

        print(" > Model restored from step %d" % checkpoint["step"], flush=True)
        args.restore_step = checkpoint["step"]
    else:
        args.restore_step = 0

    if c.lr_decay:
        scheduler = NoamLR(optimizer, warmup_steps=c.warmup_steps, last_epoch=args.restore_step - 1)
    else:
        scheduler = None

    num_params = count_parameters(model)
    print("\n > Model has {} parameters".format(num_params), flush=True)


    if use_cuda:
        model = model.cuda()
        criterion.cuda()
        if reversal_classifier is not None:
            reversal_classifier = reversal_classifier.cuda()

    global_step = args.restore_step
    _, global_step = train(model, optimizer, scheduler, criterion, data_loader, global_step, reversal_classifier)


if __name__ == "__main__":
    args, c, OUT_PATH, AUDIO_PATH, c_logger, tb_logger = init_training(sys.argv)

    try:
        main(args)
    except KeyboardInterrupt:
        remove_experiment_folder(OUT_PATH)
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)  # pylint: disable=protected-access
    except Exception:  # pylint: disable=broad-except
        remove_experiment_folder(OUT_PATH)
        traceback.print_exc()
        sys.exit(1)
