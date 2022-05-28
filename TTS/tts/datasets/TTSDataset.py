import collections
import os
import random
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
from traceback import print_tb
from typing import Dict, List
import sys

import librosa
import soundfile as sf
import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset
from scipy.io import wavfile

from TTS.tts.utils.data import prepare_data, prepare_stop_target, prepare_tensor
from TTS.tts.utils.text import pad_with_eos_bos, phoneme_to_sequence, text_to_sequence
from TTS.utils.audio import AudioProcessor


class TTSDataset(Dataset):
    def __init__(
        self,
        outputs_per_step: int,
        text_cleaner: list,
        compute_linear_spec: bool,
        ap: AudioProcessor,
        meta_data: List[List],
        characters: Dict = None,
        custom_symbols: List = None,
        add_blank: bool = False,
        return_wav: bool = False,
        batch_group_size: int = 0,
        min_seq_len: int = 0,
        max_seq_len: int = float("inf"),
        use_phonemes: bool = False,
        use_IPAg2p_phonemes: bool = False,
        use_accent_info: bool = False,
        phoneme_cache_path: str = None,
        phoneme_language: str = "en-us",
        enable_eos_bos: bool = False,
        speaker_id_mapping: Dict = None,
        d_vector_mapping: Dict = None,
        language_id_mapping: Dict = None,
        use_noise_augment: bool = False,
        verbose: bool = False,
    ):
        """Generic 📂 data loader for `tts` models. It is configurable for different outputs and needs.

        If you need something different, you can either override or create a new class as the dataset is
        initialized by the model.

        Args:
            outputs_per_step (int): Number of time frames predicted per step.

            text_cleaner (list): List of text cleaners to clean the input text before converting to sequence IDs.

            compute_linear_spec (bool): compute linear spectrogram if True.

            ap (TTS.tts.utils.AudioProcessor): Audio processor object.

            meta_data (list): List of dataset instances.

            characters (dict): `dict` of custom text characters used for converting texts to sequences.

            custom_symbols (list): List of custom symbols used for converting texts to sequences. Models using its own
                set of symbols need to pass it here. Defaults to `None`.

            add_blank (bool): Add a special `blank` character after every other character. It helps some
                models achieve better results. Defaults to false.

            return_wav (bool): Return the waveform of the sample. Defaults to False.

            batch_group_size (int): Range of batch randomization after sorting
                sequences by length. It shuffles each batch with bucketing to gather similar lenght sequences in a
                batch. Set 0 to disable. Defaults to 0.

            min_seq_len (int): Minimum input sequence length to be processed
                by the loader. Filter out input sequences that are shorter than this. Some models have a
                minimum input length due to its architecture. Defaults to 0.

            max_seq_len (int): Maximum input sequence length. Filter out input sequences that are longer than this.
                It helps for controlling the VRAM usage against long input sequences. Especially models with
                RNN layers are sensitive to input length. Defaults to `Inf`.

            use_phonemes (bool): If true, input text converted to phonemes. Defaults to false.

            phoneme_cache_path (str): Path to cache phoneme features. It writes computed phonemes to files to use in
                the coming iterations. Defaults to None.

            phoneme_language (str): One the languages from supported by the phonemizer interface. Defaults to `en-us`.

            enable_eos_bos (bool): Enable the `end of sentence` and the `beginning of sentences characters`. Defaults
                to False.

            speaker_id_mapping (dict): Mapping of speaker names to IDs used to compute embedding vectors by the
                embedding layer. Defaults to None.

            d_vector_mapping (dict): Mapping of wav files to computed d-vectors. Defaults to None.

            use_noise_augment (bool): Enable adding random noise to wav for augmentation. Defaults to False.

            verbose (bool): Print diagnostic information. Defaults to false.
        """
        super().__init__()
        self.batch_group_size = batch_group_size
        self.items = meta_data
        self.outputs_per_step = outputs_per_step
        self.sample_rate = ap.sample_rate
        self.cleaners = text_cleaner
        self.compute_linear_spec = compute_linear_spec
        self.return_wav = return_wav
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.ap = ap
        self.characters = characters
        self.custom_symbols = custom_symbols
        self.add_blank = add_blank
        self.use_phonemes = use_phonemes
        self.use_IPAg2p_phonemes = use_IPAg2p_phonemes
        self.use_accent_info = use_accent_info
        self.phoneme_cache_path = phoneme_cache_path
        self.phoneme_language = phoneme_language
        self.enable_eos_bos = enable_eos_bos
        self.speaker_id_mapping = speaker_id_mapping
        self.d_vector_mapping = d_vector_mapping
        self.language_id_mapping = language_id_mapping
        self.use_noise_augment = use_noise_augment

        self.verbose = verbose
        self.input_seq_computed = False
        self.rescue_item_idx = 1
        if use_phonemes and not os.path.isdir(phoneme_cache_path):
            os.makedirs(phoneme_cache_path, exist_ok=True)
        if self.verbose:
            print("\n > DataLoader initialization")
            print(" | > Use phonemes: {}".format(self.use_phonemes))
            if use_phonemes:
                print("   | > phoneme language: {}".format(phoneme_language))
            print(" | > Number of instances : {}".format(len(self.items)))

    @staticmethod
    def load_wav_faster(
        filename,
        resample, sample_rate, do_trim_silence, do_sound_norm,
        trim_db, win_length, hop_length,
        sr=None
    ):
        # apを介さないやり方
        # staticmethod, つまり self を利用しないので，おそらく multiprocess を行うときに高速になる
        if resample:
            x, sr = librosa.load(filename, sr=sample_rate)
        elif sr is None:
            x, sr = sf.read(filename)
            assert sample_rate == sr, "%s vs %s" % (sample_rate, sr)
        else:
            x, sr = librosa.load(filename, sr=sr)
        if do_trim_silence:
            try:
                margin = int(sample_rate * 0.01)
                x = x[margin:-margin]
                x = librosa.effects.trim(x, top_db=trim_db, frame_length=win_length, hop_length=hop_length)[0]
            except ValueError:
                print(f" [!] File cannot be trimmed for silence - {filename}")
        if do_sound_norm:
            x = AudioProcessor.sound_norm(x)
        return x

    def load_wav(self, filename):
        audio = self.ap.load_wav(filename)
        return audio

    @staticmethod
    def load_np(filename):
        data = np.load(filename).astype("float32")
        return data

    @staticmethod
    def _generate_and_cache_phoneme_sequence(
        text, accent, cache_path, cache_path_accent, cleaners, language, custom_symbols, characters, add_blank, use_IPAg2p_phonemes
    ):
        """generate a phoneme sequence from text.
        since the usage is for subsequent caching, we never add bos and
        eos chars here. Instead we add those dynamically later; based on the
        config option."""
        phonemes, accents = phoneme_to_sequence(
            text,
            accent,
            [cleaners],
            language=language,
            use_IPAg2p_phonemes=use_IPAg2p_phonemes,
            enable_eos_bos=False,
            custom_symbols=custom_symbols,
            tp=characters,
            add_blank=add_blank,
        )
        phonemes = np.asarray(phonemes, dtype=np.int32)
        accents = np.asarray(accents, dtype=np.int32)
        np.save(cache_path, phonemes)
        np.save(cache_path_accent, accents)
        return phonemes, accents

    @staticmethod
    def _load_or_generate_phoneme_sequence(
        wav_file, text, accent, phoneme_cache_path, enable_eos_bos,
        cleaners, language, custom_symbols, characters, add_blank, use_IPAg2p_phonemes
    ):
        file_name = os.path.splitext(os.path.basename(wav_file))[0]

        # different names for normal phonemes and with blank chars.
        file_name_ext = "_blanked_phoneme.npy" if add_blank else "_phoneme.npy"
        cache_path = os.path.join(phoneme_cache_path, file_name + file_name_ext)
        cache_path_accent = os.path.join(phoneme_cache_path, file_name + "_accent" + file_name_ext)
        try:
            phonemes = np.load(cache_path)
            accents = np.load(cache_path_accent)
        except FileNotFoundError:
            phonemes, accents = TTSDataset._generate_and_cache_phoneme_sequence(
                text, accent, cache_path, cache_path_accent, cleaners, language, custom_symbols, characters, add_blank, use_IPAg2p_phonemes
            )
        except (ValueError, IOError):
            print(" [!] failed loading phonemes for {}. " "Recomputing.".format(wav_file))
            phonemes, accents = TTSDataset._generate_and_cache_phoneme_sequence(
                text, accent, cache_path, cache_path_accent, cleaners, language, custom_symbols, characters, add_blank, use_IPAg2p_phonemes
            )
        if enable_eos_bos:
            phonemes = pad_with_eos_bos(phonemes, tp=characters)
            phonemes = np.asarray(phonemes, dtype=np.int32)
        return phonemes, accents

    def load_data(self, idx):
        item = self.items[idx]

        if len(item) == 6:
            text, accent, wav_file, speaker_name, language_name, attn_file = item
            wav_idx = 2
        elif len(item) == 5:
            if self.use_accent_info:
                text, accent, wav_file, speaker_name, language_name = item
                attn = None
                wav_idx = 2
            else:
                text, wav_file, speaker_name, language_name, attn_file = item
                accent = None
                wav_idx = 1
        else:
            text, wav_file, speaker_name, language_name = item
            accent = None
            attn = None
            wav_idx = 1

        raw_text = text

        wav = np.asarray(self.load_wav(wav_file), dtype=np.float32)

        # apply noise for augmentation
        if self.use_noise_augment:
            wav = wav + (1.0 / 32768.0) * np.random.rand(*wav.shape)

        if not self.input_seq_computed:
            if self.use_phonemes:
                text, accent = self._load_or_generate_phoneme_sequence(
                    wav_file,
                    text,
                    accent,
                    self.phoneme_cache_path,
                    self.enable_eos_bos,
                    self.cleaners,
                    language_name if language_name else self.phoneme_language,
                    self.custom_symbols,
                    self.characters,
                    self.add_blank,
                    self.use_IPAg2p_phonemes
                )
            else:
                text = np.asarray(
                    text_to_sequence(
                        text,
                        [self.cleaners],
                        custom_symbols=self.custom_symbols,
                        tp=self.characters,
                        add_blank=self.add_blank,
                    ),
                    dtype=np.int32,
                )

        assert text.size > 0, self.items[idx][wav_idx]
        assert wav.size > 0, self.items[idx][wav_idx]

        if "attn_file" in locals():
            attn = np.load(attn_file)

        if len(text) > self.max_seq_len:
            # return a different sample if the phonemized
            # text is longer than the threshold
            # TODO: find a better fix
            return self.load_data(self.rescue_item_idx)

        sample = {
            "raw_text": raw_text,
            "text": text,
            "wav": wav,
            "accent": accent,
            "attn": attn,
            "item_idx": self.items[idx][wav_idx],
            "speaker_name": speaker_name,
            "language_name": language_name,
            "wav_file_name": os.path.basename(wav_file),
        }
        return sample

    @staticmethod
    def _phoneme_worker(args):
        item = args[0]
        func_args = args[1]
        if len(item) == 4:
            text, wav_file, _, lang = item
            accent = None
        elif len(item) == 5:
            text, accent, wav_file, _, lang = item
        else:
            raise ValueError("itemの数がおかしいです。formattersを確認してください")
        func_args[3] = lang  # 言語はこれを使うときに"self.phoneme_language"になってしまっているので更新。ここ以前で受け取れず。
        phonemes, _ = TTSDataset._load_or_generate_phoneme_sequence(wav_file, text, accent, *func_args)
        return phonemes

    def compute_input_seq(self, num_workers=0):
        """compute input sequences separately. Call it before
        passing dataset to data loader."""
        if not self.use_phonemes:
            if self.verbose:
                print(" | > Computing input sequences ...")
            for idx, item in enumerate(tqdm.tqdm(self.items, file=sys.stdout)):
                text, *_ = item
                sequence = np.asarray(
                    text_to_sequence(
                        text,
                        [self.cleaners],
                        custom_symbols=self.custom_symbols,
                        tp=self.characters,
                        add_blank=self.add_blank,
                    ),
                    dtype=np.int32,
                )
                self.items[idx][0] = sequence

        else:
            func_args = [
                self.phoneme_cache_path,
                self.enable_eos_bos,
                self.cleaners,
                self.phoneme_language,
                self.custom_symbols,
                self.characters,
                self.add_blank,
                self.use_IPAg2p_phonemes,
            ]
            if self.verbose:
                print(" | > Computing phonemes ...")
            if num_workers == 0:
                for idx, item in enumerate(tqdm.tqdm(self.items, file=sys.stdout)):
                    phonemes = self._phoneme_worker([item, func_args])
                    self.items[idx][0] = phonemes
            else:
                with Pool(num_workers) as p:
                    phonemes = list(
                        tqdm.tqdm(
                            p.imap(TTSDataset._phoneme_worker, [[item, func_args] for item in self.items]),
                            total=len(self.items),
                            file=sys.stdout
                        )
                    )
                    for idx, p in enumerate(phonemes):
                        self.items[idx][0] = p

    @staticmethod
    def _get_mel_length(args):
        item = args[0]
        func_args = args[1]
        (
            resample, sample_rate, do_trim_silence, do_sound_norm,
            trim_db, win_length, hop_length, phoneme_cache_path
        ) = func_args

        _, _, wav_file, spk, _ = item

        file_name = os.path.splitext(os.path.basename(wav_file))[0]
        file_name_ext = "_mel_length.npy"
        cache_path = os.path.join(phoneme_cache_path, spk + "_" + file_name + file_name_ext)
        try:
            length = np.load(cache_path, allow_pickle=True)
        except FileNotFoundError:
            # 本来、ここで用意されているwav_loaderではtrim silence などを行っている。
            # なので、単にreadするだけだと長さは正確に一致しないことに注意。
            # 詳細; TTS/utils/audio.py
            wav = TTSDataset.load_wav_faster(
                wav_file, resample, sample_rate, do_trim_silence, do_sound_norm,
                trim_db, win_length, hop_length
            )
            length = np.array([np.asarray(wav, dtype=np.float32).shape[0]//hop_length])
            np.save(cache_path, length)
        return (
            length[0],
            item
        )

    def sort_items(self, num_workers=0):
        r"""Sort instances based on text length in ascending order"""
        # 本当にmelを計算する必要は一切ない．計算後の長さはhop_lengthで割るだけ
        lengths = []
        new_items = []
        ignored_cnt = 0

        func_args = [
            self.ap.resample,
            self.ap.sample_rate,
            self.ap.do_trim_silence,
            self.ap.do_sound_norm,
            self.ap.trim_db,
            self.ap.win_length,
            self.ap.hop_length,
            self.phoneme_cache_path,
        ]

        if num_workers == 0:
            for idx, item in enumerate(tqdm.tqdm(self.items, file=sys.stdout)):
                l, item = TTSDataset._get_mel_length([item, func_args])
                if (l < self.min_seq_len) or (l > self.max_seq_len):
                    ignored_cnt += 1
                else:
                    new_items.append(item)
                    lengths.append(l)
        else:
            with Pool(num_workers) as p:
                _lengths = list(
                    tqdm.tqdm(
                        p.imap(TTSDataset._get_mel_length, [[item, func_args] for item in self.items]),
                        total=len(self.items),
                        file=sys.stdout
                    )
                )
                for l, item in _lengths:
                    if (l < self.min_seq_len) or (l > self.max_seq_len):
                        ignored_cnt += 1
                    else:
                        new_items.append(item)
                        lengths.append(l)

        idxs = np.argsort(lengths)
        new_items = [new_items[idx] for idx in idxs]
        # shuffle batch groups
        if self.batch_group_size > 0:
            for i in range(len(new_items) // self.batch_group_size):
                offset = i * self.batch_group_size
                end_offset = offset + self.batch_group_size
                temp_items = new_items[offset:end_offset]
                random.shuffle(temp_items)
                new_items[offset:end_offset] = temp_items
        self.items = new_items

        if self.verbose:
            print(" | > Max length sequence: {}".format(np.max(lengths)))
            print(" | > Min length sequence: {}".format(np.min(lengths)))
            print(" | > Avg length sequence: {}".format(np.mean(lengths)))
            print(
                " | > Num. instances discarded by max-min (max={}, min={}) seq limits: {}".format(
                    self.max_seq_len, self.min_seq_len, ignored_cnt
                )
            )
            print(" | > Batch group size: {}.".format(self.batch_group_size))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.load_data(idx)

    def collate_fn(self, batch):
        r"""
        Perform preprocessing and create a final data batch:
        1. Sort batch instances by text-length
        2. Convert Audio signal to Spectrograms.
        3. PAD sequences wrt r.
        4. Load to Torch.
        """

        # Puts each data field into a tensor with outer dimension batch size
        if isinstance(batch[0], collections.abc.Mapping):

            text_lenghts = np.array([len(d["text"]) for d in batch])

            # sort items with text input length for RNN efficiency
            text_lenghts, ids_sorted_decreasing = torch.sort(torch.LongTensor(text_lenghts), dim=0, descending=True)

            wav = [batch[idx]["wav"] for idx in ids_sorted_decreasing]
            item_idxs = [batch[idx]["item_idx"] for idx in ids_sorted_decreasing]
            text = [batch[idx]["text"] for idx in ids_sorted_decreasing]
            accent = [batch[idx]["accent"] for idx in ids_sorted_decreasing]
            raw_text = [batch[idx]["raw_text"] for idx in ids_sorted_decreasing]

            speaker_names = [batch[idx]["speaker_name"] for idx in ids_sorted_decreasing]

            # get language ids from language names
            if self.language_id_mapping is not None:
                language_names = [batch[idx]["language_name"] for idx in ids_sorted_decreasing]
                language_ids = [self.language_id_mapping[ln] for ln in language_names]
            else:
                language_ids = None
            # get pre-computed d-vectors
            if self.d_vector_mapping is not None:
                wav_files_names = [batch[idx]["wav_file_name"] for idx in ids_sorted_decreasing]
                d_vectors = [self.d_vector_mapping[w]["embedding"] for w in wav_files_names]
            else:
                d_vectors = None
            # get numerical speaker ids from speaker names
            if self.speaker_id_mapping:
                speaker_ids = [self.speaker_id_mapping[sn] for sn in speaker_names]
            else:
                speaker_ids = None
            # compute features
            mel = [self.ap.melspectrogram(w).astype("float32") for w in wav]

            mel_lengths = [m.shape[1] for m in mel]

            # lengths adjusted by the reduction factor
            mel_lengths_adjusted = [
                m.shape[1] + (self.outputs_per_step - (m.shape[1] % self.outputs_per_step))
                if m.shape[1] % self.outputs_per_step
                else m.shape[1]
                for m in mel
            ]

            # compute 'stop token' targets
            stop_targets = [np.array([0.0] * (mel_len - 1) + [1.0]) for mel_len in mel_lengths]

            # PAD stop targets
            stop_targets = prepare_stop_target(stop_targets, self.outputs_per_step)

            # PAD sequences with longest instance in the batch
            text = prepare_data(text).astype(np.int32)
            accent = prepare_data(accent).astype(np.int32) if accent[0] is not None else accent

            # PAD features with longest instance
            mel = prepare_tensor(mel, self.outputs_per_step)

            # B x D x T --> B x T x D
            mel = mel.transpose(0, 2, 1)

            # convert things to pytorch
            text_lenghts = torch.LongTensor(text_lenghts)
            text = torch.LongTensor(text)
            accent = torch.LongTensor(accent)
            mel = torch.FloatTensor(mel).contiguous()
            mel_lengths = torch.LongTensor(mel_lengths)
            stop_targets = torch.FloatTensor(stop_targets)

            if d_vectors is not None:
                d_vectors = torch.FloatTensor(d_vectors)

            if speaker_ids is not None:
                speaker_ids = torch.LongTensor(speaker_ids)

            if language_ids is not None:
                language_ids = torch.LongTensor(language_ids)

            # compute linear spectrogram
            if self.compute_linear_spec:
                linear = [self.ap.spectrogram(w).astype("float32") for w in wav]
                linear = prepare_tensor(linear, self.outputs_per_step)
                linear = linear.transpose(0, 2, 1)
                assert mel.shape[1] == linear.shape[1]
                linear = torch.FloatTensor(linear).contiguous()
            else:
                linear = None

            # format waveforms
            wav_padded = None
            if self.return_wav:
                wav_lengths = [w.shape[0] for w in wav]
                max_wav_len = max(mel_lengths_adjusted) * self.ap.hop_length
                wav_lengths = torch.LongTensor(wav_lengths)
                wav_padded = torch.zeros(len(batch), 1, max_wav_len)
                for i, w in enumerate(wav):
                    mel_length = mel_lengths_adjusted[i]
                    w = np.pad(w, (0, self.ap.hop_length * self.outputs_per_step), mode="edge")
                    w = w[: mel_length * self.ap.hop_length]
                    wav_padded[i, :, : w.shape[0]] = torch.from_numpy(w)
                wav_padded.transpose_(1, 2)

            # collate attention alignments
            if batch[0]["attn"] is not None:
                attns = [batch[idx]["attn"].T for idx in ids_sorted_decreasing]
                for idx, attn in enumerate(attns):
                    pad2 = mel.shape[1] - attn.shape[1]
                    pad1 = text.shape[1] - attn.shape[0]
                    attn = np.pad(attn, [[0, pad1], [0, pad2]])
                    attns[idx] = attn
                attns = prepare_tensor(attns, self.outputs_per_step)
                attns = torch.FloatTensor(attns).unsqueeze(1)
            else:
                attns = None
            # TODO: return dictionary
            return (
                text,
                text_lenghts,
                accent,
                speaker_names,
                linear,
                mel,
                mel_lengths,
                stop_targets,
                item_idxs,
                d_vectors,
                speaker_ids,
                attns,
                wav_padded,
                raw_text,
                language_ids,
            )

        raise TypeError(
            (
                "batch must contain tensors, numbers, dicts or lists;\
                         found {}".format(
                    type(batch[0])
                )
            )
        )
