import sys
import pickle
import random
import numpy as np
import copy
sys.path.append('..')
from Evaluator.BLEU.bleu import BLEUEvaluator
from Evaluator.CIDEr.cider import CIDErEvaluator
from Evaluator.METEOR.meteor import METEOREvaluator
from Evaluator.ROUGE.rouge import RougeEvaluator


class DataStream(object):
    def __init__(self, opts):
        self._options = opts
        self.word_freq, self.id2word, self.word2id, self.raw_word2vec = self.get_vocab_info()
        (self._PAD, self.pad_id), (self._GO, self.go_id), (self._EOS, self.eos_id) = ("PAD", 0), ('GO', 1), ('EOS', 2)
        opts.pad_id, opts.go_id, opts.eos_id = self.pad_id, self.go_id, self.eos_id
        self.spec_tokens = [self._PAD, self._GO, self._EOS]
        opts.word_num = self.num_units = self.get_num_units()
        print('[Stream][Number of words number: {}]'.format(self.num_units))
        self.word2vec = self.raw_word2vec[:self.num_units]
        # load pickle file
        train_data_set, val_data_set, test_data_set = \
            list(map(self.load_pickle_file, [opts.train_data_path, opts.val_data_path, opts.test_data_path]))
        print('[Stream][Data has loaded]')
        self.raw_train_data_set = self.erase_pad(train_data_set)
        self.raw_val_data_set = self.erase_pad(val_data_set)
        self.raw_test_data_set = self.erase_pad(test_data_set)
        print('[Stream][ErasePad of caption has finished]')
        self.train_data_set = self.reshape(self.raw_train_data_set, plain=True, align=True)
        self.val_data_set = self.reshape(self.raw_val_data_set, plain=opts.data_plain, align=True, train=False)
        self.test_data_set = self.reshape(self.raw_test_data_set, plain=opts.data_plain, align=True, train=False)
        self.full_train_data_set = self.reshape(self.raw_train_data_set, plain=False, align=True, train=True)
        self.train_iter = self.val_iter = self.test_iter = self.train_full_iter = -1
        self.train_data_size, self.val_data_size, self.test_data_size, self.full_train_data_size = \
            len(self.train_data_set), len(self.val_data_set), len(self.test_data_set), len(self.full_train_data_set)
        self.evaluators = [BLEUEvaluator(4), METEOREvaluator(), RougeEvaluator(), CIDErEvaluator()]

    def get_num_units(self):
        opts = self._options
        if opts.word_num is not None:
            return opts.word_num
        elif opts.min_freq is not None:
            word_num = -1
            for i, (word, freq) in enumerate(self.word_freq):
                if i > len(self.spec_tokens) and freq < opts.min_freq:
                    word_num = i
                    break
            return word_num
        else:
            word_num = len(self.word_freq)
            return word_num

    @staticmethod
    def load_pickle_file(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        f.close()
        return data

    def erase_pad(self, data_set):
        opts, num_units, eos_id, pad_id = self._options, self.num_units, self.eos_id, self.pad_id
        max_length = opts.caption_size
        img_idx_seq, captions_seq, feature_seq, num_captions_seq = data_set

        def remove_pad(num_captions):
            removed_captions = []
            for num_caption in num_captions:
                removed_captions.append(remove_pad_single(num_caption))
            num_captions = np.array(removed_captions)
            return num_captions

        def remove_pad_single(num_caption):
            removed_caption = []
            for num in num_caption:
                if num < num_units:
                    removed_caption.append(num)
            if len(removed_caption) > max_length - 1:
                removed_caption = removed_caption[:max_length - 1] + [eos_id]
            else:
                pad_length = max_length - 1 - len(removed_caption)
                removed_caption = removed_caption + [eos_id] + [pad_id] * pad_length
            assert len(removed_caption) == max_length
            return removed_caption

        captions_sequence = [[' '.join(caption) for caption in captions] for captions in captions_seq]
        num_captions_sequence = np.array(list(map(remove_pad, num_captions_seq)))
        new_data_set = [img_idx_seq, captions_sequence, feature_seq, num_captions_sequence]
        return new_data_set

    @staticmethod
    def reshape(data_set, plain=False, align=False, train=True):
        records = []
        for img_idx, captions, feature, num_captions in \
                zip(data_set[0], data_set[1], data_set[2], data_set[3]):
            if plain:
                for caption, num_caption in zip(captions, num_captions):
                    records.append([img_idx, caption, feature, num_caption])
            else:
                if align and train:
                    records.append([img_idx, captions[:5], feature, num_captions[:5]])
                elif align:
                    records.append([img_idx, captions[:5], feature, num_captions[:5]])
                else:
                    records.append([img_idx, captions, feature, num_captions])
        return records

    @staticmethod
    def convert(records):
        column_num = len(records[0])
        data_set = [[] for _ in range(column_num)]
        for record in records:
            for i, v in enumerate(record):
                data_set[i].append(v)
        records = []
        for data in data_set:
            np_data = np.array(data)
            records.append(np_data)
        return records

    def get_next_train_batch(self):
        opts = self._options
        if self.train_iter == -1:
            random.shuffle(self.train_data_set)
        self.train_iter += 1
        start_idx = self.train_iter * opts.batch_size
        end_idx = (self.train_iter + 1) * opts.batch_size
        if start_idx >= self.train_data_size:
            self.train_iter = -1
            return self.get_next_train_batch()
        else:
            curr_train_batch = self.train_data_set[start_idx:end_idx]
            return self.convert(curr_train_batch)

    def get_next_val_batch(self):
        opts = self._options
        self.val_iter += 1
        start_idx = self.val_iter * opts.batch_size
        end_idx = (self.val_iter + 1) * opts.batch_size
        if start_idx >= self.val_data_size:
            self.val_iter = -1
            return None
        else:
            curr_val_batch = self.val_data_set[start_idx:end_idx]
            return self.convert(curr_val_batch)

    def get_next_test_batch(self):
        opts = self._options
        self.test_iter += 1
        start_idx = self.test_iter * opts.batch_size
        end_idx = (self.test_iter + 1) * opts.batch_size
        if start_idx >= self.test_data_size:
            self.test_iter = -1
            return None
        else:
            curr_test_batch = self.test_data_set[start_idx:end_idx]
            return self.convert(curr_test_batch)

    def get_next_full_train_batch(self):
        opts = self._options
        if self.train_full_iter == -1:
            random.shuffle(self.full_train_data_set)
        self.train_full_iter += 1
        start_idx = self.train_full_iter * opts.batch_size
        end_idx = (self.train_full_iter + 1) * opts.batch_size
        if end_idx >= self.full_train_data_size:
            self.train_full_iter = -1
            return self.get_next_full_train_batch()
        else:
            curr_train_batch = self.full_train_data_set[start_idx:end_idx]
            return self.convert(curr_train_batch)

    @staticmethod
    def data_shuffle(data):
        raw_idx = data[0]
        copy_data = copy.copy(data)
        num_seq = np.arange(len(raw_idx))
        while True:
            random.shuffle(num_seq)
            copy_idx = raw_idx[num_seq]
            sign = (raw_idx == copy_idx).astype(np.int32).sum()
            if sign == 0:
                break
        for i, rec in enumerate(copy_data):
            copy_data[i] = rec[num_seq]
        return copy_data

    def get_vocab_info(self):
        opts = self._options
        return self.load_pickle_file(opts.vocab_info_path)

    def tokens2sentence(self, tokens):
        gen_question = []
        for token in tokens:
            if token == self.pad_id or token == self.eos_id:
                break
            word = self.id2word.get(token)
            if word is None:
                break
            gen_question.append(word)
        if len(gen_question) == 0 or gen_question[-1] != '.':
            gen_question.append('.')
        return ' '.join(gen_question)

    def translate(self, gen_qs_tokens):
        sentences = []
        for qs_tokens in gen_qs_tokens:
            try:
                sentences.append(self.tokens2sentence(qs_tokens))
            except TypeError as e:
                print(qs_tokens)
                print(self.tokens2sentence(qs_tokens))
                raise e
        return sentences

    @staticmethod
    def build_hypo_ref(image_ids, hypos, refs):
        gen_qs_num = int(len(hypos)/len(refs))
        hypo_ref = [dict(), dict()]
        for n, image_id in enumerate(image_ids):
            image_name = str(image_id)
            hypo_ref[0][image_name] = []
            hypo_ref[1][image_name] = list(refs[n])
            for i in range(gen_qs_num):
                index = n * gen_qs_num + i
                hypo_ref[0][image_name].append(hypos[index])
        return hypo_ref

    def measure_score(self, hypos, refs, mode='all'):
        scores = []
        for evaluator in self.evaluators:
            scores.extend(evaluator.compute_score(refs, hypos, mode))
        score_names = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR', 'ROUGE', 'CIDEr']
        scores_dict = dict(zip(score_names, scores))
        return scores_dict

    def every_measure_score(self, hypos, refs):
        cider_eval = self.evaluators[-1]
        cider_scores = cider_eval.compute_score(refs, hypos, 'every')
        scores = np.squeeze(np.array(cider_scores))
        return scores

    def quick_measure_score(self, hypos, refs):
        meteor_score = self.evaluators[1].compute_score(refs, hypos)[0]
        cider_score = self.evaluators[3].compute_score(refs, hypos)[0]
        return {'METEOR': meteor_score, 'CIDEr': cider_score}

    @staticmethod
    def write_record(file_path, line):
        with open(file_path, 'a') as f:
            f.write(line)
        f.close()
