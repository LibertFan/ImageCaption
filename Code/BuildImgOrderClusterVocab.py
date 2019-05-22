import sys
import argparse
import csv
import os
import pickle
from collections import Counter
import nltk
import numpy as np
sys.path.append('..')
from FeatureExtractorPytorch import ResNet152Extractor as extractor
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


class BuildVocab(object):
    def __init__(self, opts):
        self._options = opts
        self.model = extractor()
        self._GO, self._EOS, self._PAD = 'GO', 'EOS', 'PAD'
        self.spec_words = [self._PAD, self._GO, self._EOS]
        train_img_idx_sequence, train_filename_sequence, train_captions_sequence, data_split_sequence = \
            self.read_csv(opts.train_data_path)
        freq_words = self.get_freq_word(train_captions_sequence)
        self.idx2word, self.word2idx, self.word2vec, self.word_freq = self.load_word_info(freq_words)
        self.train_data = self.interpreter(opts.train_data_path, data_split='train')
        self.val_data = self.interpreter(opts.val_data_path, data_split='val')
        self.test_data = self.interpreter(opts.test_data_path, data_split='test')
        train_sequence_data = [self.train_data.get('Idx'), self.train_data.get('Caption'),
                               self.train_data.get('ImgFeature'), self.train_data.get('NumCaption')]
        val_sequence_data = [self.val_data.get('Idx'), self.val_data.get('Caption'),
                             self.val_data.get('ImgFeature'), self.val_data.get('NumCaption')]
        test_sequence_data = [self.test_data.get('Idx'), self.test_data.get('Caption'),
                              self.test_data.get('ImgFeature'), self.test_data.get('NumCaption')]
        self.save_pickle_file(train_sequence_data, os.path.join(opts.train_dir, opts.train_data_name))
        self.save_pickle_file(val_sequence_data, os.path.join(opts.train_dir, opts.val_data_name))
        self.save_pickle_file(test_sequence_data, os.path.join(opts.train_dir, opts.test_data_name))
        self.save_pickle_file([self.word_freq, self.idx2word, self.word2idx, self.word2vec],
                              os.path.join(opts.train_dir, opts.info_name))

    @staticmethod
    def read_csv(path):
        img_idx_sequence, filename_sequence, captions_sequence, data_split_sequence = [], [], [], []
        with open(path, 'r') as f:
            reader = csv.reader(f)
            index = 0
            for line in reader:
                if index == 0:
                    index += 1
                    continue
                else:
                    img_idx, filename, data_split, captions = line
                    captions = captions.split('---')
                    new_captions = []
                    for caption in captions:
                        token_caption = nltk.tokenize.word_tokenize(caption)
                        new_captions.append(token_caption)
                    captions_sequence.append(new_captions)
                    filename_sequence.append(filename)
                    img_idx_sequence.append(img_idx)
                    data_split_sequence.append(data_split)
        f.close()
        print('[The number of records in {}: {}]'.format(path, len(img_idx_sequence)))
        return img_idx_sequence, filename_sequence, captions_sequence, data_split_sequence

    @staticmethod
    def get_freq_word(question_list_list):
        raw_word_freq = Counter()
        for question_list in question_list_list:
            for question in question_list:
                raw_word_freq.update(question)
        freq_words = sorted(raw_word_freq.items(), key=lambda x: x[1], reverse=True)
        print('[FreqWord][The total of word in data set: {}][The number of word: {}][That of removed ones: {}]'.
              format(sum(raw_word_freq.values()), len(freq_words), len(raw_word_freq) - len(freq_words)))
        return freq_words

    def load_word_info(self, freq_words):
        opts = self._options
        raw_word2vec = dict()
        with open(opts.external_word_vectors_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                v = line.strip().split(" ")
                raw_word2vec[v[0]] = np.array([float(nv) for nv in v[1:]])
        f.close()
        common_id2word, word2vec, invalid_num, word_freq = [], [], 0, []
        invalid_word, invalid_counter = [], Counter()
        freq_words = [(self._PAD, 0), (self._GO, 0), (self._EOS, 0)] + freq_words
        for word, freq in freq_words:
            if raw_word2vec.get(word) is not None:
                common_id2word.append(word)
                word2vec.append(raw_word2vec[word])
                word_freq.append((word, freq))
            else:
                word_freq.append((word, freq))
                common_id2word.append(word)
                random_word2vec = 0.2 * (np.random.random(opts.word_emb_size) - 0.5)
                word2vec.append(random_word2vec)
                invalid_counter.update([freq])
                invalid_word.append(word)
                invalid_num += 1
        id2word = common_id2word
        id2word = {i: w for i, w in enumerate(id2word)}
        word2id = {w: i for i, w in id2word.items()}
        word2vec = np.array(word2vec)
        print('[Invalid number: {}, invalid_counter: {}]\n'
              '[The shape of word2vec: {}, the length of word2idx: {}, that of word_freq: {}]'.
              format(invalid_num, invalid_counter, word2vec.shape, len(word2id), len(word_freq)))
        return id2word, word2id, word2vec, word_freq

    def interpreter(self, path=None, data_split=None):
        img_idx_sequence, filename_sequence, captions_sequence, data_split_sequence = self.read_csv(path)
        num_captions_sequence = self.word2id_mapper(captions_sequence)
        print('[Interpreter][Word2id mapper has finished!]')
        print('[Size of file_sequence: {}][Size of image_idx_sequence: {}]'.
              format(len(filename_sequence), len(img_idx_sequence)))
        feature_sequence = self.get_image_feature(filename_sequence, data_split)
        print('[We have got image features!]')
        print('[ImgIdxSequence:{}][FileNameSequence:{}][CaptionSequence:{}][FeatureSequence:{}]]'
              '[NumCaptionsSequence:{}]'.format(len(img_idx_sequence), len(filename_sequence), len(captions_sequence),
                                                len(feature_sequence), len(num_captions_sequence)))
        data = {'Idx': filename_sequence, 'Caption': captions_sequence, 'ImgFeature': feature_sequence,
                'NumCaption': num_captions_sequence}
        return data

    def word2id_mapper(self, captions_sequence):
        opts = self._options
        word2idx = self.word2idx
        min_length = opts.min_sentence_length if opts.min_sentence_length is not None else 0
        num_captions_sequence = []
        for captions in captions_sequence:
            num_captions = []
            for caption in captions:
                num_caption = []
                for word in caption:
                    if word2idx.get(word) is not None:
                        num_caption.append(word2idx.get(word))
                if len(num_caption) < min_length:
                    continue
                num_captions.append(num_caption)
            num_captions_sequence.append(num_captions)
        return num_captions_sequence

    def get_image_feature(self, filename_sequence, data_split=None):
        opts = self._options
        file_path_sequence = []
        for file_name in filename_sequence:
            if data_split == 'train':
                sub_folder = 'train2014'
            elif data_split == 'val':
                sub_folder = 'val2014'
            elif data_split == 'test':
                sub_folder = 'val2014'
            else:
                raise Exception('DataSplit {} is illegal'.format(data_split))
            if data_split == 'train':
                if 'val' in file_name:
                    sub_folder = 'val2014'
            file_path = os.path.join(opts.image_dir, sub_folder+'/'+file_name)
            file_path_sequence.append(file_path)
        print('[FeatureExtractStart!][ImgFileSize:{}]'.format(len(file_path_sequence)))
        features = self.model.get_feature(file_path_sequence)
        return features

    @staticmethod
    def save_pickle_file(data, path):
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        f.close()


def read_commands():
    parser = argparse.ArgumentParser(usage='Pre-processing data set')
    root = os.path.abspath('..')
    raw_root = os.path.join(root, 'Data/raw')
    parser.add_argument('--data_name', type=str, default='coco')
    parser.add_argument('--data_id', type=int, default=100)
    parser.add_argument('--train_data_path', type=str, default=os.path.join(raw_root, 'coco/coco_train_v2.csv'))
    parser.add_argument('--val_data_path', type=str, default=os.path.join(raw_root, 'coco/coco_val_v2.csv'))
    parser.add_argument('--test_data_path', type=str, default=os.path.join(raw_root, 'coco/coco_test_v2.csv'))
    parser.add_argument('--image_dir', type=str, default=os.path.join(raw_root, 'img'))
    parser.add_argument('--external_word_vectors_file_path', type=str,
                        default=os.path.join(root, 'Data/word2vec/glove.6B.300d.txt'))
    parser.add_argument('--max_sentence_length', type=int, default=16)
    parser.add_argument('--word_emb_size', type=int, default=300)
    parser.add_argument('--min_sentence_length', type=int, default=0)
    parser.add_argument('--min_count', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--word_num', type=int, default=None)
    parser.add_argument('--min_freq', type=int, default=5)
    args = parser.parse_args()
    args.train_dir = os.path.join(root, 'Data/train/{}_v{}'.format(args.data_name, args.data_id))
    args.train_data_name = '{}_train_v{}.pkl'.format(args.data_name, args.data_id)
    args.val_data_name = '{}_val_v{}.pkl'.format(args.data_name, args.data_id)
    args.test_data_name = '{}_test_v{}.pkl'.format(args.data_name, args.data_id)
    args.info_name = '{}_info_v{}.pkl'.format(args.data_name, args.data_id)
    return args


def main():
    opts = read_commands()
    BuildVocab(opts)


if __name__ == '__main__':
    main()
