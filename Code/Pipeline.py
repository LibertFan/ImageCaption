import torch
import torch.nn.functional as func
import numpy as np
import sys
import nltk
sys.path.append('../..')
from Retriever import Hierarchical as Retriever
from Generator import Generate as Generator


class Pipeline(torch.nn.Module):
    def __init__(self, opts, streamer):
        super(Pipeline, self).__init__()
        self._options = opts
        self.streamer = streamer
        word2vec = streamer.word2vec
        self.high_stat = self.set_high_words()
        high2vec = self.high_stat.get('High2Vec')
        opts.high_num, opts.high_emb_size = high2vec.shape
        opts.word_num, opts.word_emb_size = word2vec.shape
        print('[Word2Vec:{}]'.format(self.streamer.word2vec.shape))
        print('[High2Vec:{}]'.format(high2vec.shape))
        self.retriever = Retriever(opts, high2vec, word2vec)
        self.generator = Generator(opts, streamer)

    def set_high_words(self):
        opts = self._options
        from nltk.corpus import stopwords
        stopwords = stopwords.words('english')
        opts.high_freq_num = 400
        streamer = self.streamer
        high_freq_words = []
        for i, (word, freq) in enumerate(streamer.word_freq):
            tag = nltk.pos_tag([word])[0][-1]
            if freq >= opts.high_freq_num and word not in stopwords and tag[0] in ['N', 'V', 'J', 'R']:
                high_freq_words.append(word)
        high_indices = [self.streamer.word2id.get(word) for word in high_freq_words]
        high2idx = {i: idx for i, idx in enumerate(high_indices)}
        idx2high = {idx: i for i, idx in high2idx.items()}
        high2vec = np.array([self.streamer.word2vec[idx] for idx in high_indices])
        high_stat = {'Idx2High': idx2high, 'High2Idx': high2idx, 'High2Vec': high2vec}
        print('[SizeOfHighFreqWords:{}]'.format(len(high_indices)))
        return high_stat

    def solve_words(self, batch_captions):
        idx2high = self.high_stat.get('Idx2High')
        high_words, whole_words = [], []
        high_size, whole_size = 0, 0
        for captions in batch_captions:
            if isinstance(captions[0], list) or isinstance(captions[0], np.ndarray):
                words = set()
                for sentence in captions:
                    words = words | set(sentence)
            else:
                words = set(captions)
            whole_words.append(list(words))
            high_word = []
            for word in list(words):
                idx = idx2high.get(word)
                if idx is not None:
                    high_word.append(idx)
            high_words.append(high_word)
            if len(high_word) > high_size:
                high_size = len(high_word)
            if len(words) > whole_size:
                whole_size = len(words)
        for i, (high_word, whole_word) in enumerate(zip(high_words, whole_words)):
            high_word = high_word + [high_word[-1]] * (high_size - len(high_word))
            whole_word = whole_word + [0] * (whole_size - len(whole_word))
            high_words[i] = high_word
            whole_words[i] = whole_word
        return np.array(high_words).astype(np.int64), np.array(whole_words).astype(np.int64)

    def forward(self, img, caption, str_caption=None, mode=None):
        opts = self._options
        if mode == 'Retrieve':
            img = torch.from_numpy(img).float()
            if torch.cuda.is_available():
                img = img.cuda()
            self_high_vocabs, self_vocabs = self.solve_words(caption)
            self_high_vocabs = torch.from_numpy(self_high_vocabs).long()
            self_vocabs = torch.from_numpy(self_vocabs).long()
            if torch.cuda.is_available():
                self_high_vocabs = self_high_vocabs.cuda()
                self_vocabs = self_vocabs.cuda()
            self.retriever.train()
            self.retriever(img, self_high_vocabs, self_vocabs)
        elif mode.startswith('Generate'):
            img = torch.from_numpy(img).float()
            caption = torch.from_numpy(caption).long()
            if torch.cuda.is_available():
                img = img.cuda()
                caption = caption.cuda()
            self.retriever.eval()
            concept_score = self.retriever.full_forward(img)
            vocab = concept_score.topk(opts.select_word_size, -1)[1]
            self.generator.train()
            self.generator(img, concept_score, vocab, caption, str_caption, mode=mode)
        else:
            raise Exception('Current Mode is not support')

    def eval_retriever(self, img, caption, ret_word=False):
        self.retriever.eval()
        img = torch.from_numpy(img).float()
        if torch.cuda.is_available():
            img = img.cuda()
        if not ret_word:
            self_high_vocabs, self_vocabs = self.solve_words(caption)
            stat = self.retriever.predict(img, self_high_vocabs, self_vocabs, all_ret=False)
            return stat
        else:
            sel_high_words, sel_words = self.retriever.predict(img)
            return sel_high_words, sel_words

    def eval_generator(self, img, caption=None):
        opts = self._options
        self.generator.eval()
        self.retriever.eval()
        img = torch.from_numpy(img).float()
        if torch.cuda.is_available():
            img = img.cuda()
        concept_score = self.retriever.full_forward(img)
        vocab = concept_score.topk(opts.select_word_size, -1)[1]
        gen_captions = self.generator.generate(img, concept_score)
        sel_gen_captions = self.generator.generate(img, concept_score, vocab)
        return sel_gen_captions, gen_captions
