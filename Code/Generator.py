import torch
import torch.nn.functional as func
import numpy as np


class Generate(torch.nn.Module):
    def __init__(self, opts, streamer):
        super(Generate, self).__init__()
        self._options = opts
        self.word2vec = streamer.word2vec
        self.translator = streamer.tokens2sentence
        self.measurer = streamer.every_measure_score
        self.img_encoder = ImgEncoder(opts)
        self.decoder = Decoder(opts, self.word2vec)
        self.beam_searcher = BeamSearch(opts)
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=opts.learning_rate)

    def forward(self, img, concept_score, vocab, caption=None, str_caption=None, mode='Generate_MLE'):
        self.train()
        opts = self._options
        caption = caption.view(-1, opts.caption_size)
        img_vec = self.img_encoder(img)
        if mode == 'Generate_MLE':
            if img_vec.size(0) < caption.size(0):
                img_vec_size = img_vec.size(-1)
                rep_size = caption.size(0) // img_vec.size(0)
                img_vec = img_vec.unsqueeze(1).repeat(1, rep_size, 1).view(-1, img_vec_size)
            init_vec = [img_vec]
            gen_captions, word_prob = self.professor_decode(init_vec, concept_score, caption)
            word_loss = self.solve_word_loss(caption, word_prob)
            reg_loss = self.solve_reg_loss()
            loss = word_loss + reg_loss
            self.backward(loss)
            if opts.display:
                print('[WordLoss:{:.5f}][RegLoss:{:.5f}][Loss:{:.5f}]'.format(word_loss, reg_loss, loss))
        else:
            assert str_caption is not None
            batch_size = len(str_caption)
            self.eval()
            greedy_result, greedy_prob = self.greedy_decode([img_vec], concept_score, vocab)
            self.train()
            random_result, random_prob = self.sample_decode([img_vec], concept_score, vocab, opts.sample_decode_size)
            result = torch.cat([greedy_result.view(batch_size, -1, opts.caption_size),
                                random_result.view(batch_size, -1, opts.caption_size)], 1)
            score = self.get_reward(result, str_caption).data
            base_score = score[:, 0]
            random_score = score[:, 1:].view(-1)
            if base_score.size(0) < random_score.size(0):
                rep_size = random_score.size(0) // base_score.size(0)
                reward = random_score.sub(base_score.repeat(rep_size))
            else:
                reward = random_score.sub(base_score)
            reinforce_loss = self.solve_reinforce_loss(random_prob, random_result, reward)
            reg_loss = self.solve_reg_loss()
            loss = reinforce_loss + reg_loss
            self.backward(loss)
            if opts.display:
                print('[ReinforceLoss:{:.5f}][RegLoss:{:.5f}][Loss:{:.5f}]'.
                      format(reinforce_loss, reg_loss, loss))

    def professor_decode(self, init_vec, concept_score, captions):
        opts = self._options
        assert isinstance(init_vec, list)
        captions = captions.view(-1, opts.caption_size)
        batch_size = captions.size(0)
        caption_sequence = captions.unbind(-1)
        cell_state = self.decoder.cell_init(batch_size, concept_score)
        word_prob_sequence = []
        for t in range(opts.caption_size+len(init_vec)):
            if t < len(init_vec):
                cell_input = init_vec[t]
                repeat_size = batch_size // cell_input.size(0)
                cell_input_size = cell_input.size(-1)
                cell_input = cell_input.unsqueeze(1).repeat(1, repeat_size, 1).view(-1, cell_input_size)
            elif t == len(init_vec):
                start_token = torch.ones(batch_size).long()
                if torch.cuda.is_available():
                    start_token = start_token.cuda()
                cell_input = start_token
            else:
                cell_input = caption_sequence[t-len(init_vec)-1]
            word_emb = False if t < len(init_vec) else True
            word_project = False if t < len(init_vec) else True
            word_prob, cell_state = self.decoder(
                cell_input, cell_state, word_emb=word_emb, word_project=word_project)
            if t >= len(init_vec):
                word_prob_sequence.append(word_prob)
        word_prob = torch.stack(word_prob_sequence, -1)
        return captions, word_prob

    def greedy_decode(self, init_vec, concept_score, vocab):
        opts = self._options
        assert isinstance(init_vec, list)
        batch_size = init_vec[0].size(0)
        cell_state = self.decoder.cell_init(batch_size, concept_score, vocab)
        word_score_sequence = []
        caption_sequence = []
        for t in range(opts.caption_size+len(init_vec)):
            if t < len(init_vec):
                cell_input = init_vec[t]
            elif t == len(init_vec):
                start_token = torch.ones(batch_size).long()
                if torch.cuda.is_available():
                    start_token = start_token.cuda()
                cell_input = start_token
            else:
                cell_input = caption_sequence[-1]
            word_emb = False if t < len(init_vec) else True
            word_project = False if t < len(init_vec) else True
            word_score, cell_state = self.decoder(
                cell_input, cell_state, word_emb=word_emb, word_project=word_project)
            if t >= len(init_vec):
                cell_word_score, cell_input = word_score.max(-1)
                caption_sequence.append(cell_input)
                word_score_sequence.append(cell_word_score)
        word_scores = torch.stack(word_score_sequence, -1)
        captions = torch.stack(caption_sequence, -1)
        masks = captions.le(2.5).float().cumsum(-1).cumsum(-1).le(1.5).long()
        captions = captions.mul(masks)
        word_scores = word_scores.mul(masks.float())
        return captions, word_scores

    def sample_decode(self, init_vec, concept_score, vocab, gen_size=1):
        opts = self._options
        assert isinstance(init_vec, list)
        batch_size = init_vec[0].size(0) * gen_size
        cell_state = self.decoder.cell_init(batch_size, concept_score, vocab)
        word_score_sequence = []
        caption_sequence = []
        for t in range(opts.caption_size+len(init_vec)):
            if t < len(init_vec):
                cell_input = init_vec[t]
                cell_input_size = cell_input.size(-1)
                cell_input = cell_input.unsqueeze(1).repeat(1, gen_size, 1).view(-1, cell_input_size)
            elif t == len(init_vec):
                start_token = torch.ones(batch_size).long()
                if torch.cuda.is_available():
                    start_token = start_token.cuda()
                cell_input = start_token
            else:
                cell_input = caption_sequence[-1]
            word_emb = False if t < len(init_vec) else True
            word_project = False if t < len(init_vec) else True
            word_score, cell_state = self.decoder(
                cell_input, cell_state, word_emb=word_emb, word_project=word_project)
            if t >= len(init_vec):
                word_prob = word_score.data.log().div(opts.temperature).exp()
                cell_input = torch.multinomial(word_prob, 1).data
                cell_word_score = word_score.gather(1, cell_input).squeeze(-1)
                word_score_sequence.append(cell_word_score)
                cell_input = cell_input.squeeze(-1)
                caption_sequence.append(cell_input)
        word_scores = torch.stack(word_score_sequence, -1)
        captions = torch.stack(caption_sequence, -1)
        masks = captions.le(2.5).float().cumsum(-1).cumsum(-1).le(1.5).long()
        captions = captions.mul(masks)
        word_scores = word_scores.mul(masks.float())
        return captions, word_scores

    def get_reward(self, captions, str_captions):
        opts = self._options
        captions = captions.view(-1, opts.caption_size)
        hypos, refs = dict(), dict()
        hypo_size = captions.size(0)
        ref_size = len(str_captions)
        group_size = hypo_size // ref_size
        for idx, num_caption in enumerate(captions.data.cpu().numpy()):
            gen_token = self.translator(num_caption)
            hypos[idx] = [gen_token]
            raw_caption = str_captions[idx // group_size]
            if isinstance(raw_caption, str):
                raw_caption = [raw_caption]
            elif not isinstance(raw_caption, list):
                raw_caption = list(raw_caption)
            refs[idx] = raw_caption
        scores = self.measurer(hypos, refs, bleu=opts.bleu, metric=opts.metric)
        scores = torch.from_numpy(scores).float()
        if torch.cuda.is_available():
            scores = scores.cuda()
        if group_size is not None:
            scores = scores.view(-1, group_size)
        return scores

    def backward(self, loss):
        opts = self._options
        optimizer = self.optimizer
        optimizer.zero_grad()
        loss.backward()
        if opts.grad_value_clip:
            torch.nn.utils.clip_grad_value_(self.parameters(), opts.grad_clip_value)
        elif opts.grad_norm_clip:
            torch.nn.utils.clip_grad_norm_(self.parameters(), opts.grad_norm_clip_value)
        optimizer.step()

    def generate(self, img, concept_score, vocab=None):
        self.eval()
        opts = self._options
        # iterative
        img_vec = self.img_encoder(img)
        init_vec = [img_vec]
        batch_size = img_vec.size(0)
        cell_state = self.decoder.cell_init(batch_size, concept_score, vocab)
        for vec in init_vec:
            if vec.size(0) < batch_size:
                group_size = batch_size // vec.size(0)
                vec_size = vec.size(-1)
                vec = vec.unsqueeze(1).repeat(1, group_size, 1).view(-1, vec_size)
            _, cell_state = self.decoder(vec, cell_state, False, False)
        start_token = torch.ones(batch_size).long()
        if torch.cuda.is_available():
            start_token = start_token.cuda()
        gen_tokens = self.beam_searcher.forward(
            opts.caption_size, self.decoder, start_token, cell_state)
        gen_tokens = gen_tokens.view(batch_size, opts.caption_size)
        return gen_tokens

    @staticmethod
    def solve_reinforce_loss(word_prob, caption, reward):
        mask = caption.le(2.5).float().cumsum(-1).le(1.5).float().data
        every_loss = word_prob.clamp(1e-10).log().mul(reward.unsqueeze(-1)).neg()
        loss = every_loss.mul(mask).sum(-1).div(mask.sum(-1)).mean(0)
        return loss

    @staticmethod
    def solve_word_loss(ref_words, word_prob):
        word_size = word_prob.size(-1)
        ref_words = ref_words.index_select(-1, index=torch.arange(word_size).long())
        word_loss = func.nll_loss(word_prob.log(), ref_words, reduction='none')
        mask = (ref_words > 0.5).float()
        word_loss = word_loss.mul(mask).sum().div(mask.sum())
        return word_loss

    def solve_reg_loss(self, scope=None):
        opts, l1_loss_sum, l2_loss_sum = self._options, 0.0, 0.0
        if scope is None:
            named_parameters = self.named_parameters()
        else:
            named_parameters = scope.named_parameters()
        for name, param in named_parameters:
            if 'word_emb' not in name:
                l1_loss_sum += param.abs().sum()
                l2_loss_sum += param.pow(2).sum()
        reg_loss = opts.l1_factor * l1_loss_sum + opts.l2_factor * l2_loss_sum
        return reg_loss


class ImgEncoder(torch.nn.Module):
    def __init__(self, opts):
        super(ImgEncoder, self).__init__()
        self._options = opts
        self.bn_l0 = torch.nn.BatchNorm1d(num_features=opts.img_size, eps=1e-05, momentum=0.1,
                                          affine=True, track_running_stats=True)
        self.fc_l1 = torch.nn.Linear(opts.img_size, opts.img_size, bias=False)
        self.bn_l1 = torch.nn.BatchNorm1d(num_features=opts.img_size, eps=1e-05, momentum=0.1,
                                          affine=True, track_running_stats=True)
        self.fc_l2 = torch.nn.Linear(opts.img_size, opts.word_emb_size, bias=False)
        self.bn_l2 = torch.nn.BatchNorm1d(num_features=opts.word_emb_size, eps=1e-05, momentum=0.1,
                                          affine=True, track_running_stats=True)
        self.var_init()

    def forward(self, img):
        img_vec = self.bn_l2(func.relu(self.fc_l2(
            self.bn_l1(func.relu(self.fc_l1(
                self.bn_l0(img)
            )))
        )))
        return img_vec

    def var_init(self):
        torch.nn.init.normal_(tensor=self.bn_l0.weight)
        torch.nn.init.constant_(tensor=self.bn_l0.bias, val=0.0)
        torch.nn.init.xavier_uniform_(tensor=self.fc_l1.weight)
        torch.nn.init.normal_(tensor=self.bn_l1.weight)
        torch.nn.init.constant_(tensor=self.bn_l1.bias, val=0.0)
        torch.nn.init.xavier_uniform_(tensor=self.fc_l2.weight)
        torch.nn.init.normal_(tensor=self.bn_l2.weight)
        torch.nn.init.constant_(tensor=self.bn_l2.bias, val=0.0)


class Decoder(torch.nn.Module):
    def __init__(self, opts, word2vec=None):
        super(Decoder, self).__init__()
        self._options = opts
        self.word2vec = word2vec
        self.word_emb = torch.nn.Embedding(opts.word_num, opts.word_emb_size)
        self.cell = SCNCore(opts, opts.word_emb_size, opts.rnn_size, opts.word_num)
        self.word_project = torch.nn.Linear(opts.rnn_size, opts.word_num, True)
        self.dropout = torch.nn.Dropout(opts.dropout_rate)
        self.concept_score = torch.zeros(opts.batch_size, opts.word_num)
        self.vocab = None
        self.var_init()

    def forward(self, cell_input, cell_state, word_emb=True, word_project=True):
        if word_emb:
            cell_input = self.word_emb(cell_input)
        cell_input = self.dropout(cell_input)
        cell_state = self.cell(cell_input, cell_state)
        if word_project:
            feature = cell_state[0]
            word_score = self.word_project(feature)
            if self.vocab is not None:
                vocab = self.vocab
                if feature.size(0) > vocab.size(0):
                    rep_size = feature.size(0) // vocab.size(0)
                    vocab_size = vocab.size(-1)
                    vocab = vocab.unsqueeze(1).repeat(1, rep_size, 1).view(-1, vocab_size)
                word_prob = func.softmax(word_score.masked_fill(vocab.le(0.5), -np.inf), -1).clamp(1e-20)
            else:
                word_prob = func.softmax(word_score, -1)
            return word_prob, cell_state
        return None, cell_state

    def var_init(self):
        if self.word2vec is not None:
            self.word_emb.from_pretrained(
                embeddings=torch.from_numpy(self.word2vec), freeze=False)
        else:
            torch.nn.init.uniform_(self.word_emb.weight, -0.08, 0.08)
        torch.nn.init.xavier_uniform_(tensor=self.word_project.weight)

    def cell_init(self, batch_size, concept, vocab=None):
        opts = self._options
        if vocab is not None:
            vocab = torch.zeros(batch_size, opts.word_num).scatter(1, vocab, 1).long()
        self.vocab = vocab
        return self.cell.cell_init(batch_size, concept)


class SCNCore(torch.nn.Module):
    def __init__(self, opts, input_size, hidden_size, concept_size):
        super(SCNCore, self).__init__()
        self.hidden_size = hidden_size
        self.mix_input_w = torch.nn.Linear(4*hidden_size, 4*hidden_size, True)
        self.mix_state_w = torch.nn.Linear(4*hidden_size, 4*hidden_size, True)
        self.input_w = torch.nn.Linear(input_size, 4*hidden_size, False)
        self.state_w = torch.nn.Linear(hidden_size, 4*hidden_size, False)
        self.concept_input_w = torch.nn.Linear(concept_size, 4*hidden_size, False)
        self.concept_state_w = torch.nn.Linear(concept_size, 4*hidden_size, False)
        self.concept = None
        self.dropout = torch.nn.Dropout(opts.dropout_rate)
        self.concept_dropout = torch.nn.Dropout(opts.concept_dropout_rate)
        self.var_init()

    def var_init(self):
        torch.nn.init.xavier_normal_(self.mix_input_w.weight)
        torch.nn.init.xavier_normal_(self.mix_state_w.weight)
        torch.nn.init.constant_(self.mix_input_w.bias, 0.0)
        torch.nn.init.constant_(self.mix_state_w.bias, 0.0)
        torch.nn.init.xavier_normal_(self.input_w.weight)
        torch.nn.init.xavier_normal_(self.state_w.weight)
        torch.nn.init.xavier_normal_(self.concept_input_w.weight)
        torch.nn.init.xavier_normal_(self.concept_state_w.weight)

    def forward(self, core_input, state):
        assert (isinstance(state, tuple) or isinstance(state, list)) and len(state) == 2
        h_state, c_state = state
        concept = self.concept
        if concept.size(0) < core_input.size(0):
            rep_size, concept_size = core_input.size(0) // concept.size(0), concept.size(-1)
            concept = concept.unsqueeze(-1).repeat(1, rep_size, 1).view(-1, concept_size)
        hidden_vec = self.concept_input_w(concept).mul(self.input_w(core_input)). \
            add(self.concept_state_w(concept).mul(self.state_w(h_state))).sigmoid()
        i_t, f_t, o_t, c_t = hidden_vec.split(hidden_vec.size(1) // 4, 1)
        c_t = i_t.mul(c_t).add(f_t.mul(c_state))
        h_t = o_t.mul(c_t.tanh())
        state = (h_t, c_t)
        return state

    def cell_init(self, batch_size, concept):
        self.concept = self.concept_dropout(concept.detach())
        return (torch.zeros(batch_size, self.hidden_size, requires_grad=True).float(),
                torch.zeros(batch_size, self.hidden_size, requires_grad=True).float())


class BeamSearch(object):
    def __init__(self, opts):
        self._options = opts
        self.word_length, self.stops, self.prob = None, None, None
        self.batch_size = None
        self.time = None
        self.prev_index_sequence = None

    def init(self, batch_size):
        self.batch_size = batch_size
        self.word_length = torch.zeros(batch_size).to(torch.int64)
        self.stops = torch.zeros(batch_size).to(torch.int64)
        self.prob = torch.ones(batch_size)
        self.prev_index_sequence = list()

    def forward(self, length, cell, word, state, **kwargs):
        self.init(word.size(0))
        word_list = []
        for i in range(length):
            self.time = i
            word_prob, next_state = cell(word, state)
            word, state = self.step(next_state, word_prob)
            word_list.append(word)
        word = self.get_output_words(word_list)
        return word

    def get_output_words(self, word_list):
        opts = self._options
        word_sequence = []
        index = torch.arange(self.batch_size).mul(opts.beam_size).long()
        prev_index_sequence = self.prev_index_sequence
        for word, prev_index in zip(word_list[::-1], prev_index_sequence[::-1]):
            output_word = word.index_select(0, index)
            index = prev_index.index_select(0, index)
            word_sequence.append(output_word)
        return torch.stack(word_sequence[::-1], 1)

    def step(self, next_state, word_prob):
        word_prob = self.solve_prob(word_prob)
        word_length = self.solve_length()
        next_word, prev_index = self.solve_score(word_prob, word_length)
        next_state = self.update(prev_index, next_word, next_state, word_prob)
        return next_word, next_state

    def solve_prob(self, word_prob):
        opts = self._options
        stops = self.stops
        stops = stops.unsqueeze(dim=-1)
        unstop_word_prob = torch.mul(word_prob, (1 - stops).float())
        batch_size = self.batch_size if self.time == 0 else self.batch_size * opts.beam_size
        pad = torch.tensor([[opts.pad_id]]).long()
        if torch.cuda.is_available():
            pad = pad.cuda()
        stop_prob = torch.zeros(1, opts.word_num).scatter_(1, pad, 1.0).repeat(batch_size, 1)
        stop_word_prob = stop_prob.mul(stops.float())
        word_prob = unstop_word_prob.add(stop_word_prob)
        prob = self.prob
        prob = prob.unsqueeze(-1)
        word_prob = prob.mul(word_prob)
        return word_prob

    def solve_length(self):
        opts, stops, word_length = self._options, self.stops, self.word_length
        stops = stops.unsqueeze(dim=-1)
        word_length = word_length.unsqueeze(dim=-1)
        batch_size = self.batch_size if self.time == 0 else self.batch_size * opts.beam_size
        pad = torch.tensor([[opts.eos_id, opts.pad_id]]).long()
        if torch.cuda.is_available():
            pad = pad.cuda()
        unstop_tokens = torch.ones(1, opts.word_num).scatter_(1, pad, 0.0).\
            repeat(batch_size, 1).long()
        add_length = unstop_tokens.mul(1 - stops)
        word_length = word_length.add(add_length)
        return word_length

    def solve_score(self, word_prob, word_length):
        opts = self._options
        beam_size = 1 if self.time == 0 else opts.beam_size
        length_penalty = ((word_length + 5).float().pow(opts.length_penalty_factor)).\
            div((torch.tensor([6.0])).pow(opts.length_penalty_factor))
        word_score = word_prob.clamp(1e-20, 1.0).log().div(length_penalty)
        # mini = word_score.min()
        word_score = word_score.view(-1, beam_size * opts.word_num)
        beam_score, beam_words = word_score.topk(opts.beam_size)
        prev_index = torch.arange(self.batch_size).long().mul(beam_size).view(-1, 1).\
            add(beam_words.div(opts.word_num)).view(-1)
        next_words = beam_words.fmod(opts.word_num).view(-1).long()
        self.prev_index_sequence.append(prev_index)
        return next_words, prev_index

    def update(self, index, word, state, prob):
        opts = self._options
        next_state = (state[0].index_select(0, index), state[1].index_select(0, index))
        self.stops = word.le(opts.eos_id).long()
        self.prob = prob.index_select(0, index).gather(1, word.view(-1, 1)).squeeze(1)
        self.word_length = self.word_length.gather(0, index).add(1-self.stops)
        return next_state
