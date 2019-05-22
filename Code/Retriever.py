import torch
import torch.nn.functional as func
import numpy as np


# 在这个模型当中，我们仍旧使用层次化的抽取模型。
# 第一层抽取模型抽取高频词，同时获得在所有高频词上的得分
# 第二层抽取模型在所有高频词的得分的基础上再做一次抽取
# 第一层加入到第二层里面的信息为一个得分向量
# 整个模型参考 Semantic Compositional Network
class Hierarchical(torch.nn.Module):
    def __init__(self, opts, high2vec=None, word2vec=None):
        super(Hierarchical, self).__init__()
        self._options = opts
        self.high2vec = high2vec
        self.word2vec = word2vec
        opts.high_num, opts.high_emb_size = self.high2vec.shape
        opts.word_num, opts.word_emb_size = self.word2vec.shape
        self.high_bn_l1 = torch.nn.BatchNorm1d(opts.img_size)
        self.high_fc_l1 = torch.nn.Linear(opts.img_size, opts.high_num)
        self.high_score_fc_l1 = torch.nn.Linear(opts.high_num, opts.hidden_size)
        self.img_high_fc_l1 = torch.nn.Linear(opts.img_size, opts.hidden_size)
        self.word_fc_l1 = torch.nn.Linear(opts.hidden_size, opts.hidden_size)
        self.word_bn_l2 = torch.nn.BatchNorm1d(opts.hidden_size+opts.img_size)
        self.word_fc_l2 = torch.nn.Linear(opts.hidden_size+opts.img_size, opts.word_num)
        self.dropout = torch.nn.Dropout(opts.dropout_rate)
        self.var_init()
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=opts.learning_rate)

    def var_init(self):
        opts = self._options
        torch.nn.init.xavier_normal_(self.high_fc_l1.weight)
        torch.nn.init.xavier_normal_(self.high_score_fc_l1.weight)
        torch.nn.init.xavier_normal_(self.img_high_fc_l1.weight)
        torch.nn.init.xavier_normal_(self.word_fc_l1.weight)
        torch.nn.init.xavier_normal_(self.word_fc_l2.weight)
        torch.nn.init.normal_(self.word_bn_l2.weight)
        torch.nn.init.constant_(self.word_bn_l2.bias, 0.0)

    def forward(self, img, high_words, words):
        opts = self._options
        self.train()
        batch_size = img.size(0)
        high_prob = self.high_fc_l1(self.high_bn_l1(img)).sigmoid()
        joint_vec = self.img_high_fc_l1(self.dropout(img)).mul(self.high_score_fc_l1(high_prob))
        word_prob = self.word_fc_l2(self.dropout(
            self.word_bn_l2(torch.cat([func.relu(self.word_fc_l1(self.dropout(joint_vec))), img], -1))
        )).sigmoid()
        high_word_labels = torch.zeros(batch_size, opts.high_num).scatter(1, high_words, 1).long()
        high_word_loss = self.solve_high_loss(high_prob, high_word_labels)
        word_labels = torch.zeros(batch_size, opts.word_num).scatter(1, words, 1).long()
        word_loss = self.solve_word_loss(word_prob, word_labels)
        reg_loss = self.solve_reg_loss()
        loss = opts.high_word_factor * high_word_loss + \
            opts.word_factor * word_loss + reg_loss
        self.backward(loss)
        if opts.display:
            print('[Loss:{:.4f}][HighWordLoss:{:.4f}][WordLoss:{:.4f}][RegLoss:{:.4f}][SelHighSize:{}][SelSize:{}]'.
                  format(loss, high_word_loss, word_loss, reg_loss, opts.select_high_size, opts.select_word_size))
            sel_high_words = high_prob.topk(opts.select_high_size)[1]
            sel_words = word_prob.topk(opts.select_word_size)[1]
            high_recall, high_precision = \
                self.check_cover(high_words.data.cpu().numpy(), sel_high_words.data.cpu().numpy(), mode='high')
            recall, precision = \
                self.check_cover(words.data.cpu().numpy(), sel_words.data.cpu().numpy(), mode='all')
            print('[HighStat Recall:{}]'.format(high_recall))
            print('[HighStat Precision:{}]'.format(high_precision))
            print('[Stat Recall:{}]'.format(recall))
            print('[Stat Precision:{}]'.format(precision))
            print('-'*100)

    def full_forward(self, img):
        high_prob = self.high_fc_l1(self.high_bn_l1(img)).sigmoid()
        joint_vec = self.img_high_fc_l1(self.dropout(img)).mul(self.high_score_fc_l1(high_prob))
        word_prob = self.word_fc_l2(self.dropout(
            self.word_bn_l2(torch.cat([func.relu(self.word_fc_l1(self.dropout(joint_vec))), img], -1))
        )).sigmoid()
        return word_prob

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

    def predict(self, img, high_words=None, words=None, all_ret=False):
        opts = self._options
        self.eval()
        high_prob = self.high_fc_l1(self.high_bn_l1(img)).sigmoid()
        joint_vec = self.img_high_fc_l1(img).mul(self.high_score_fc_l1(high_prob))
        word_prob = self.word_fc_l2(
            self.word_bn_l2(torch.cat([func.relu(self.word_fc_l1(joint_vec)), img], -1))
        ).sigmoid()
        sel_high_words = high_prob.topk(opts.select_high_size)[1]
        sel_words = word_prob.topk(opts.select_word_size)[1]
        if high_words is not None and words is not None:
            high_recall, high_precision = self.check_cover(high_words, sel_high_words.data.cpu().numpy(), mode='high')
            recall, precision = self.check_cover(words, sel_words.data.cpu().numpy(), mode='all')
            if all_ret:
                return sel_high_words, sel_words, high_recall, high_precision, recall, precision
            else:
                return high_recall, high_precision, recall, precision

        else:
            return sel_high_words, sel_words

    def check_cover(self, words, selects, mode='high'):
        opts = self._options
        if mode == 'high':
            indices = [2, 3, 4, 6, 8, 2*opts.select_high_size, opts.select_high_size]
        else:
            indices = [16, 32, 64, 128, 256, opts.select_word_size]
        recall = [[] for _ in range(len(indices))]
        precision = [[] for _ in range(len(indices))]
        for word, select in zip(words, selects):
            word = set(word)
            for i, idx in enumerate(indices):
                idx_select = set(select[:idx])
                cover = word.intersection(idx_select)
                recall[i].append(len(cover)/float(len(word)))
                precision[i].append(len(cover)/float(len(idx_select)))
        recall = np.round(np.array(recall).mean(-1), 4)
        precision = np.round(np.array(precision).mean(-1), 4)
        return recall, precision

    def solve_high_loss(self, word_prob, label):
        opts = self._options
        pos_mask = label.float().ge(0.5).float()
        neg_mask = torch.ones_like(pos_mask).sub(pos_mask)
        every_loss = func.binary_cross_entropy(word_prob, label.float(), reduction='none')
        loss = every_loss.sum(-1).mean()
        if opts.display:
            print('[HighWord][PosSampleNum:{}][NegSampleNum:{}]'.
                  format(pos_mask.sum().data.cpu().numpy(), neg_mask.sum().data.cpu().numpy()))
            mean_pos_prob = pos_mask.mul(word_prob).sum().div(pos_mask.sum())
            mean_neg_prob = neg_mask.mul(word_prob).sum().div(neg_mask.sum())
            print('[HighWord][MeanPosProb:{:.4f}][MeanNegProb:{:.4f}]'.format(mean_pos_prob, mean_neg_prob))
        return loss

    def solve_word_loss(self, word_prob, label):
        opts = self._options
        pos_mask = label.float().ge(0.5).float()
        neg_mask = torch.ones_like(pos_mask).sub(pos_mask)
        pos_prob = word_prob.mul(pos_mask)
        neg_prob = word_prob.mul(neg_mask)
        neg_mask = neg_prob.ge(pos_prob.add(neg_mask).min(-1, keepdim=True)[0].sub(0.0)).float().mul(neg_mask)
        every_loss = func.binary_cross_entropy(word_prob, label.float(), reduction='none')
        loss = every_loss.sum(-1).mean()
        if opts.display:
            print('[Word][PosSampleNum:{}][NegSampleNum:{}]'.
                  format(pos_mask.sum().data.cpu().numpy(), neg_mask.sum().data.cpu().numpy()))
            mean_pos_prob = pos_mask.mul(word_prob).sum().div(pos_mask.sum())
            mean_neg_prob = neg_mask.mul(word_prob).sum().div(neg_mask.sum())
            print('[Word][MeanPosProb:{:.4f}][MeanNegProb:{:.4f}]'.format(mean_pos_prob, mean_neg_prob))
        return loss

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

