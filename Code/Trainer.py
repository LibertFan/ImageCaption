import os
import sys
import shutil
import time
import argparse
import torch
import numpy as np
from collections import Counter
import nltk
import pickle
sys.path.append('../..')
from OrderClusterStream import DataStream
from Pipeline import Pipeline as Model


def main():
    opts = read_commands()
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.device
    trainer = Trainer(opts)
    if opts.is_training:
        trainer.process()
    else:
        trainer.test(mode='Retrieve')
        trainer.test(mode='Generate_MLE')
        trainer.test(mode='Generate_Reinforce')


def read_commands():
    data_root = os.path.abspath('../Data')
    train_root = os.path.join(data_root, 'train')
    model_root = os.path.join(data_root, 'model')
    parser = argparse.ArgumentParser(usage='MS COCO Data Train Parameters')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_name', type=str, default='RLSCNV3')
    parser.add_argument('--device', type=str, default='2')
    parser.add_argument('--data_name', type=str, default='coco')
    parser.add_argument('--data_id', type=int, default=100)
    parser.add_argument('--tag_id', type=int, default=100)
    parser.add_argument('--data_plain', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default=os.path.join(data_root, 'log'))
    parser.add_argument('--save_dir', type=str, default=os.path.join(data_root, 'model'))
    parser.add_argument('--save_folder', type=str, default=None)
    parser.add_argument('--util_dir', type=str, default=os.path.join(data_root, 'util'))
    parser.add_argument('--util_folder', type=str, default=None)
    parser.add_argument('--pre_reinforce_path', type=str, default=None)
    parser.add_argument('--pre_reinforce_constraint_path', type=str, default=None)
    parser.add_argument('--pre_mle_path', type=str, default=None)
    parser.add_argument('--pre_ret_path', type=str, default=None)
    parser.add_argument('--word_num', type=int, default=None)
    parser.add_argument('--min_freq', type=int, default=5)
    parser.add_argument('--beam_size', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--grad_value_clip', type=int, default=1)
    parser.add_argument('--grad_clip_value', type=float, default=0.2)
    parser.add_argument('--grad_norm_clip', type=int,  default=0)
    parser.add_argument('--grad_norm_clip_value', type=float, default=2.0)
    parser.add_argument('--grad_global_norm_clip', type=int,  default=0)
    parser.add_argument('--grad_global_norm_clip_value', type=float, default=5.0)
    parser.add_argument('--l1_reg', type=float, default=1e-7)
    parser.add_argument('--l2_reg', type=float, default=1e-7)
    parser.add_argument('--epochs', type=int, default=500000)
    parser.add_argument('--word_emb_size', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--caption_size', type=int, default=17)
    parser.add_argument('--group_size', type=int, default=5)
    parser.add_argument('--img_size', type=int, default=2048)
    parser.add_argument('--rnn_size', type=int, default=512)
    parser.add_argument('--transpose_size', type=int, default=1024)
    parser.add_argument('--hidden_size', type=int, default=2048)
    parser.add_argument('--reinforce_size', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    parser.add_argument('--concept_dropout_rate', type=float, default=0.1)
    parser.add_argument('--word_dropout_keep', type=float, default=1.0)
    parser.add_argument('--length_penalty_factor', type=float, default=0.6)
    parser.add_argument('--display_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=50)
    args = parser.parse_args()
    args.tag = 'THV{}_V{}'.format(args.data_id, args.tag_id)
    args.is_training = True if args.is_training == 1 else False
    args.data_plain = True if args.data_plain == 1 else False
    args.train_data_path = os.path.join(train_root, '{}_v{}/{}_train_v{}.pkl'.format(
        args.data_name, args.data_id, args.data_name, args.data_id))
    args.val_data_path = os.path.join(train_root, '{}_v{}/{}_val_v{}.pkl'.format(
        args.data_name, args.data_id, args.data_name, args.data_id))
    args.test_data_path = os.path.join(train_root, '{}_v{}/{}_test_v{}.pkl'.format(
        args.data_name, args.data_id, args.data_name, args.data_id))
    args.vocab_info_path = os.path.join(train_root, '{}_v{}/{}_info_v{}.pkl'.format(
        args.data_name, args.data_id, args.data_name, args.data_id))
    args.grad_value_clip = True if args.grad_value_clip == 1 else False
    args.grad_norm_clip = True if args.grad_norm_clip == 1 else False
    args.grad_global_norm_clip = True if args.grad_global_norm_clip == 1 else False
    args.data_plain = True if args.data_plain == 1 else False
    return args


class Trainer(object):
    def __init__(self, opts):
        self._options = opts
        self.model_name = opts.model_name + '_' + opts.tag
        self.log_file = os.path.join(opts.log_dir, self.model_name+'_{}.txt'.format(
            time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))))
        self.save_folder = os.path.join(opts.save_dir, self.model_name) \
            if opts.save_folder is None else opts.save_folder
        self.util_folder = os.path.join(opts.util_dir, self.model_name) \
            if opts.util_folder is None else opts.util_folder
        if opts.is_training:
            if os.path.exists(self.log_file):
                del_cmd = input('[Warning][LogFile {} exists][Delete it?]'.format(self.log_file))
                if del_cmd:
                    os.remove(self.log_file)
            if os.path.exists(self.save_folder):
                del_cmd = bool(eval(input('[Warning][SaveFile {} exists][Delete it?]'.format(self.save_folder))))
                if del_cmd:
                    shutil.rmtree(self.save_folder)
            os.mkdir(self.save_folder)
            if os.path.exists(self.util_folder):
                del_cmd = bool(eval(input('[Warning][UtilFile {} exists][Delete it?]'.format(self.util_folder))))
                if del_cmd:
                    shutil.rmtree(self.util_folder)
            os.mkdir(self.util_folder)
        self.streamer = DataStream(opts)
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.model = Model(opts, self.streamer)
        self.epoch = 0
        self.g_best_score = self.r_best_score = 0.0

    def process(self):
        opts = self._options
        self.epoch = 0
        opts.epochs = 100
        self.train(mode='Retrieve')
        self.epoch = 0
        opts.epochs = 100
        self.train(mode='Generate_MLE')
        self.epoch = 0
        opts.epochs = 100
        self.train(mode='Generate_Reinforce')
        self.epoch = 0
        opts.epochs = 100
        self.train(mode='Generate_Reinforce_Constraint')

    def train(self, mode=None):
        if mode not in ['Retrieve', 'Generate_MLE', 'Generate_Reinforce', 'Generate_Reinforce_Constraint']:
            raise Exception('Current train mode is not supported')
        opts, model, streamer = self._options, self.model, self.streamer
        self.adjust(mode=mode)
        for e in range(opts.epochs):
            self.epoch += 1
            self.adjust(mode=mode)
            image_idx, batch_caption, batch_feature, batch_num_caption = streamer.get_next_full_train_batch()
            model(batch_feature, batch_num_caption, batch_caption, mode=mode)
            if opts.save:
                self.val(mode=mode)

    def adjust(self, mode=None):
        if mode not in ['Retrieve', 'Generate_MLE', 'Generate_Reinforce', 'Generate_Reinforce_Constraint']:
            print('[Mode:{}]'.format(mode))
            raise Exception('Current train mode is not supported')
        epoch, opts = self.epoch, self._options
        if mode == 'Retrieve':
            opts.learning_rate = 1e-4
            opts.high_word_factor = 0.1
            opts.word_factor = 1.0
            opts.l1_factor = 1e-5
            opts.l2_factor = 1e-5
            opts.batch_size = 64
            opts.select_high_size = 5
            opts.select_word_size = 64
            param_groups = self.model.retriever.optimizer.param_groups
        else:
            opts.schedule_sampling = False
            if mode == 'Generate_MLE':
                opts.l1_factor = 1e-6
                opts.l2_factor = 1e-6
                opts.learning_rate = 5e-4 * (0.8 ** (epoch // 20000))
                opts.batch_size = 64
                param_groups = self.model.generator.optimizer.param_groups
            elif mode == 'Generate_Reinforce':
                opts.sample_decode_size = 1
                opts.l1_factor = 1e-7
                opts.l2_factor = 1e-7
                opts.temperature = 1.0
                opts.mle_factor = 0.0
                opts.reinforce_factor = 1.0
                opts.learning_rate = 5e-5 * (0.8 ** (epoch // 30000))
                opts.batch_size = 64
                opts.select_high_size = 5
                opts.select_word_size = opts.word_num
                param_groups = self.model.generator.optimizer.param_groups
            else:
                opts.sample_decode_size = 1
                opts.l1_factor = 1e-7
                opts.l2_factor = 1e-7
                opts.temperature = 1.0
                opts.learning_rate = 5e-5 * (0.8 ** (epoch // 30000))
                opts.batch_size = 64
                opts.select_high_size = 5
                opts.select_word_size = 64
                param_groups = self.model.generator.optimizer.param_groups
        opts.display = (epoch % opts.display_every) == 0
        opts.save = (epoch % opts.save_every) == 0
        opts.pe = getattr(opts, 'pe', -1) + 1
        for param_group in param_groups:
            param_group['lr'] = opts.learning_rate
        if opts.display:
            print('[Adjust][{}][Epoch:{}][LearningRate:{:.6f}][L1:{}][L2:{}][Dropout:{}][SelectWordsSize:{}]'.
                  format(mode, epoch, opts.learning_rate, opts.l1_factor, opts.l2_factor, opts.dropout_rate,
                         opts.select_word_size))

    def val(self, mode=None):
        opts, model, streamer, epoch = self._options, self.model, self.streamer, self.epoch
        if mode not in ['Retrieve', 'Generate_MLE', 'Generate_Reinforce', 'Generate_Reinforce_Constraint']:
            raise Exception('Current train mode is not supported')
        elif mode != 'Retrieve':
            sel_hypos, hypos, refs = dict(), dict(), dict()
            val_idx = 0
            opts.batch_size = 32
            while True:
                val_idx += 1
                data = streamer.get_next_val_batch()
                if data is None:
                    break
                image_idx, batch_caption, batch_feature, batch_num_caption = data
                sel_gen_tokens, gen_tokens = model.eval_generator(batch_feature)
                if not isinstance(gen_tokens, np.ndarray):
                    gen_tokens = gen_tokens.data.cpu().numpy()
                if not isinstance(sel_gen_tokens, np.ndarray):
                    sel_gen_tokens = sel_gen_tokens.data.cpu().numpy()
                for i, (idx, sel_gen_token, gen_token, captions) in \
                        enumerate(zip(image_idx, sel_gen_tokens, gen_tokens, list(batch_caption))):
                    index = str(idx)
                    refs[index] = list(captions)
                    gen_caption = streamer.tokens2sentence(gen_token)
                    hypos[index] = [gen_caption]
                    sel_gen_caption = streamer.tokens2sentence(sel_gen_token)
                    sel_hypos[index] = [sel_gen_caption]
            print('[Val][Epoch:{}]'.format(epoch))
            base_scores = streamer.quick_measure_score(hypos, refs)
            print('[Val][Similarity Metrics]')
            for name, score in base_scores.items():
                print('[Base][{}: {}]'.format(name, score))
            sel_scores = streamer.quick_measure_score(sel_hypos, refs)
            for name, score in sel_scores.items():
                print('[Sel][{}: {}]'.format(name, score))
            val_score = sel_scores.get('METEOR') + sel_scores.get('CIDEr')
            print('[ValScore:{}]'.format(val_score))
            if val_score > self.g_best_score and opts.is_training:
                if mode == 'Generate_Reinforce':
                    self.g_best_score = val_score
                    path = os.path.join(self.save_folder, self.model_name + '_BestRLGModel.pkl')
                    print('[Val][Reinforce_Generate][NewBestScore: {}][SavePath: {}]'.format(val_score, path))
                    torch.save(self.model.state_dict(), path)
                    path = os.path.join(self.save_folder, self.model_name + '_Single_BestRLGModel.pkl')
                    torch.save(self.model.generator.state_dict(), path)
                elif mode == 'Generate_Reinforce_Constraint':
                    self.g_best_score = val_score
                    path = os.path.join(self.save_folder, self.model_name + '_BestRLGCModel.pkl')
                    print('[Val][Reinforce_Generate][NewBestScore: {}][SavePath: {}]'.format(val_score, path))
                    torch.save(self.model.state_dict(), path)
                    path = os.path.join(self.save_folder, self.model_name + '_Single_BestRLGCModel.pkl')
                    torch.save(self.model.generator.state_dict(), path)
                else:
                    self.g_best_score = val_score
                    path = os.path.join(self.save_folder, self.model_name + '_BestGModel.pkl')
                    print('[Val][Generate][NewBestScore: {}][SavePath: {}]'.format(val_score, path))
                    torch.save(self.model.state_dict(), path)
                    path = os.path.join(self.save_folder, self.model_name + '_Single_BestGModel.pkl')
                    torch.save(self.model.generator.state_dict(), path)
        else:
            val_idx = 0
            opts.batch_size = 64
            opts.display = False
            num_sum, recall_sum, precision_sum, high_recall_sum, high_precision_sum = 0, 0.0, 0.0, 0.0, 0.0
            while True:
                val_idx += 1
                data = streamer.get_next_val_batch()
                if data is None:
                    break
                image_idx, batch_caption, batch_feature, batch_num_caption = data
                batch_size = len(image_idx)
                high_recall, high_precision, recall, precision = model.eval_retriever(batch_feature, batch_num_caption)
                recall_sum += recall * batch_size
                precision_sum += precision * batch_size
                high_recall_sum += high_recall * batch_size
                high_precision_sum += high_precision * batch_size
                num_sum += batch_size
            recall = np.round(recall_sum / float(num_sum), 4)
            precision = np.round(precision_sum / float(num_sum), 4)
            high_recall = np.round(high_recall_sum / float(num_sum), 4)
            high_precision = np.round(high_precision_sum / float(num_sum), 4)
            print('[Val][Epoch:{}]'.format(epoch))
            print('[Recall]', recall)
            print('[Precision]', precision)
            print('[F1]', (2 * np.array(recall) * np.array(precision)) / (np.array(recall) + np.array(precision)))
            print('[HighRecall]', high_recall)
            print('[HighPrecision]', high_precision)
            print('[HighF1]', (2 * np.array(high_recall) * np.array(high_precision)) /
                  (np.array(high_recall) + np.array(high_precision)))
            val_score = recall[-1]
            print('[ValScore:{}]'.format(val_score))
            if val_score > self.r_best_score and opts.is_training:
                self.r_best_score = val_score
                path = os.path.join(self.save_folder, self.model_name+'_BestRModel.pkl')
                print('[Val][Retrieve][NewBestScore: {}][SavePath: {}]'.format(val_score, path))
                torch.save(self.model.state_dict(), path)
                path = os.path.join(self.save_folder, self.model_name + '_Single_BestRModel.pkl')
                torch.save(self.model.retriever.state_dict(), path)

    def test(self, mode=None, from_file=True):
        opts, model, streamer = self._options, self.model, self.streamer
        if mode not in ['Retrieve', 'Generate_MLE', 'Generate_Reinforce', 'Generate_Reinforce_Constraint']:
            raise Exception('Current train mode is not supported')
        elif mode != 'Retrieve':
            self.adjust(mode=mode)
            if from_file:
                if mode == 'Generate_MLE':
                    if opts.pre_mle_path is None:
                        path = os.path.join(self.save_folder, self.model_name + '_BestGModel.pkl')
                    else:
                        path = opts.pre_mle_path
                    print('[Test][MLE][LoadPath: {}]'.format(path))
                    try:
                        model.generator.load_state_dict(torch.load(path))
                    except RuntimeError:
                        model.load_state_dict(torch.load(path))
                elif mode == 'Generate_Reinforce':
                    if opts.pre_reinforce_path is None:
                        path = os.path.join(self.save_folder, self.model_name + '_BestRLGModel.pkl')
                    else:
                        path = opts.pre_reinforce_path
                    print('[Test][Reinforce][LoadPath: {}]'.format(path))
                    try:
                        model.generator.load_state_dict(torch.load(path))
                    except RuntimeError:
                        model.load_state_dict(torch.load(path))
                else:
                    if opts.pre_reinforce_constraint_path is None:
                        path = os.path.join(self.save_folder, self.model_name + '_BestRLGCModel.pkl')
                    else:
                        path = opts.pre_reinforce_constraint_path
                    print('[Test][Reinforce][LoadPath: {}]'.format(path))
                    try:
                        model.generator.load_state_dict(torch.load(path))
                    except RuntimeError:
                        model.load_state_dict(torch.load(path))

            sel_hypos, hypos, refs = dict(), dict(), dict()
            val_idx = 0
            opts.batch_size = 32
            while True:
                val_idx += 1
                if val_idx > 1000:
                    break
                data = streamer.get_next_test_batch()
                if data is None:
                    break
                image_idx, batch_caption, batch_feature, batch_num_caption = data
                sel_gen_tokens, gen_tokens = model.eval_generator(batch_feature)
                if not isinstance(gen_tokens, np.ndarray):
                    gen_tokens = gen_tokens.data.cpu().numpy()
                if not isinstance(sel_gen_tokens, np.ndarray):
                    sel_gen_tokens = sel_gen_tokens.data.cpu().numpy()
                for i, (idx, sel_gen_token, gen_token, captions) in \
                        enumerate(zip(image_idx, sel_gen_tokens, gen_tokens, list(batch_caption))):
                    index = str(idx)
                    refs[index] = list(captions)
                    gen_caption = streamer.tokens2sentence(gen_token)
                    hypos[index] = [gen_caption]
                    sel_gen_caption = streamer.tokens2sentence(sel_gen_token)
                    sel_hypos[index] = [sel_gen_caption]
            print('[Test]')
            sel_scores = streamer.measure_score(sel_hypos, refs)
            for name, score in sel_scores.items():
                print('[Sel][{}: {}]'.format(name, score))
            base_scores = streamer.measure_score(hypos, refs)
            for name, score in base_scores.items():
                print('[Base][{}: {}]'.format(name, score))
        else:
            self.adjust(mode=mode)
            if from_file:
                if opts.pre_ret_path is None:
                    path = os.path.join(self.save_folder, self.model_name + '_BestRModel.pkl')
                else:
                    path = opts.pre_ret_path
                print('[Test][LoadPath: {}]'.format(path))
                try:
                    model.retriever.load_state_dict(torch.load(path))
                except RuntimeError:
                    model.load_state_dict(torch.load(path))
            val_idx = 0
            opts.batch_size = 64
            opts.display = False
            num_sum, recall_sum, precision_sum, high_recall_sum, high_precision_sum = 0, 0.0, 0.0, 0.0, 0.0
            while True:
                val_idx += 1
                data = streamer.get_next_test_batch()
                if data is None:
                    break
                image_idx, batch_caption, batch_feature, batch_num_caption = data
                batch_size = len(image_idx)
                high_recall, high_precision, recall, precision = model.eval_retriever(batch_feature, batch_num_caption)
                recall_sum += recall * batch_size
                precision_sum += precision * batch_size
                high_recall_sum += high_recall * batch_size
                high_precision_sum += high_precision * batch_size
                num_sum += batch_size
            recall = np.round(recall_sum / float(num_sum), 4)
            precision = np.round(precision_sum / float(num_sum), 4)
            high_recall = np.round(high_recall_sum / float(num_sum), 4)
            high_precision = np.round(high_precision_sum / float(num_sum), 4)
            print('[Test]')
            print('[Recall]', recall)
            print('[Precision]', precision)
            print('[F1]', (2 * np.array(recall) * np.array(precision)) / (np.array(recall) + np.array(precision)))
            print('[HighRecall]', high_recall)
            print('[HighPrecision]', high_precision)
            print('[HighF1]', (2 * np.array(high_recall) * np.array(high_precision)) /
                  (np.array(high_recall) + np.array(high_precision)))


if __name__ == '__main__':
    main()
