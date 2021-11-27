import os
import collections
from tqdm import tqdm
from pprint import pprint

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader

from src.param import args
import src.config as config

from src.tasks.vqa_model import VQAModel
from src.lxrt.optimization import BertAdam
from src.tasks.losses import FocalLoss, Plain
from src.pretrain.qa_answer_table import load_lxmert_qa
from src.tasks.vqa_data import VQADataset, VQATorchDataset, VQAEvaluator

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


def get_data_tuple(splits: str, bs: int, shuffle=False, drop_last=False) -> DataTuple:
    dset = VQADataset(splits)
    tset = VQATorchDataset(dset)
    evaluator = VQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class VQA:
    def __init__(self):
        # Datasets
        self.train_tuple = get_data_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=False)
        if args.valid != "":
            self.valid_tuple = get_data_tuple(
                args.valid, bs=1024, shuffle=False, drop_last=False)
        else:
            self.valid_tuple = None
        
        # Model
        self.model = VQAModel(self.train_tuple.dataset.num_answers)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)
        
        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Loss and Optimizer
        if args.loss_fn == 'Plain':
            self.loss_fn = Plain()
        elif args.loss_fn == 'Focal':
            self.loss_fn = FocalLoss()
        else:
            raise RuntimeError('not implement for {}'.format(args.loss_fn))
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("BertAdam Total Iters: %d" % t_total)
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(self.model.parameters(), args.lr)
        
        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        for epoch in range(args.epochs):
            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, target, mask, score) in iter_wrapper(enumerate(loader)):
                self.model.train()
                self.optim.zero_grad()

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                logit = self.model(feats, boxes, sent)
                assert logit.dim() == target.dim() == 2

                dict_args = {}
                if config.use_miu:
                    dict_args['miu'] = score.cuda()
                    dict_args['mask'] = mask.cuda()
                loss = self.loss_fn(logit, target, **dict_args)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score, label = logit.max(1)
                for qid, ans in zip(ques_id, label.cpu().numpy()):
                    quesid2ans[qid.item()] = ans

            print("\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans)[0] * 100.))

            if self.valid_tuple is not None:  # Do Validation
                # valid_score = self.evaluate(eval_tuple)
                valid_score, save_results = self.evaluate(eval_tuple)
                if valid_score > best_valid:

                    best_results = save_results

                    best_valid = valid_score
                    self.save(args.name, epoch, best_valid)

                print("Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) +
                      "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.))

        import json
        cp = 'CP' if config.cp_data else 'NC'
        miu = 'miu' if config.use_miu else 'base'
        with open('./result_{}_{}_{}.json'.format(
                cp, config.version, miu), 'w') as fd:
            json.dump(best_results, fd)

    def predict(self, eval_tuple: DataTuple, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}

        for i, datum_tuple in tqdm(enumerate(loader), total=len(loader)):
            ques_id, feats, boxes, sent, target = datum_tuple[:5]
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        quesid2ans = self.predict(eval_tuple, dump)
        return eval_tuple.evaluator.evaluate(quesid2ans)

    def save(self, name, epoch, best_val_score):
        results = {
            'epoch': epoch + 1,
            'best_val_score': best_val_score,
            'model_state': self.model.state_dict(),
            'optim_state': self.optim.state_dict(),
            'loss_state': self.loss_fn.state_dict(),
        }
        torch.save(results, os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s" % path)
        self.model.load_state_dict(state_dict['model_state'])
        self.optim.load_state_dict(state_dict['optim_state'])
        self.loss_fn.load_state_dict(state_dict['loss_state'])


if __name__ == "__main__":
    print(args)
    print_keys = ['cp_data', 'version', 'use_miu']
    print_dict = {key: getattr(config, key) for key in print_keys}
    pprint(print_dict, width=150)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.benchmark = True

    # Build Class
    vqa = VQA()

    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None:
        vqa.load(args.load)

    # Test or Train
    if args.test is not None:
        if 'test' in args.test:
            vqa.predict(
                get_data_tuple(args.test, bs=950,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'test_predict.json')
            )
        elif 'val' in args.test:    
            # Since part of valididation data are used in pre-training/fine-tuning,
            # only validate on the minival set.
            result = vqa.evaluate(
                get_data_tuple('val', bs=950,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'val_predict.json')
            )
        else:
            assert False, "No such test option for %s" % args.test
    else:
        vqa.train(vqa.train_tuple, vqa.valid_tuple)
