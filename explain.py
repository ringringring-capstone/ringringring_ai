# -*- coding: utf-8 -*-
import argparse
import logging

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

# 이제까지 동일한 에러 문제 -> optimizer에서 adamw가 작동하지 않아 생긴 문제. adam으로 바꾸니 정상작동
# epoch 한번에 124개씩 훈련 시 한시간 10분정도 소요.


"""
 파싱(parsing)
 - 주어진 문자열 해석하고 그 안에 포함된 정보 추출하는 과정.
 - argparse는 사용자가 입력한 옵션과 값들을 프로그래매에서 사용할 수 있는 형태로 변환.
   -> 옵션 : --chat, --sentiment, --model_params, --trian
"""
parser = argparse.ArgumentParser(description='Simsimi based on KoGPT-2')

# 주어진 사용자 입력에 대한 응답생성
parser.add_argument('--chat',
                    action='store_true',
                    default=False,
                    help='response generation on given user input')

# sentiment : 감정
parser.add_argument('--sentiment',
                    type=str,
                    default='0',
                    help='sentiment for system. 0 is neutral, 1 is negative, 2 is positive.')

# 채팅 시작을 위한 모델 바이너리
parser.add_argument('--model_params',
                    type=str,
                    default='model_chp/model_-last.ckpt',
                    help='model binary for starting chat')

parser.add_argument('--train',
                    action='store_true',
                    default=False,
                    help='for training')

logger = logging.getLogger()
logger.setLevel(logging.INFO)


"""
 특정 토큰들에 대한 상수 정의.
"""
U_TKN = '<usr>'  # 사용자 입력 나타내는 토큰
S_TKN = '<sys>'  # 시스템 출력 나타내는 토큰
BOS = '</s>'     # 문장의 시작을 나타내는 토큰(Beginning Of Sentence)
EOS = '</s>'     # 문장의 끝을 나타내는 토큰(End Of Sentence)
MASK = '<unused0>'  # 마스킹을 위한 토큰. 모델 학습 중 일부 토큰을 가리고 making해 모델이 예측하도록 함.
SENT = '<unused1>'  # 문장의 시작과 끝을 나타내기 위해 사용되는 토큰.
PAD = '<pad>'       # padding을 나타내는 토큰. Sequence(=문장)길이 맞추기 위해 사용될 수 있음.

# Hugging Face의 Transformers 라이브러리 사용해 사전 훈련된 kogpt2 토크나이저를 로드하는 작업 수행.
# 이 코드의 목적은 텍스트 토큰화하고 모델에 입력으로 제공하기 위한 토크나이저 설정.
# 미리 훈련된 kogpt2의 토크나이저 가져옴.
TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK) 


class CharDataset(Dataset):
    def __init__(self, chats, max_len=32):
        self._data = chats
        self.first = True
        self.q_token = U_TKN
        self.a_token = S_TKN
        self.sent_token = SENT
        self.bos = BOS
        self.eos = EOS
        self.mask = MASK
        self.pad = PAD
        self.max_len = max_len
        self.tokenizer = TOKENIZER 

    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx):
        # 데이터에서 한 행 데이터 가져오기 및 Q,A,label 정보 가져옴.
        turn = self._data.iloc[idx]
        q = turn['Q']
        a = turn['A']
        sentiment = str(turn['label'])

        # q, a 문장 토큰화.
        # q는 질문과 감정 정보를 함께 처리함.
        q_toked = self.tokenizer.tokenize(self.q_token + str(q) + \
                                          self.sent_token + sentiment)   
        q_len = len(q_toked)
        a_toked = self.tokenizer.tokenize(self.a_token + str(a) + self.eos)  # eos : 종료토큰
        a_len = len(a_toked)

        # 입력 문장 길이(max_len)보다 길 경우 동작함.
        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len  # 길이 조정을 위해 max_len-q_len 진행.
            if a_len <= 0:
                # 답변 seq 길이가 음수가 되는 경우 예외처리.
                q_toked = q_toked[-(int(self.max_len/2)):]
                q_len = len(q_toked)
                a_len = self.max_len - q_len
                assert a_len > 0
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)
            assert a_len == len(a_toked), f'{a_len} ==? {len(a_toked)}'
        
        # 마스킹을 위한 리스트 생성
        # [mask, mask, ...., mask, ..., <bos>,..A.. <eos>, <pad>....]
        labels = [
            self.mask,
        ] * q_len + a_toked[1:]

        # 데이터 처리 과정을 디버깅하기 위한 로그 기록
        if self.first:
            logging.info("contexts : {}".format(q))
            logging.info("toked ctx: {}".format(q_toked))
            logging.info("response : {}".format(a))
            logging.info("toked response : {}".format(a_toked))
            logging.info('labels {}'.format(labels))
            self.first = False

        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)  # 마스크 생성
        self.max_len
        # label을 토큰 ID로 변환.
        labels_ids = self.tokenizer.convert_tokens_to_ids(labels)
        while len(labels_ids) < self.max_len:
            labels_ids += [self.tokenizer.pad_token_id]
        # 질문과 답변을 합친 후 tokenizer 사용해 토큰 ID로 변환
        token_ids = self.tokenizer.convert_tokens_to_ids(q_toked + a_toked)
        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]

        # 변환된 토큰 ID, mask, 레이블(마스킹 된 seq) 반환.
        return(token_ids, np.array(mask), labels_ids)


# chatbot 모델 학습을 위해 pytorch lightning 모듈 사용
class KoGPT2Chat(LightningModule):
    def __init__(self, hparams, **kwargs):
        super(KoGPT2Chat, self).__init__()
        self.hparams = hparams
        self.neg = -1e18
        self.kogpt2 = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max-len',
                            type=int,
                            default=32,
                            help='max sentence length on input (default: 32)')

        parser.add_argument('--batch-size',
                            type=int,
                            default=96,
                            help='batch size for training (default: 96)')
        parser.add_argument('--lr',
                            type=float,
                            default=5e-5,
                            help='The initial learning rate')
        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')
        return parser
    
    # 순방향 연산
    def forward(self, inputs):
        # (batch, seq_len, hiddens)
        # inputs : 토큰ID, sequence(챗봇을 통해 입력받은 값)
        output = self.kogpt2(inputs, return_dict=True) # 모델에 입력 전달 후 출력 반환 됨.
        return output.logits

    def training_step(self, batch, batch_idx):
        token_ids, mask, label = batch
        out = self(token_ids)
        # mask : 어떤 부분을 입력으로 사용해야하는지 나타내는 이진 벡터.
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, self.neg * torch.ones_like(out))
        # 손실 기록
        loss = self.loss_function(mask_out.transpose(2, 1), label) 
        loss_avg = loss.sum() / mask.sum()
        self.log('train_loss', loss_avg)
        return loss_avg # 손실 평균 반환
    
    # optimizer, scheduler 설정.
    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        # Adam을 사용
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.hparams.lr) #, correct_bias=False
        # warm up lr
        num_train_steps = len(self.train_dataloader()) * self.hparams.max_epochs
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 'name': 'cosine_schedule_with_warmup',
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

    def _collate_fn(self, batch):
        data = [item[0] for item in batch]
        mask = [item[1] for item in batch]
        label = [item[2] for item in batch]
        return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

    def train_dataloader(self):
        data = pd.read_csv('Chatbot_data/c_cut(0-7999).csv')
        self.train_set = CharDataset(data, max_len=self.hparams.max_len)
        train_dataloader = DataLoader(
            self.train_set, batch_size=self.hparams.batch_size, num_workers=2,
            shuffle=True, collate_fn=self._collate_fn)
        return train_dataloader
    
    # 챗봇 대화 기능 구현 부분
    def chat(self, sent='0'):
        tok = TOKENIZER
        sent_tokens = tok.tokenize(sent)
        with torch.no_grad():
            while 1:
                q = input('user > ').strip()
                # 입력으로 '감사합니다' 들어올때까지 반복.
                # 1. 일단 사용자가 마지막 말로 감사합니다 쓴 경우 감사합니다 보내기.
                if '감사합니다' in q:
                    print("네 감사합니다.") # 감사합니다란 문장 보내고 끝내기.
                    break
                a = ''
                while 1:
                    input_ids = torch.LongTensor(tok.encode(U_TKN + q + SENT + sent + S_TKN + a)).unsqueeze(dim=0)
                    pred = self(input_ids)
                    gen = tok.convert_ids_to_tokens(
                        torch.argmax(
                            pred,
                            dim=-1).squeeze().numpy().tolist())[-1]
                    if gen == EOS:
                        break
                    a += gen.replace('▁', ' ')
                print("Simsimi > {}".format(a.strip()))


parser = KoGPT2Chat.add_model_specific_args(parser)
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()
logging.info(args)

if __name__ == "__main__":
    if args.train:
        checkpoint_callback = ModelCheckpoint(
            dirpath='model_chp',
            filename='{cafe}-{epoch:02d}-{train_loss:.2f}',
            verbose=True,
            save_last=True,
            monitor='train_loss',
            mode='min',
            prefix='model_'
        )
        # python train_torch.py --train --gpus 1 --max_epochs 3
        print("모델 불러오기");
        model = KoGPT2Chat(args)
        model.train()
        trainer = Trainer.from_argparse_args( args,
            checkpoint_callback=checkpoint_callback, gradient_clip_val=1.0)
        trainer.fit(model)

        logging.info('best model path {}'.format(checkpoint_callback.best_model_path))
        # save_pretrained는 kogpt2chat에 없는 attribute라는 에러 발생해서 아래로 변경
        torch.save(model.state_dict(), "Chatbot_data")
        print("모델 훈련 끝")
    if args.chat:
        model = KoGPT2Chat.load_from_checkpoint(args.model_params)
        model.chat()
