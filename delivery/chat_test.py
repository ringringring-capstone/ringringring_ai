import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

# chat
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

token_path = "model/KoGPT2_checkpoint_(05.14).tar"
tokenizer = PreTrainedTokenizerFast.from_pretrained(token_path,
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK)

# 모델 불러오기
# 저장한 모델과 토크나이저를 불러오기 (로컬 경로로 지정)
model_path = "model/KoGPT2_checkpoint_(05.14)"
model = GPT2LMHeadModel.from_pretrained(model_path)

# 상황과 시작 문장
situation = "짜장면 주문 연습 시작"
start_sentence = "안녕하세요. 짜장면집 입니다. 주문 하시겠어요?"

# 상황과 시작 문장 출력
print("상황:", situation)
print("시작 문장:", start_sentence)

# 사용자의 입력 받기
user_input = input("사용자: ")

# 모델을 사용하여 응답 생성
input_text = start_sentence + " " + user_input
input_ids = tokenizer.encode(input_text, return_tensors='pt')
output = model.generate(input_ids, max_length=50, num_return_sequences=1)

# 생성된 응답 출력
response = tokenizer.decode(output[0], skip_special_tokens=True)
response = response.replace(user_input, "").strip()
print("챗봇:", response)