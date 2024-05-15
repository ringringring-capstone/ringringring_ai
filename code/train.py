import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader

import csv
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 데이터 전처리
class DeliveryDataset(Dataset):
    def __init__(self, sen_data, tokenizer):
        self.sen_data = sen_data
        self.tokenizer = tokenizer
        self.conver_pair = []

        # data 불러오기
        with open(self.sen_data, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.conver_pair.append({"customer": row["Q"], "staff": row["A"]})

        # Determine maximum sequence length dynamically
        self.max_length = 45#max(len(tokenizer.encode(input_text)) for input_text in self.inputs)

    def __len__(self):
        return len(self.conver_pair)

    def __getitem__(self, idx):
        pair = self.conver_pair[idx]
        customer_text = pair["customer"]
        staff_text = pair["staff"]

        # Tokenize and encode input and label texts
        input_encoding = self.tokenizer(
            customer_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        label_encoding = self.tokenizer(
            staff_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Return input and label tensors
        return {
            'input_ids': input_encoding['input_ids'].squeeze(0),
            'attention_mask': input_encoding['attention_mask'].squeeze(0),
            'labels': label_encoding['input_ids'].squeeze(0)  # Assuming same max_length for input and label
        }


def main():
    # 모델 및 토크나이저 로드
    BOS = '</s>'
    EOS = '</s>'
    MASK = '<unused0>'
    SENT = '<unused1>'
    PAD = '<pad>'

    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                bos_token=BOS, eos_token=EOS, unk_token='<unk>',
                pad_token=PAD, mask_token=MASK)
    
    sen_data = "data/final_first.csv"
    # Create dataset
    delivery_dataset = DeliveryDataset(sen_data, tokenizer)

    # Create DataLoader
    batch_size = 128  # Batch size
    delivery_dataloader = DataLoader(delivery_dataset, batch_size=batch_size, shuffle=True)
    #train, test = train_test_split(test_size=0.4, shuffle=True)

    # 모델 불러오기.
    model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")

    # Define training parameters
    epochs = 10
    learning_rate = 5e-5
    warmup_steps = 100
    total_steps = len(delivery_dataloader) * epochs

    # Prepare optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # 시작
    # Training loop
    for epoch in range(epochs):
        #print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        total_loss = 0

        for batch in tqdm(delivery_dataloader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()

            # 데이터 불러오기
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            # model
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs.loss

            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        # 모델 저장

        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, 'model/KoGPT2_checkpoint_(05.14).tar')

        model.save_pretrained("model/kogpt2_fine_model(05.14)")

        avg_loss = total_loss / len(delivery_dataloader)
        print(f"Average Loss: {avg_loss:.4f}")

    print("Training finished!")


if __name__ == "__main__":
    main()