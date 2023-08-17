Upload file full\_train.csv lên Google drive cá nhân

Upload file test\_shuffle.csv lên Google drive cá nhân

Chạy foody\_sentiment.ipynb trong GoogleColab

Để phân tích 1 bình luận là tích cực hay tiêu cực, chúng ta làm theo như trong ví dụ

Giả sử ta có :

test\_data1 = ["Anh ấy quá tuyệt vời", "Tởm quá", "Changf trai 97 sinh ra ở Nghệ An"]

\# Lần lượt chạy các lệnh sau (với điều kiện là chạy hết các lệnh trước đó trong file.ipynb):

test\_corpus = []

for sent in  test\_data1:

if(type(sent) != str):

sent = "bình luận không xác định"

test\_corpus.append((id, sent))

sents = [sent for \_,sent in test\_corpus]

purified\_sents = [purify(str(sent)) for sent in sents]

preprocessed\_sents = preprocess(purified\_sents)

padded\_sents, attention\_masks = encode\_plus(preprocessed\_sents, max\_len=256)

batch\_size = 32

inputs = torch.tensor(padded\_sents)

masks = torch.tensor(attention\_masks)

test\_data = TensorDataset(masks, inputs)

test\_sampler = SequentialSampler(test\_data)

test\_dataloader = DataLoader(test\_data, sampler=test\_sampler, batch\_size=batch\_size)

sub = []

test\_model.eval()

total\_loss = total = 0

for batch in tqdm(test\_dataloader):

\# batch = torch.toTensor(t.to(device) for t in batch)

input\_mask, input\_ids = batch

input\_ids = input\_ids.to(device)

input\_mask = input\_mask.to(device)

\# Forwards pass

with torch.no\_grad():

outputs = test\_model(input\_ids, attention\_mask=input\_mask)

\# loss = outputs[0]

\#     outputs = outputs.logits

loss = outputs.detach().cpu().numpy()


preds\_flat = np.argmax(loss, axis=1).flatten()

for tup in preds\_flat:

sub.append(tup)

Kết quả in ra là 1 mảng nhị phân sub tương ứng với từng bình luận, với 0 là bình luận tiêu cực và 1 là bình luận tích cực.
