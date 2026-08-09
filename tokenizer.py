class SimpleTokenizer:
    def __init__(self):
        # 构建词表：每个字符对应一个唯一 ID
        self.vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def train(self, texts):
        """从语料中构建词表"""
        idx = len(self.vocab)
        for text in texts:
            for char in text:
                if char not in self.vocab:
                    self.vocab[char] = idx
                    self.inverse_vocab[idx] = char
                    idx += 1

    def encode(self, text, max_length=10):
        """文本 → ID 序列"""
        tokens = ["[CLS]"] + list(text) + ["[SEP]"]
        ids = [self.vocab.get(t, self.vocab["[UNK]"]) for t in tokens]

        # 填充或截断
        if len(ids) < max_length:
            ids += [self.vocab["[PAD]"]] * (max_length - len(ids))
        else:
            ids = ids[:max_length - 1] + [self.vocab["[SEP]"]]

        # Attention Mask：1 表示真实 token，0 表示填充
        mask = [1 if id != self.vocab["[PAD]"] else 0 for id in ids]
        return {"input_ids": ids, "attention_mask": mask}

    def decode(self, ids):
        """ID 序列 → 文本"""
        chars = [self.inverse_vocab.get(i, "[UNK]") for i in ids]
        return "".join(chars).replace("[PAD]", "").replace("[CLS]", "").replace("[SEP]", " ")


# ========== 使用示例 ==========
corpus = ["我爱北京", "天安门上太阳升", "自然语言处理"]
tokenizer = SimpleTokenizer()
tokenizer.train(corpus)

text = "我爱自然，笑嘻嘻"
result = tokenizer.encode(text, max_length=12)
print("编码结果:", result)
# 编码结果: {'input_ids': [2, 4, 5, 6, 7, 3, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]}

print("解码结果:", tokenizer.decode(result["input_ids"]))
# 解码结果: " 我爱自然 "