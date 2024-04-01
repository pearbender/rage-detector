import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font='Ms Gothic')

tokenizer_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

emotion_names = ['Joy', 'Sadness', 'Anticipation',
                 'Surprise', 'Anger', 'Fear', 'Disgust', 'Trust']
emotion_names_jp = ['喜び', '悲しみ', '期待', '驚き', '怒り', '恐れ', '嫌悪', '信頼']  # 日本語版
num_labels = len(emotion_names)

checkpoint = 'test_trainer/checkpoint-2000'
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=num_labels)

# https://www.delftstack.com/ja/howto/numpy/numpy-softmax/


def np_softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


def analyze_emotion(text, show_fig=False, ret_prob=False):
    model.eval()

    tokens = tokenizer(text, truncation=True, return_tensors="pt")
    tokens.to(model.device)
    preds = model(**tokens)
    prob = np_softmax(preds.logits.cpu().detach().numpy()[0])
    out_dict = {n: p for n, p in zip(emotion_names_jp, prob)}

    if show_fig:
        plt.figure(figsize=(8, 3))
        df = pd.DataFrame(out_dict.items(), columns=['name', 'prob'])
        sns.barplot(x='name', y='prob', data=df, palette='Set1')
        plt.title('入力文 : ' + text, fontsize=15)
        plt.show()

    if ret_prob:
        return out_dict


analyze_emotion('今日から長期休暇だぁーーー！！！', show_fig=True)
