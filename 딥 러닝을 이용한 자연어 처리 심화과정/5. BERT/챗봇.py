#!pip install transformers
#!pip install sentence_transformers

import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
import urllib.request
from sentence_transformers import SentenceTransformer

#학습데이터
urllib.request.urlretrieve("https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv", filename="ChatBotData.csv")
train_data = pd.read_csv('ChatBotData.csv')
train_data.head()

#모델 로드
model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

#데이터에서 모든 질문열. 즉, train_data['Q']에 대해서 문장 임베딩 값을 구한 후 embedding이라는 새로운 열에 저장합니다.

train_data['embedding'] = train_data.apply(lambda row: model.encode(row.Q), axis = 1)

#코사인 유사도
def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

#답변 함수
def return_answer(question) :
    embedding = model.encode(question)
    train_data['score'] = train_data.apply(lambda x: cos_sim(x['embedding'], embedding), axis=1)
    return train_data.loc[train_data['score'].idxmax()]['A']


return_answer('화가납니다')