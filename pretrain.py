#!/usr/bin/env python
# coding: utf-8




userbehavior={}
id=0
from nltk.tokenize import wordpunct_tokenize
with open('behavior.tsv','r') as f:
    line=f.readline()
    while line:
        uetline=line.strip().split('\t')
        userid=uetline[0]
        text=uetline[2].split('#|NTABN|#')
        time=uetline[3].split('#|NTABN|#')
        temp=sorted([[wordpunct_tokenize(text[x]),int(time[x])] for x in range(len(text))],key=lambda x:x[1])
        userbehavior[userid]=temp[:200]
        line=f.readline()
        id+=1
        if id%1000==0:
            print(id)



word_dict0={'PADDING':[0,999999],'UNK':[1,999999]}

for i in userbehavior:
    for k in userbehavior[i]:
        for j in k[0]:
            if j in word_dict0:
                word_dict0[j][1]+=1
            else:
                word_dict0[j]=[len(word_dict0),1]



word_dict={}
for i in word_dict0:
    if word_dict0[i][1]>=30:
        word_dict[i]=[len(word_dict),word_dict0[i][1]]
print(len(word_dict),len(word_dict0))




import numpy as np
embdict={}
plo=0
import pickle
with open('glove.840B.300d.txt','rb')as f:
    linenb=0
    while True:
        j=f.readline()
        if len(j)==0:
            break
        k = j.split()
        word=k[0].decode()
        linenb+=1
        if len(word) != 0:
            tp=[float(x) for x in k[1:]]
            if word in word_dict:
                embdict[word]=tp
                if plo%10000==0:
                    print(plo,linenb,word)
                plo+=1


# In[5]:


from numpy.linalg import cholesky
word_dict1=word_dict
print(len(embdict),len(word_dict1))
print(len(word_dict1))
lister=[0]*len(word_dict1)
xp=np.zeros(300,dtype='float32')

cand=[]
for i in embdict.keys():
    lister[word_dict1[i][0]]=np.array(embdict[i],dtype='float32')
    cand.append(lister[word_dict1[i][0]])
cand=np.array(cand,dtype='float32')

mu=np.mean(cand, axis=0)
Sigma=np.cov(cand.T)

norm=np.random.multivariate_normal(mu, Sigma, 1)
print(mu.shape,Sigma.shape,norm.shape)

for i in range(len(lister)):
    if type(lister[i])==int:
        lister[i]=np.reshape(norm, 300)
lister[0]=np.zeros(300,dtype='float32')
lister=np.array(lister,dtype='float32')
print(lister.shape)


# In[6]:


import random
userbehaviorkeys=list(userbehavior.keys())


# In[7]:


def samplenegrecord(record,userbehaviorkeys,userbehavior):
    i=random.sample(userbehaviorkeys,1)[0]
    j=random.sample(userbehavior[i],1)[0][0]
    while j==record:
        i=random.sample(userbehaviorkeys,1)[0]
        j=random.sample(userbehavior[i],1)[0][0]
    return j


npratio=4
cutnum=2
import numpy as np
def batchgenerator(batch_size):
    
    
    userbehaviorkeysshuffle=list(userbehaviorkeys)
    random.shuffle(userbehaviorkeysshuffle)
    
    batches = [userbehaviorkeysshuffle[batch_size*i:min(len(userbehaviorkeysshuffle), batch_size*(i+1))] for i in range(len(userbehaviorkeysshuffle)//batch_size+1)]

    while (True):
        for bt in batches:

            userbehaviorwords=[]
            userbehaviorwords2=[]
            userbehaviormasker=[]
            userbehaviorpos=[]
            userbehaviorlefttime=[]
            userbehaviorrighttime=[]
            userbehaviorpos2=[]
            userbehaviormaskercandidate=[]
            userbehaviormaskerlabel=[]
            userbehaviorlastcandidate=[]
            pairlabel=[]
            taskmask=[]
            taskmask2=[]
            for i in bt:
                tq=[]
                for j in userbehavior[i]:
                    tp=[]
                    for x in j[0]:
                        if x in word_dict:
                            tp.append(word_dict[x][0])
                        else:
                            tp.append(1)
                    tp=tp[:30]
                    tq.append(tp+[0]*(30-len(tp)))
                tq=tq[:100]
                masknb=int(np.floor((len(tq)-2)*0.2)+1)
                
                maskindex=random.randint(0,(len(tq))-1)
                recordpos=list(range(1,len(tq)+1))+[0]*(100-len(tq))
                lefttime=[0]+[int(np.log2(userbehavior[i][x+1][1]-userbehavior[i][x][1]+1)) for x in range(len(userbehavior[i][:100])-1)]

                righttime=lefttime[1:]+[0]
                    
                lefttime+=[0]*(100-len(tq)) 
                 
                righttime+=[0]*(100-len(tq))
                #print(len(righttime),lefttime)
                tq=tq+[[0]*30]*(100-len(tq))
                candidate=[]  
                candidate_label=[]
                
                nplabel=[0]*npratio+[1]
                npcand=[]
                npindex=np.arange(1+npratio)
                np.random.shuffle(npindex)
                for neg in range(npratio):
                    negone=samplenegrecord(userbehavior[i][maskindex][0],userbehaviorkeys,userbehavior)
                    tp=[]
                    for x in negone:
                        if x in word_dict:
                            tp.append(word_dict[x][0])
                        else:
                            tp.append(1)
                    tp=tp[:30]
                    npcand.append(tp+[0]*(30-len(tp)))
                npcand.append(tq[maskindex]) 
                npcand=np.array(npcand)[npindex]
                nplabel=np.array(nplabel)[npindex]
                
                
                
                
                tq[maskindex]=[0]*30
                    
                userbehaviormasker.append([maskindex]*(1+npratio))    
                userbehaviorwords.append(tq)
                userbehaviorpos.append(recordpos)
                userbehaviorlefttime.append(lefttime)
                userbehaviorrighttime.append(righttime)
                userbehaviormaskercandidate.append(npcand)
                userbehaviormaskerlabel.append(nplabel)
                userbehaviorlastcandidate.append([[[0]*30]*(1+npratio)]*cutnum)
                pairlabel.append([np.array([0]*(1+npratio))]*cutnum)
                taskmask.append(1)
                taskmask2.append([0]*cutnum)
                
            for i in bt:
                tq=[]
                for j in userbehavior[i]:
                    tp=[]
                    for x in j[0]:
                        if x in word_dict:
                            tp.append(word_dict[x][0])
                        else:
                            tp.append(1)
                    tp=tp[:30]
                    tq.append(tp+[0]*(30-len(tp)))
                tq=tq[:100]
                
                
                candidates=[]  
                candidates_label=[]
                
                for maskid in range(min(cutnum,len(tq))):
                    nplabel=[0]*npratio+[1]
                    npcand=[]
                    npindex=np.arange(1+npratio)
                    np.random.shuffle(npindex)
                    for neg in range(npratio):
                        negone=samplenegrecord(userbehavior[i][-1-maskid][0],userbehaviorkeys,userbehavior)
                        tp=[]
                        for x in negone:
                            if x in word_dict:
                                tp.append(word_dict[x][0])
                            else:
                                tp.append(1)
                        tp=tp[:30]
                        npcand.append(tp+[0]*(30-len(tp)))
                    
                    npcand.append(tq[-1-maskid]) 
                    tq[-1-maskid]=[0]*30
                    npcand=np.array(npcand)[npindex]
                    nplabel=np.array(nplabel)[npindex]
                    candidates.append(npcand)
                    candidates_label.append(nplabel)
                
                
                candidates_label+=[np.array([0]*(1+npratio))]*(cutnum-len(candidates))
                candidates+=[[[0]*30]*(1+npratio)]*(cutnum-len(candidates))
                
                temptqlen=len(tq)
                
                tq=tq+[[0]*30]*(100-len(tq))
                
                userbehaviormaskercandidate.append([[0]*30]*5)
                userbehaviormaskerlabel.append([0]*5)
                userbehaviormasker.append([0]*(1+npratio))   
                userbehaviorlastcandidate.append(candidates)
                
                userbehaviorwords.append(tq)
                userbehaviorpos.append(recordpos)
                userbehaviorlefttime.append(lefttime)
                userbehaviorrighttime.append(righttime)
                pairlabel.append(candidates_label)
                taskmask.append(0)
                taskmask2.append([1]*min(cutnum,temptqlen)+[0]*(cutnum-min(cutnum,temptqlen)))

                
            userbehaviorwords=np.array(userbehaviorwords,dtype='int32')
            userbehaviorpos=np.array(userbehaviorpos,dtype='int32')
            userbehaviorlefttime=np.array(userbehaviorlefttime,dtype='int32')
            userbehaviorrighttime=np.array(userbehaviorrighttime,dtype='int32')
            userbehaviorlastcandidate=np.array(userbehaviorlastcandidate,dtype='int32')

            pairlabel=np.array(pairlabel,dtype='float32')
            taskmask=np.array(taskmask,dtype='float32')
            taskmask2=np.array(taskmask2,dtype='float32')
            userbehaviormasker=np.array(userbehaviormasker,dtype='int32')
            userbehaviormaskercandidate=np.array(userbehaviormaskercandidate,dtype='int32')
            userbehaviormaskerlabel=np.array(userbehaviormaskerlabel,dtype='int32')
            yield ([userbehaviorwords,userbehaviormaskercandidate,userbehaviorpos,userbehaviorrighttime,userbehaviormasker]+[userbehaviorlastcandidate[:,k,:,:] for k in range(userbehaviorlastcandidate.shape[1])],
                   [userbehaviormaskerlabel,pairlabel],[taskmask,taskmask2])




from keras.utils.np_utils import to_categorical
from keras.layers import *
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers #keras2
from sklearn.metrics import accuracy_score, classification_report
from keras.optimizers import *
import keras
from keras.layers import *



class Attention(Layer):

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ', 
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK', 
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV', 
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)
        
    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12
                
    def call(self, x):
        if len(x) == 3:
            Q_seq,K_seq,V_seq = x
            Q_len,V_len = None,None
        elif len(x) == 5:
            Q_seq,K_seq,V_seq,Q_len,V_len = x
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0,2,1,3))
        A = K.batch_dot(Q_seq, K_seq, axes=[3,3]) / self.size_per_head**0.5
        A = K.permute_dimensions(A, (0,3,2,1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0,3,2,1))    
        A = K.softmax(A)
        O_seq = K.batch_dot(A, V_seq, axes=[3,2])
        O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)

keras.backend.clear_session()

MAX_SENT_LENGTH=30
MAX_SENTS=100

sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')

embedding_layer = Embedding(len(word_dict), 300, weights=[lister], trainable=True)
wordposition_emb=Embedding(MAX_SENT_LENGTH+1, 300, trainable=True)
embedded_sequences = embedding_layer(sentence_input)
wordpos_embedded_sequences = wordposition_emb(Lambda(lambda x:K.zeros_like(x,dtype='int32')+K.arange(x.shape[1]))(sentence_input))
embedded_sequences =add([embedded_sequences,wordpos_embedded_sequences ])
wordrep=Dropout(0.2)(embedded_sequences)


wordrep = Attention(16,16)([wordrep,wordrep,wordrep])
d_wordrep=Dropout(0.2)(wordrep)

attention = Dense(200,activation='tanh')(d_wordrep)
attention = Flatten()(Dense(1)(attention))
attention_weight = Activation('softmax')(attention)
l_att=Dot((1, 1))([d_wordrep, attention_weight])



textencoder = Model([sentence_input], l_att)


behavior_input = Input((MAX_SENTS,MAX_SENT_LENGTH,), dtype='int32') 
candidate_vecs = TimeDistributed(textencoder)(behavior_input)
behaviorpos_input = Input((MAX_SENTS,), dtype='int32') 
posembedding_layer = Embedding(MAX_SENTS+cutnum+1, 256, trainable=True)
posembedding=posembedding_layer(behaviorpos_input)

timeembedding_layer = Embedding(32, 256, trainable=True)

timeinterval= Input((MAX_SENTS,), dtype='int32') 
timeembedding=timeembedding_layer(timeinterval)
candidate_vecs=add([candidate_vecs,posembedding,timeembedding])
behaviorrep = Attention(16,16)([candidate_vecs,candidate_vecs,candidate_vecs])
d_behaviorrep=Dropout(0.2)(behaviorrep)
attentionu = Dense(200,activation='tanh')(d_behaviorrep)
attentionu = Flatten()(Dense(1)(attentionu))
attentionu_weight = Activation('softmax')(attentionu)
l_attu=Dot((1, 1))([d_behaviorrep, attentionu_weight])



candidates =Input((1+npratio,MAX_SENT_LENGTH), dtype='int32')
candidatespos =Input((1+npratio,), dtype='int32')
candidateposembedding=posembedding_layer(candidatespos)

candidate_vecs = TimeDistributed(textencoder)(candidates)
candidate_vecs=add([candidate_vecs,candidateposembedding])


candidates_next =[Input((1+npratio,MAX_SENT_LENGTH), dtype='int32')for _ in range(cutnum)]

candidates_nextpos = [posembedding_layer(Lambda(lambda x:K.ones_like(x,dtype='int32')*(MAX_SENTS+_+1))(candidatespos)) for _ in range(cutnum)]

candidate_vecs_next = [add([TimeDistributed(textencoder)(candidates_next[_]),candidates_nextpos[_]]) for _ in range(len(candidates_next))]


logits2 =[Activation(keras.activations.softmax)(dot([l_attu, p], axes=-1)) for p in candidate_vecs_next]
logits2 = concatenate([Lambda(lambda x: K.expand_dims(x,axis=1))(news) for news in logits2],axis=1)



logits = dot([l_attu, candidate_vecs], axes=-1)
logits = Activation(keras.activations.softmax)(logits)

model = Model([behavior_input,candidates,behaviorpos_input,  timeinterval,candidatespos]+candidates_next, [logits,logits2])
model.compile(loss=['categorical_crossentropy']*2, optimizer=Adam(lr=0.0001), metrics=['acc'],sample_weight_mode=['None','temporal'])

model0= Model([behavior_input,behaviorpos_input,  timeinterval], [l_attu])

for ep in range(2):
    traingen=batchgenerator(32)#real batchsize=32*2=64
    model.fit_generator(traingen, epochs=1,steps_per_epoch=len(userbehaviorkeys)//32)
model0.save_weights('pretrainedum.h5')

