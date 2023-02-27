import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import torch

def QNAI_accuracy_calc(y_pred, y_true):
        pred_mask_f1 = torch.where(y_pred == 0.0, 0, 1).long()
        true_mask_f1 = torch.where(y_true == 0.0, 0, 1).long()
        final_score = 0
        for col in range(6):
            tp = sum([1 for idx in range(y_pred.size(0)) if pred_mask_f1[idx][col]==1 and pred_mask_f1[idx][col]==true_mask_f1[idx][col]])
            if tp == 0:
                continue
            precision_denom = pred_mask_f1[:, col].sum().item()
            recall_denom = true_mask_f1[:, col].sum().item()

            precision = tp/precision_denom
            recall = tp/recall_denom

            f1_score = (2*precision*recall)/(precision+recall)
            rss = ((y_true[:, col] - y_pred[:, col])**2).sum().item()
            k = 16*y_pred.size(0)

            r2_score = 1 - rss/k
            final_score += f1_score*r2_score
        return final_score/6

def sentihood_strict_acc(y_true, y_pred):
    """
    Calculate "strict Acc" of aspect detection task of Sentihood.
    """
    total_cases=int(len(y_true)/4)
    true_cases=0
    for i in range(total_cases):
        if y_true[i*4]!=y_pred[i*4]:continue
        if y_true[i*4+1]!=y_pred[i*4+1]:continue
        if y_true[i*4+2]!=y_pred[i*4+2]:continue
        if y_true[i*4+3]!=y_pred[i*4+3]:continue
        true_cases+=1
    aspect_strict_Acc = true_cases/total_cases
    return aspect_strict_Acc


def sentihood_macro_F1(y_true, y_pred):
    """
    Calculate "Macro-F1" of aspect detection task of Sentihood.
    """
    p_all, r_all, count=0, 0, 0
    for i in range(len(y_pred)//4):
        a, b = set(), set()
        for j in range(4):
            if y_pred[i*4+j]!=0:
                a.add(j)
            if y_true[i*4+j]!=0:
                b.add(j)
        if len(b)==0: continue
        a_b=a.intersection(b)
        if len(a_b)>0:
            p=len(a_b)/len(a)
            r=len(a_b)/len(b)
        else:
            p, r = 0, 0
        count+=1
        p_all+=p
        r_all+=r
    Ma_p=p_all/count
    Ma_r=r_all/count
    # avoid zero division
    aspect_Macro_F1 = 0 if Ma_p+Ma_r==0 else 2*Ma_p*Ma_r/(Ma_p+Ma_r)
    return aspect_Macro_F1


def sentihood_AUC_Acc(y_true, score):
    """
    Calculate "Macro-AUC" of both aspect detection and sentiment classification tasks of Sentihood.
    Calculate "Acc" of sentiment classification task of Sentihood.
    """
    # aspect-Macro-AUC
    aspect_y_true=[]
    aspect_y_score=[]
    aspect_y_trues=[[],[],[],[]]
    aspect_y_scores=[[],[],[],[]]
    for i in range(len(y_true)):
        if y_true[i]>0:
            aspect_y_true.append(0)
        else:
            aspect_y_true.append(1) # "None": 1
        tmp_score=score[i][0] # probability of "None"
        aspect_y_score.append(tmp_score)
        aspect_y_trues[i%4].append(aspect_y_true[-1])
        aspect_y_scores[i%4].append(aspect_y_score[-1])

    aspect_auc=[]
    for i in range(4):
        aspect_auc.append(metrics.roc_auc_score(aspect_y_trues[i], aspect_y_scores[i]))
    aspect_Macro_AUC = np.mean(aspect_auc)
    
    # sentiment-Macro-AUC
    sentiment_y_true=[]
    sentiment_y_pred=[]
    sentiment_y_score=[]
    sentiment_y_trues=[[],[],[],[]]
    sentiment_y_scores=[[],[],[],[]]
    for i in range(len(y_true)):
        if y_true[i]>0:
            sentiment_y_true.append(y_true[i]-1) # "Postive":0, "Negative":1
            tmp_score=score[i][2]/(score[i][1]+score[i][2])  # probability of "Negative"
            sentiment_y_score.append(tmp_score)
            if tmp_score>0.5:
                sentiment_y_pred.append(1) # "Negative": 1
            else:
                sentiment_y_pred.append(0)
            sentiment_y_trues[i%4].append(sentiment_y_true[-1])
            sentiment_y_scores[i%4].append(sentiment_y_score[-1])

    sentiment_auc=[]
    for i in range(4):
        sentiment_auc.append(metrics.roc_auc_score(sentiment_y_trues[i], sentiment_y_scores[i]))
    sentiment_Macro_AUC = np.mean(sentiment_auc)

    # sentiment Acc
    sentiment_y_true = np.array(sentiment_y_true)
    sentiment_y_pred = np.array(sentiment_y_pred)
    sentiment_Acc = metrics.accuracy_score(sentiment_y_true,sentiment_y_pred)

    return aspect_Macro_AUC, sentiment_Acc, sentiment_Macro_AUC

#%%
def semeval_PRF(y_true, y_pred):
    """
    Calculate "Micro P R F" of aspect detection task of SemEval-2014.
    """
    s_all, g_all, s_g_all = 0, 0, 0
    for i in range(len(y_pred)//5):
        s,g=set(),set()
        for j in range(5):
            if y_pred[i*5+j]!=4:
                s.add(j)
            if y_true[i*5+j]!=4:
                g.add(j)
        if len(g)==0:continue
        s_g=s.intersection(g)
        s_all+=len(s)
        g_all+=len(g)
        s_g_all+=len(s_g)

    p = 0.0 if s_all == 0 else s_g_all/s_all # avoid zero division
    r = 0.0 if g_all == 0 else s_g_all/g_all # avoid zero division
    f = 0.0 if (p+r) == 0 else 2*p*r/(p+r) # avoid zero division

    return p,r,f


def semeval_Acc(y_true, y_pred, score, classes=4):
    """
    Calculate "Acc" of sentiment classification task of SemEval-2014.
    """
    assert classes in [2, 3, 4], "classes must be 2 or 3 or 4."

    if classes == 4:
        total, total_right = 0, 0
        for i in range(len(y_true)):
            if y_true[i]==4:continue
            total+=1
            tmp=y_pred[i]
            if tmp==4:
                if score[i][0]>=score[i][1] and score[i][0]>=score[i][2] and score[i][0]>=score[i][3]:
                    tmp=0
                elif score[i][1]>=score[i][0] and score[i][1]>=score[i][2] and score[i][1]>=score[i][3]:
                    tmp=1
                elif score[i][2]>=score[i][0] and score[i][2]>=score[i][1] and score[i][2]>=score[i][3]:
                    tmp=2
                else:
                    tmp=3
            if y_true[i]==tmp:
                total_right+=1
        sentiment_Acc = total_right/total
    elif classes == 3:
        total=0
        total_right=0
        for i in range(len(y_true)):
            if y_true[i]>=3:continue
            total+=1
            tmp=y_pred[i]
            if tmp>=3:
                if score[i][0]>=score[i][1] and score[i][0]>=score[i][2]:
                    tmp=0
                elif score[i][1]>=score[i][0] and score[i][1]>=score[i][2]:
                    tmp=1
                else:
                    tmp=2
            if y_true[i]==tmp:
                total_right+=1
        sentiment_Acc = total_right/total
    else:
        total=0
        total_right=0
        for i in range(len(y_true)):
            if y_true[i]>=3 or y_true[i]==1:continue
            total+=1
            tmp=y_pred[i]
            if tmp>=3 or tmp==1:
                if score[i][0]>=score[i][2]:
                    tmp=0
                else:
                    tmp=2
            if y_true[i]==tmp:
                total_right+=1
        sentiment_Acc = total_right/total
    return sentiment_Acc
