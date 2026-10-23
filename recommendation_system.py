import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("social_media_interactions.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
weights = {"view":1, "like":3, "comment":4, "share":5}
df["engagement"] = df["interaction_type"].map(weights)

user_item = df.groupby(["user_id","item_id"], as_index=False)["engagement"].sum()
matrix = user_item.pivot_table(index="user_id", columns="item_id",
                               values="engagement", fill_value=0)

item_similarity = cosine_similarity(matrix.T)
item_sim = pd.DataFrame(item_similarity, index=matrix.columns,
                        columns=matrix.columns)

def recommend(user_id, k=5):
    scores = pd.Series(0.0, index=matrix.columns)
    user_vector = matrix.loc[user_id]
    for item, value in user_vector[user_vector > 0].items():
        scores += item_sim[item] * value
    seen = list(user_vector[user_vector > 0].index)
    scores.loc[seen] = -np.inf
    return scores.nlargest(k).index.tolist()

def evaluate(k=5):
    hits=[]; ndcgs=[]
    for user in matrix.index:
        u = df[df.user_id==user].sort_values("timestamp")
        if len(u)<8: continue
        test_item=u.iloc[-1].item_id
        train=u.iloc[:-1]
        m=train.groupby(["user_id","item_id"])["engagement"].sum().unstack(fill_value=0)
        sim=cosine_similarity(m.T)
        sim=pd.DataFrame(sim,index=m.columns,columns=m.columns)
        scores=pd.Series(0.0,index=m.columns)
        for item,val in m.loc[user][m.loc[user]>0].items():
            scores += sim[item]*val
        scores.loc[list(m.loc[user][m.loc[user]>0].index)]=-np.inf
        recs=scores.nlargest(k).index.tolist()
        if test_item in recs:
            hits.append(1); ndcgs.append(1/np.log2(recs.index(test_item)+2))
        else:
            hits.append(0); ndcgs.append(0)
    precision=sum(hits)/(len(hits)*k)
    recall=sum(hits)/len(hits)
    ndcg=float(np.mean(ndcgs))
    return precision, recall, ndcg

if __name__ == "__main__":
    print("Top-5:", recommend("U001",5))
    print("Precision@5, Recall@5, NDCG@5:", evaluate(5))
