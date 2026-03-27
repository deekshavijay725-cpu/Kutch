import streamlit as st
import pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

st.set_page_config(layout="wide")
st.title("Kutch Craft Analytics Dashboard")
df=pd.read_csv("kutch_survey_dataset_2000.csv")
st.dataframe(df.head())

fig, ax = plt.subplots()
df["Age_Group"].value_counts().plot(kind="bar", ax=ax)
ax.set_title("Age Distribution")
st.pyplot(fig)
st.caption("Shows dominant age groups for campaign targeting.")

X=df.drop(columns=["Purchase_Interest"])
y=df["Purchase_Interest"].map(lambda x: 1 if x in ["Highly likely","Likely"] else 0)

pre=ColumnTransformer([("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),("oh", OneHotEncoder(handle_unknown="ignore"))]), X.columns.tolist())])
models={"Decision Tree":DecisionTreeClassifier(random_state=42),"Random Forest":RandomForestClassifier(random_state=42),"Gradient Boosting":GradientBoostingClassifier(random_state=42)}
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

results=[]
figroc, axroc = plt.subplots()
for n,m in models.items():
    pipe=Pipeline([("pre",pre),("m",m)])
    pipe.fit(X_train,y_train)
    pred=pipe.predict(X_test)
    prob=pipe.predict_proba(X_test)[:,1]
    results.append({"Model":n,"Train Accuracy":pipe.score(X_train,y_train),"Test Accuracy":accuracy_score(y_test,pred),"Precision":precision_score(y_test,pred),"Recall":recall_score(y_test,pred),"F1":f1_score(y_test,pred)})
    fpr,tpr,_=roc_curve(y_test,prob)
    axroc.plot(fpr,tpr,label=f"{n} AUC={auc(fpr,tpr):.2f}")
    cm=confusion_matrix(y_test,pred)
    figcm, axcm = plt.subplots()
    axcm.imshow(cm)
    for i in range(2):
        for j in range(2):
            axcm.text(j,i,str(cm[i,j]),ha="center")
    axcm.set_title(f"{n} Confusion Matrix")
    st.pyplot(figcm)

axroc.plot([0,1],[0,1],"--")
axroc.legend()
st.pyplot(figroc)
st.dataframe(pd.DataFrame(results))

uploaded=st.file_uploader("Upload test CSV")
if uploaded:
    new=pd.read_csv(uploaded)
    final=Pipeline([("pre",pre),("m",RandomForestClassifier(random_state=42))])
    final.fit(X,y)
    new["Predicted_Interest"]=final.predict(new)
    st.dataframe(new.head())
    st.download_button("Download predictions", new.to_csv(index=False).encode(), "predicted_results.csv")
