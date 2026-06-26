import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report



df=pd.read_csv("interview_project/Loan_Approval.csv")

print(df.head())
print(df.describe())
print(df.isnull().sum())

df["Gender"] = df["Gender"].fillna(
    df["Gender"].mode()[0]
)
df["Self_Employed"]=df["Self_Employed"].fillna(df["Self_Employed"].mode()[0])
df["Credit_History"]=df["Credit_History"].fillna(df["Credit_History"].mode()[0])

df["LoanAmount"]=df["LoanAmount"].fillna(df["LoanAmount"].median())
df["Loan_Amount_Term"]=df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median())

df.dropna(
    subset=["Loan_Status"],
    inplace=True
)

print("after filna")
print(df.isnull().sum())

print("duplicate")
print(df.duplicated().sum())

print("remove duplicte")
print(df.drop_duplicates(inplace=True))

df.drop(
    "Loan_ID",
    axis=1,
    inplace=True
)


sns.countplot(
    x="Loan_Status",
    data=df
)

plt.title("Loan Approval Count")
plt.show()

sns.countplot(
    x="Education",
    hue="Loan_Status",
    data=df
)

plt.title("Education vs Loan Approval")
plt.show()

# Property Area vs Loan Status

sns.countplot(
    x="Property_Area",
    hue="Loan_Status",
    data=df
)

plt.title("Property Area vs Loan Approval")
plt.show()

# Applicant Income Distribution

plt.figure(figsize=(8,5))

sns.histplot(
    df["ApplicantIncome"],
    bins=10,
    kde=True
)
plt.title("Applicant Income Distribution")
plt.show()


# Convert Categorical Data
df["Gender"] = df["Gender"].map({ "Male":1,"Female":0})

df["Married"] = df["Married"].map({"Yes":1,"No":0})

df["Education"] = df["Education"].map({"Graduate":1,"Not Graduate":0})

df["Self_Employed"] = df["Self_Employed"].map({"Yes":1,"No":0})

df["Loan_Status"] = df["Loan_Status"].map({"Y":1,"N":0})

df["Dependents"] = df["Dependents"].replace({"3+":3})

df["Dependents"] = df["Dependents"].astype(int)

df["Property_Area"] = df["Property_Area"].map({ "Urban":2,"Semiurban":1,"Rural":0})

# Correlation Heatmap
plt.figure(figsize=(10,6))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True
)

plt.title("Correlation Heatmap")
plt.show()

#Feature Selection
X = df[
    [
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Property_Area"
    ]
]

y = df["Loan_Status"]


X_train, X_test, y_train, y_test = train_test_split( X,y,test_size=0.20,random_state=42)

model = LogisticRegression(max_iter=1000)

model.fit(X_train,y_train)

print("\nModel Trained Successfully")


y_pred = model.predict(X_test)


accuracy = accuracy_score( y_test,y_pred)

print("\n========== ACCURACY ==========")
print(round(accuracy * 100,2), "%")

cm = confusion_matrix(y_test, y_pred)

print("\n========== CONFUSION MATRIX ==========")
print(cm)

print("\n========== CLASSIFICATION REPORT ==========")

print( classification_report( y_test, y_pred))
new_customer = pd.DataFrame({
    "Gender":[1],
    "Married":[1],
    "Dependents":[1],
    "Education":[1],
    "Self_Employed":[0],
    "ApplicantIncome":[6000],
    "CoapplicantIncome":[2000],
    "LoanAmount":[120],
    "Loan_Amount_Term":[360],
    "Credit_History":[1],
    "Property_Area":[2]
})

prediction = model.predict(new_customer)

print("\n========== LOAN PREDICTION ==========")

if prediction[0] == 1:
    print(" Loan Approved")
else:
    print(" Loan Rejected")