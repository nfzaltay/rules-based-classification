###############################################
# PROJECT : Potential Customer Return Calculation with Rule-Based Classification
# Main object: Recognizing a simple data and by performing certain data arrangements
#              calculating the potential return of mobile game market users
###############################################

import pandas as pd

# Reading persona.csv as pandas DataFrame
df = pd.read_csv(r"data/persona.csv")

# Analyze descriptive statistics
df.head()
df.shape
df.describe().T
df.isnull().values.any()
df.isnull().sum()

# Number of unique <SOURCE>
df["SOURCE"].nunique()
df["SOURCE"].value_counts()

# Number of unique <PRICE>

df["PRICE"].nunique()

# Frequency of <COUNTRY>

df["COUNTRY"].value_counts()

# Country breakdown of sales

df.groupby("COUNTRY").agg({"PRICE": "sum"})

# Country breakdown of sales

df['SOURCE'].value_counts()

# Country breakdown of income averages

df.groupby("COUNTRY")["PRICE"].agg({"mean"})

# Country and Source breakdown of income averages

df.groupby(["COUNTRY", 'SOURCE'])["PRICE"].mean()

# Average income on the basis of variables,

agg_df = df.groupby(["COUNTRY", 'SOURCE', "SEX", "AGE"])["PRICE"].mean().sort_values(ascending=False)

# Convert the index names to variable names

agg_df = agg_df.reset_index()
agg_df.head()

# Convert AGE variable to categorical variable and adding it to agg_df

my_labels = ['0_18', '19_23', '24_30', '31_40', '41_70']
agg_df["AGE_CUT"] = pd.cut(x=agg_df["AGE"], bins=[0, 18, 23, 30, 40, 70], labels=my_labels)
agg_df.tail(10)

# Identify new level-based customers (Personas)

agg_df["customers_level_based"] = [f"{i[0]}_{i[1]}_{i[2]}_{i[-1]}" for i in agg_df.values]

agg_df = agg_df.loc[:, ["customers_level_based", "PRICE"]].groupby("customers_level_based") \
    .agg({"PRICE": "mean"}).sort_values(by="PRICE", ascending=False).reset_index()


agg_df["customers_level_based"].head()

# Segment new customers (Personas)

agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])

# Describe the segments and especially "C"

agg_df.groupby(["SEGMENT"]).agg({"PRICE": ["mean", "max", "sum"]})

agg_df[agg_df["SEGMENT"] == "C"].describe()

# Classify new customers by segment and estimate how much income they bring
# What segment does a 33-year-old Turkish woman using ANDROID belong to
# and how much income is expected to earn on average?

new_user = "TUR_ANDROID_FEMALE_31_40"
print(agg_df[agg_df["customers_level_based"] == new_user])

#      customers_level_based      PRICE SEGMENT
#   3  TUR_ANDROID_FEMALE_31_40  41.833333       A


# What segment does a 35-year-old French woman using IOS belong to
# and how much income is expected to earn on average?

new_user = "FRA_IOS_FEMALE_31_40"
print(agg_df[agg_df["customers_level_based"] == new_user])

#      customers_level_based      PRICE SEGMENT
#   78  FRA_IOS_FEMALE_31_40  32.818182       C
