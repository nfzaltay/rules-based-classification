###############################################
# PROJECT : Potential Customer Return Calculation with Rule-Based Classification
# Main object: Recognizing a simple data and by performing certain data arrangements
#              calculating the potential return of mobile game market users

# A game company wants to create level-based new customer definitions (personas) by using some
# features ( Country, Source, Age, Sex) of its customers, and to create segments according to these new customer
# definitions and to estimate how much profit can be generated from  the new customers according to these segments.
###############################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reading persona.csv as pandas DataFrame
df = pd.read_csv(r"data/persona.csv")


def check_df(dataframe):
    # Analyze descriptive statistics
    print(f"""
        ##################### Shape #####################\n\n\t{dataframe.shape}\n
        ##################### Types #####################\n\n{dataframe.dtypes}\n
        ##################### Head #####################\n\n{dataframe.head(3)}\n
        ##################### NA #####################\n\n{dataframe.isnull().sum()}\n
        ##################### Quantiles #####################\n\n{dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T}
        \n""")


check_df(df)


def num_summary(dataframe, plot=False):
    numerical_col = ['PRICE', 'AGE']
    quantiles = [0.25, 0.50, 0.75, 1]
    for col_name in numerical_col:
        print("########## Summary Statistics of " + col_name + " ############")
        print(dataframe[numerical_col].describe(quantiles).T)

        if plot:
            sns.histplot(data=dataframe, x=col_name)
            plt.xlabel(col_name)
            plt.title("The distribution of " + col_name)
            plt.grid(True)
            plt.show(block=True)


num_summary(df, plot=True)

def data_analysis(dataframe):
    # Data Analysis
    # Number of unique <SOURCE>
    print("Unique Values of Source:\n", dataframe["SOURCE"].nunique())
    print("Frequency of Source:\n", dataframe["SOURCE"].value_counts())

    # Number of unique <PRICE>
    print("Unique Values of Price:\n", dataframe["PRICE"].nunique())

    # Frequency of <COUNTRY>
    print("Number of product sales by country:\n", dataframe["COUNTRY"].value_counts())

    # Total & average amount of sales by country
    print("Total & average amount of sales by country:\n", dataframe.groupby("COUNTRY").agg({"PRICE": ["mean", "sum"]}))

    # Average amount of sales by source and country:
    print("Average amount of sales by source and country:\n", dataframe.pivot_table(values=['PRICE'],
                                                                                    index=['COUNTRY'],
                                                                                    columns=["SOURCE"],
                                                                                    aggfunc=["mean"]))


data_analysis(df)

# Average income on the basis of variables,

agg_df = df.groupby(["COUNTRY", 'SOURCE', "SEX", "AGE"])["PRICE"].mean().sort_values(ascending=False)

# Convert the index names to variable names

agg_df = agg_df.reset_index()
print(agg_df.head())

# Convert AGE variable to categorical variable and adding it to agg_df

my_labels = ['0_18', '19_23', '24_30', '31_40', '41_70']
agg_df["AGE_CUT"] = pd.cut(x=agg_df["AGE"], bins=[0, 18, 23, 30, 40, 70], labels=my_labels)
print(agg_df.tail(10))

# Identify new level-based customers (Personas)

agg_df["customers_level_based"] = [f"{i[0]}_{i[1]}_{i[2]}_{i[-1]}" for i in agg_df.values]

agg_df = agg_df.loc[:, ["customers_level_based", "PRICE"]].groupby("customers_level_based") \
    .agg({"PRICE": "mean"}).sort_values(by="PRICE", ascending=False).reset_index()


print(agg_df["customers_level_based"].head())

# Segment new customers (Personas)

agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])

# Describe the segments and especially "C"

agg_df.groupby(["SEGMENT"]).agg({"PRICE": ["mean", "max", "sum"]})

agg_df[agg_df["SEGMENT"] == "C"].describe()

# Classify new customers by segment and estimate how much income they bring
# What segment does a 25-year-old French man using ANDROID belong to
# and how much income is expected to earn on average?

new_user = "fra_android_male_24_30"
print(agg_df[agg_df["customers_level_based"] == new_user])

#      customers_level_based  PRICE SEGMENT
#  74  fra_android_male_24_30   33.0       C


# What segment does a 35-year-old Turkish woman using IOS belong to
# and how much income is expected to earn on average?

new_user = "tur_ios_female_31_40"
print(agg_df[agg_df["customers_level_based"] == new_user])

#    customers_level_based      PRICE SEGMENT
#  81  tur_ios_female_31_40  32.333333       D
