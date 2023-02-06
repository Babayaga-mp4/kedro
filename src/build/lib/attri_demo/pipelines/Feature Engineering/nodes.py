
from string import ascii_letters
import random
import pandas as pd
def load_dataset(chunked_df):
    chunked_df = chunked_df.drop(["Time"] ,axis = 1)
    return chunked_df

def statistical_feature_engineering(df):
    return df

def SNA_feature_engineering(df):
    return df

def feature_selection(df,df2):
    return df

def feature_transformation(df,df2,df3):
    return df

def data_viz(df):
    return df, df

def collect_dataset(Data_Source_1, Data_Source_2, Data_Source_3):
    return pd.DataFrame[Data_Source_1, Data_Source_2, Data_Source_3]

def random_string_generator(size=10, chars=ascii_letters):
    df={}
    df["udmi"]="".join(random.choice(chars) for _ in range(size))
    return pd.DataFrame(df,index=[0])
