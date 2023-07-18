from keras.datasets import mnist, fashion_mnist
import random
import numpy as np
import pandas as pd


def create_all_datasets_for_exp():
    all_datasets = {
        "MNIST_10%": MNIST_10_perc(),
        "MNIST_1%": MNIST_1_perc(),
        "Fashion_MNIST_10%": Fashion_MNIST_10_perc(),
        "Fashion_MNIST_1%": Fashion_MNIST_1_perc(),
        "Breast_cancer": breast_cancer(),
        "California_13%": california_13_perc(),
        "California_5%": california_5_perc(),
        "Creditcard_10%": credit_card_10_perc(), 
        "Creditcard_2%": credit_card_2_perc(),
        "White_wine": white_wine()
    }
    return all_datasets





def MNIST_10_perc():
    (X_tr, y_tr), (X_test, y_test) = mnist.load_data()

    def reshape_mnist(X):
        Datas = pd.DataFrame(X.reshape(X.shape[0], X.shape[1] * X.shape[2]))
        Datas.index.name = "num_sample"
        return Datas

    X_tr, X_test = reshape_mnist(X_tr), reshape_mnist(X_test)

    y_tr, y_test = pd.Series(y_tr.reshape(y_tr.shape[0]), name="y"), pd.Series(y_test.reshape(y_test.shape[0]), name="y")
    y_tr.index.name, y_test.index.name = "num_sample", "num_sample"

    def binary_prob(number):
        if number == 0:
            return 1
        else:
            return 0
        
    y_tr, y_test = y_tr.apply(binary_prob), y_test.apply(binary_prob)

    X = pd.concat([X_tr, X_test])
    X.index = [i for i in range(X.shape[0])]
    y = pd.concat([y_tr, y_test])
    y.index = [i for i in range(y.shape[0])]
    
    return X, y
    
    
def MNIST_1_perc():
    (X_tr, y_tr), (X_test, y_test) = mnist.load_data()

    def reshape_mnist(X):
        Datas = pd.DataFrame(X.reshape(X.shape[0], X.shape[1] * X.shape[2]))
        Datas.index.name = "num_sample"
        return Datas

    X_tr, X_test = reshape_mnist(X_tr), reshape_mnist(X_test)

    y_tr, y_test = pd.Series(y_tr.reshape(y_tr.shape[0]), name="y"), pd.Series(y_test.reshape(y_test.shape[0]), name="y")
    y_tr.index.name, y_test.index.name = "num_sample", "num_sample"

    def binary_prob(number):
        if number == 0:
            return 1
        else:
            return 0
        
    y_tr, y_test = y_tr.apply(binary_prob), y_test.apply(binary_prob)

    # Select only 1% aproximately of positive class, instead of 10%
    y_tr_pos = y_tr[y_tr == 1]
    rows_pos_sample_to_remove_tr = random.sample(list(y_tr_pos.index), int(len(y_tr_pos) * 0.9))
    y_tr = y_tr.drop(rows_pos_sample_to_remove_tr)
    X_tr = X_tr.drop(rows_pos_sample_to_remove_tr)

    y_test_pos = y_test[y_test == 1]
    rows_pos_sample_to_remove_test = random.sample(list(y_test_pos.index), int(len(y_test_pos) * 0.9))
    y_test = y_test.drop(rows_pos_sample_to_remove_test)
    X_test = X_test.drop(rows_pos_sample_to_remove_test)


    X = pd.concat([X_tr, X_test])
    X.index = [i for i in range(X.shape[0])]
    y = pd.concat([y_tr, y_test])
    y.index = [i for i in range(y.shape[0])]
    
    return X, y
    
    
def Fashion_MNIST_10_perc():
    (X_tr, y_tr), (X_test, y_test) = fashion_mnist.load_data()

    def reshape_mnist(X):
        Datas = pd.DataFrame(X.reshape(X.shape[0], X.shape[1] * X.shape[2]))
        Datas.index.name = "num_sample"
        return Datas

    X_tr, X_test = reshape_mnist(X_tr), reshape_mnist(X_test)

    y_tr, y_test = pd.Series(y_tr.reshape(y_tr.shape[0]), name="y"), pd.Series(y_test.reshape(y_test.shape[0]), name="y")
    y_tr.index.name, y_test.index.name = "num_sample", "num_sample"

    def binary_prob(number):
        if number == 0:
            return 1
        else:
            return 0
        
    y_tr, y_test = y_tr.apply(binary_prob), y_test.apply(binary_prob)

    X = pd.concat([X_tr, X_test])
    X.index = [i for i in range(X.shape[0])]
    y = pd.concat([y_tr, y_test])
    y.index = [i for i in range(y.shape[0])]
    
    return X, y
    
    
def Fashion_MNIST_1_perc():
    (X_tr, y_tr), (X_test, y_test) = fashion_mnist.load_data()

    def reshape_mnist(X):
        Datas = pd.DataFrame(X.reshape(X.shape[0], X.shape[1] * X.shape[2]))
        Datas.index.name = "num_sample"
        return Datas

    X_tr, X_test = reshape_mnist(X_tr), reshape_mnist(X_test)

    y_tr, y_test = pd.Series(y_tr.reshape(y_tr.shape[0]), name="y"), pd.Series(y_test.reshape(y_test.shape[0]), name="y")
    y_tr.index.name, y_test.index.name = "num_sample", "num_sample"

    def binary_prob(number):
        if number == 0:
            return 1
        else:
            return 0
        
    y_tr, y_test = y_tr.apply(binary_prob), y_test.apply(binary_prob)

    # Select only 1% aproximately of positive class, instead of 10%
    y_tr_pos = y_tr[y_tr == 1]
    rows_pos_sample_to_remove_tr = random.sample(list(y_tr_pos.index), int(len(y_tr_pos) * 0.9))
    y_tr = y_tr.drop(rows_pos_sample_to_remove_tr)
    X_tr = X_tr.drop(rows_pos_sample_to_remove_tr)

    y_test_pos = y_test[y_test == 1]
    rows_pos_sample_to_remove_test = random.sample(list(y_test_pos.index), int(len(y_test_pos) * 0.9))
    y_test = y_test.drop(rows_pos_sample_to_remove_test)
    X_test = X_test.drop(rows_pos_sample_to_remove_test)


    X = pd.concat([X_tr, X_test])
    X.index = [i for i in range(X.shape[0])]
    y = pd.concat([y_tr, y_test])
    y.index = [i for i in range(y.shape[0])]
    
    return X, y
    

def breast_cancer():
    breast_dataset = pd.read_csv("datas/breast_data.csv", sep=",", index_col=0)

    X = breast_dataset.iloc[:, 1:]
    y = breast_dataset["diagnosis"]

    def to_bits(n):
        if n == "M":
            return 1
        else:
            return 0
        
    y = y.apply(to_bits)
    y.name = "y"

    y_pos = y[y == 1]
    rows_pos_sample_to_remove = random.sample(list(y_pos.index), int(len(y_pos) * 0.8))
    y = y.drop(rows_pos_sample_to_remove)
    X = X.drop(rows_pos_sample_to_remove)
    
    return X, y


def california_13_perc():
    california_dataset = pd.read_csv("datas/california_housing_prices_data.csv").dropna()

    def house_sup_to(value, sup):
        if value >= sup:
            return 1
        else:
            return 0
        
    def ocean_prox(prox):
        if prox == "NEAR BAY":
            return 0
        elif prox == "NEAR OCEAN":
            return 1
        elif prox == "<1H OCEAN":
            return 2
        elif prox == "INLAND":
            return 3
        elif prox == "ISLAND":
            return 4
    
    california_dataset["ocean_proximity"] = california_dataset["ocean_proximity"].copy().apply(ocean_prox)
    california_dataset["median_house_value"] = california_dataset["median_house_value"].copy().apply(house_sup_to, args= [350000])

    X = california_dataset.iloc[:, [i for i in range(california_dataset.shape[1] - 2)] + [-1]]
    y = california_dataset["median_house_value"]
    y.name = "y"
    
    return X, y

def california_5_perc():
    california_dataset = pd.read_csv("datas/california_housing_prices_data.csv").dropna()

    def house_sup_to(value, sup):
        if value >= sup:
            return 1
        else:
            return 0
        
    def ocean_prox(prox):
        if prox == "NEAR BAY":
            return 0
        elif prox == "NEAR OCEAN":
            return 1
        elif prox == "<1H OCEAN":
            return 2
        elif prox == "INLAND":
            return 3
        elif prox == "ISLAND":
            return 4
    
    california_dataset["ocean_proximity"] = california_dataset["ocean_proximity"].copy().apply(ocean_prox)
    california_dataset["median_house_value"] = california_dataset["median_house_value"].copy().apply(house_sup_to, args= [500000])

    X = california_dataset.iloc[:, [i for i in range(california_dataset.shape[1] - 2)] + [-1]]
    y = california_dataset["median_house_value"]
    y.name = "y"
    
    return X, y


def credit_card_10_perc():
    creditcard_dataset = pd.concat([pd.read_csv("datas/creditcard_data1.csv").dropna(), pd.read_csv("datas/creditcard_data2.csv").dropna()], axis=0)
    X = creditcard_dataset.iloc[:, :-1]
    y = creditcard_dataset["Class"]
    y.name = "y"

    y_neg = y[y == 0]
    rows_neg_sample_to_remove = random.sample(list(y_neg.index), int(len(y_neg) * 0.985))
    y = y.drop(rows_neg_sample_to_remove)
    X = X.drop(rows_neg_sample_to_remove)
    
    return X, y

def credit_card_2_perc():
    creditcard_dataset = pd.concat([pd.read_csv("datas/creditcard_data1.csv").dropna(), pd.read_csv("datas/creditcard_data2.csv").dropna()], axis=0)
    X = creditcard_dataset.iloc[:, :-1]
    y = creditcard_dataset["Class"]
    y.name = "y"

    y_neg = y[y == 0]
    rows_neg_sample_to_remove = random.sample(list(y_neg.index), int(len(y_neg) * 0.9))
    y = y.drop(rows_neg_sample_to_remove)
    X = X.drop(rows_neg_sample_to_remove)
    
    return X, y


def white_wine():
    wine_white_dataset = pd.read_csv("datas/winequality_white_data.csv", sep=";")

    def quality_nb(quality):
        if quality == 8 or quality == 9:
            return 1
        else:
            return 0
        
    wine_white_dataset["quality"] = wine_white_dataset["quality"].copy().apply(quality_nb)
    X = wine_white_dataset.iloc[:, :-1]
    y = wine_white_dataset["quality"]
    y.name = "y"
    
    return X, y