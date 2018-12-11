import requests
from bs4 import BeautifulSoup
import pandas as pd
import math
from datetime import datetime
from datetime import date
from functools import reduce
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

def getData(file, cols): 
    "Takes in an HTML DOM and returns a pandas dataframes"
    with open(file) as f:
        soup = BeautifulSoup( f, 'html.parser' )
        text = str(soup.find_all('script')[5])

        first = 0
        results = []
        while first != -1: 
            beginning = text.find( "[new Date", first )
            ending = text.find( "]", beginning)
            dataText = text[beginning+1:ending] 
            row = dataText.split(",")
            results.append( row )
            first = ending
        results = results[:-1]
        for i,row in enumerate(results): 
            date = row[0]
            date = date[:-1]
            junkIdx = date.find( "(" )
            row[0] = date[junkIdx+1:]
            results[i] = row

        df = pd.DataFrame(results, columns=cols )

        df['Date'] = df['Date'].apply(lambda x: datetime.strptime(x, '"%Y/%m/%d"') )
        #covert to float 
        numerical_columns = cols[1:]
        for num_col in numerical_columns: 
            df[num_col] = df[num_col].apply(lambda x: float(0) if x == "null" else math.log( float(x) ) )
        return df
      

#getData( "transactionsData.html", ["Date", "Bitcoin No. of Transactions", "Ethereum No. of Transactions", "Ripple No. of Transactions"])

def MSE( predictions, actual ): 
    diffs = predictions - actual 
    u = 0 
    for elem in diffs: 
        u += ( elem ** 2 )
    return u / len(predictions)

#takes in the appropriate dataframes and returns a Linreg
def runBTCLinReg( dataset, y_df ): 
    #creating a dataframe with all Bitcoin data, got rid of Hashrate and # of transactions
    btc_columns = ["Date", "Bitcoin Avg. Transaction Value", "Bitcoin Avg. Transaction Fee", "Bitcoin No. of Transactions"]
    btc_df = dataset[ btc_columns ].fillna(value = 0)
    btc_df.set_index("Date", inplace = True) 
    y_btc_df = y_df[ ["Bitcoin Market Cap"] ] 

    trainingX_btc_df = btc_df.loc[:date(year=2017,month=12,day=31)]
    trainingY_btc_df = y_btc_df.loc[:date(year=2017,month=12,day=31)]
    #all btc 2017 data
    train = pd.concat([trainingX_btc_df, trainingY_btc_df], axis=1, join = "inner")
    res = LinearRegression().fit( train[train.columns.tolist()[:-1]], train['Bitcoin Market Cap'] )
    #linreg score
    score = res.score(train[train.columns.tolist()[:-1]], train['Bitcoin Market Cap'])
    print("The fit of this BTC data with a linreg model is %f" % score)
    predictX_btc_df = btc_df.loc[date(year=2018, month=1, day =1):]
    predictY_btc_df = y_btc_df.loc[date(year=2018,month=1,day=1):date(year=2018,month=11,day=27)]
    predictData = pd.concat([predictX_btc_df, predictY_btc_df], axis=1, join = "inner")

    predictions = res.predict( predictData[ predictData.columns.tolist()[:-1] ] )
    actual = predictY_btc_df["Bitcoin Market Cap"].values.tolist()

    y_true_mean = predictY_btc_df["Bitcoin Market Cap"].mean()
    print( "R-squared of BTC running linreg is %f" % r2_score( predictions, actual ) )
    print( "MSE of BTC running linreg is %f" % MSE( predictions, actual ) )
    print("\n")

#takes in the appropriate dataframes and returns a Linreg
def runETHLinReg( dataset, y_df ): 
    eth_columns = ["Date", "Ethereum Difficulty", "Ethereum Hashrate", "Ethereum Mining Prob", "Ethereum Avg. Transaction Value", "Ethereum No. of Transactions","Ethereum Avg. Transaction Fee"]
    eth_df = dataset[ eth_columns ].fillna(value = 0)
    eth_df.set_index("Date", inplace = True) 
    y_eth_df = y_df[ ["Ethereum Market Cap"] ]

    trainingX_eth_df = eth_df.loc[:date(year=2017,month=12,day=31)]
    trainingY_eth_df = y_eth_df.loc[:date(year=2017,month=12,day=31)]

    train = pd.concat([trainingX_eth_df, trainingY_eth_df], axis=1, join = "inner")
    res = LinearRegression().fit( train[train.columns.tolist()[:-1]], train['Ethereum Market Cap'] )
    score = res.score(train[train.columns.tolist()[:-1]], train['Ethereum Market Cap'])
    print("The fit of this ETH data with a linreg model is %f" % score)
    predictX_eth_df = eth_df.loc[date(year=2018, month=1, day =1):]
    predictY_eth_df = y_eth_df.loc[date(year=2018,month=1,day=1):date(year=2018,month=11,day=27)]
    predictData = pd.concat([predictX_eth_df, predictY_eth_df], axis=1, join = "inner")

    predictions = res.predict( predictData[ predictData.columns.tolist()[:-1] ] )

    actual = predictY_eth_df["Ethereum Market Cap"].values.tolist()
    print( "ETH range: %f" % ( max(actual) - min(actual) ) )

    y_true_mean = predictY_eth_df["Ethereum Market Cap"].mean()
    print( "R-squared of ETH running linreg is %f" % r2_score( predictions, actual ) )
    print( "MSE of ETH running linreg is %f" % MSE( predictions, actual ) )
    print( "\n" )

  


def fitXRPDataLinReg( dataset, y_df ): 
    #got rid of XRP_Difficulty and Hashrate
    xrp_columns = ["Date", "Ripple Mining Prob", "Ripple Avg. Transaction Value", "Ripple No. of Transactions", "Ripple Avg. Transaction Fee"]
    xrp_df = dataset[ xrp_columns ].fillna(value = 0)
    xrp_df.set_index("Date", inplace = True)
    y_xrp_df = y_df[ ["Ripple Market Cap"] ]
 
    trainingX_xrp_df = xrp_df.loc[:date(year=2017,month=12,day=31)]
    trainingY_xrp_df = y_xrp_df[:date(year=2017,month=12,day=31)]

    train = pd.concat([trainingX_xrp_df, trainingY_xrp_df], axis=1, join = "inner")
    res = LinearRegression().fit( train[train.columns.tolist()[:-1]], train['Ripple Market Cap'] )
    score = res.score(train[train.columns.tolist()[:-1]], train['Ripple Market Cap'])
    print("The fit of this XRP data with a linreg model is %f" % score)

    predictX_xrp_df = xrp_df.loc[date(year=2018, month=1, day =1):]
    predictY_xrp_df = y_xrp_df.loc[date(year=2018, month=1, day =1):date(year=2018,month=11,day=27)] 

    predictData = pd.concat([predictX_xrp_df, predictY_xrp_df], axis=1, join = "inner")

    predictions = res.predict( predictData[ predictData.columns.tolist()[:-1] ] )

    actual = predictY_xrp_df["Ripple Market Cap"].values.tolist()
    print( "XRP range: %f" % ( max(actual) - min(actual) ) )
    y_true_mean = predictY_xrp_df["Ripple Market Cap"].mean()
    print( "R-squared of XRP running linreg is %f" % r2_score( predictions, actual ) )
    print( "MSE of XRP running linreg is %f" % MSE( predictions, actual ) )
    print( "\n" )

def fitBTCDataDecisionTrees( dataset, y_df ): 
    btc_columns = ["Date", "Bitcoin Avg. Transaction Value", "Bitcoin Avg. Transaction Fee"]
    btc_df = dataset[ btc_columns ].fillna(value = 0)
    btc_df.set_index("Date", inplace = True) 
    y_btc_df = y_df[ ["Bitcoin Market Cap"] ] 


    trainingX_btc_df = btc_df.loc[:date(year=2017,month=12,day=31)]
    trainingY_btc_df = y_btc_df.loc[:date(year=2017,month=12,day=31)]
    #all btc 2017 data
    train = pd.concat([trainingX_btc_df, trainingY_btc_df], axis=1, join = "inner")

    regr1 = DecisionTreeRegressor().fit( train[train.columns.tolist()[:-1]], train['Bitcoin Market Cap'])
    score = regr1.score(train[train.columns.tolist()[:-1]], train['Bitcoin Market Cap'])
    print("The fit of this BTC data with a Decision Trees model is %f" % score)

    predictX_btc_df = btc_df.loc[date(year=2018, month=1, day =1):]
    predictY_btc_df = y_btc_df.loc[date(year=2018,month=1,day=1):date(year=2018,month=11,day=27)]
    predictData = pd.concat([predictX_btc_df, predictY_btc_df], axis=1, join = "inner")

    predictions = regr1.predict( predictData[ predictData.columns.tolist()[:-1] ] )

    actual = predictY_btc_df["Bitcoin Market Cap"].values.tolist()
    print("Bitcoin range %f" %  ( max(actual) - min(actual) ) )
    y_true_mean = predictY_btc_df["Bitcoin Market Cap"].mean()
    print( "R-squared of BTC running Decision Trees is %f" % r2_score( predictions, actual ) )
    print( "MSE of BTC running Decision Trees is %f" % MSE( predictions, actual ) )
    print( "\n" )

    

#takes in the appropriate dataframes and returns a Linreg
def fitETHDataDecisionTrees( dataset, y_df ): 
    eth_columns = ["Date", "Ethereum Difficulty", "Ethereum Mining Prob", "Ethereum Avg. Transaction Value", "Ethereum No. of Transactions","Ethereum Avg. Transaction Fee"]
    eth_df = dataset[ eth_columns ].fillna(value = 0)
    eth_df.set_index("Date", inplace = True) 
    y_eth_df = y_df[ ["Ethereum Market Cap"] ]

    trainingX_eth_df = eth_df.loc[:date(year=2017,month=12,day=31)]
    trainingY_eth_df = y_eth_df.loc[:date(year=2017,month=12,day=31)]

    train = pd.concat([trainingX_eth_df, trainingY_eth_df], axis=1, join = "inner")
    regr1 = DecisionTreeRegressor().fit( train[train.columns.tolist()[:-1]], train['Ethereum Market Cap'])
    score = regr1.score(train[train.columns.tolist()[:-1]], train['Ethereum Market Cap'])
    print("The fit of this ETH data with a Decision Trees model is %f" % score)

    predictX_eth_df = eth_df.loc[date(year=2018, month=1, day =1):]
    predictY_eth_df = y_eth_df.loc[date(year=2018,month=1,day=1):date(year=2018,month=11,day=27)]
    predictData = pd.concat([predictX_eth_df, predictY_eth_df], axis=1, join = "inner")

    predictions = regr1.predict( predictData[ predictData.columns.tolist()[:-1] ] )

    actual = predictY_eth_df["Ethereum Market Cap"].values.tolist()
    y_true_mean = predictY_eth_df["Ethereum Market Cap"].mean()
    print( "R-squared of ETH running Decision Trees is %f" % r2_score( predictions, actual ) )
    print( "MSE of ETH running Decision Trees is %f" % MSE( predictions, actual ) )
    print( "\n" )


def fitXRPDataDecisionTrees( dataset, y_df ): 
    xrp_columns = ["Date", "Ripple Mining Prob", "Ripple Avg. Transaction Value", "Ripple No. of Transactions", "Ripple Avg. Transaction Fee"]
    xrp_df = dataset[ xrp_columns ].fillna(value = 0)
    xrp_df.set_index("Date", inplace = True)
    y_xrp_df = y_df[ ["Ripple Market Cap"] ]
 
    trainingX_xrp_df = xrp_df.loc[:date(year=2017,month=12,day=31)]
    trainingY_xrp_df = y_xrp_df[:date(year=2017,month=12,day=31)]

    train = pd.concat([trainingX_xrp_df, trainingY_xrp_df], axis=1, join = "inner")
    res = DecisionTreeRegressor().fit( train[train.columns.tolist()[:-1]], train['Ripple Market Cap'] )
    score = res.score(train[train.columns.tolist()[:-1]], train['Ripple Market Cap'])
    print("The fit of this XRP data with a Decision Tree model is %f" % score)

    predictX_xrp_df = xrp_df.loc[date(year=2018, month=1, day =1):]
    predictY_xrp_df = y_xrp_df.loc[date(year=2018, month=1, day =1):date(year=2018,month=11,day=27)] 

    predictData = pd.concat([predictX_xrp_df, predictY_xrp_df], axis=1, join = "inner")

    predictions = res.predict( predictData[ predictData.columns.tolist()[:-1] ] )

    actual = predictY_xrp_df["Ripple Market Cap"].values.tolist()
    y_true_mean = predictY_xrp_df["Ripple Market Cap"].mean()
    print( "R-squared of XRP running Decision Tree is %f" % r2_score( predictions, actual ) )
    print( "MSE of XRP running Decision Trees is %f" % MSE( predictions, actual ) )
    print("\n")



def cleanData():
    x_Values = [ ("difficulty.html", ["Date", "Bitcoin Difficulty", "Ethereum Difficulty", "Ripple Difficulty"]), 
    ( "hashrate.html", ["Date", "Bitcoin Hashrate", "Ethereum Hashrate", "Ripple Hashrate"]), 
    ( "miningProb.html", ["Date", "Bitcoin Mining Prob", "Ethereum Mining Prob", "Ripple Mining Prob"]), 
    ( "averageTransactionValue.html", ["Date", "Bitcoin Avg. Transaction Value", "Ethereum Avg. Transaction Value", "Ripple Avg. Transaction Value"]), 
    ( "transactionsData.html", ["Date", "Bitcoin No. of Transactions", "Ethereum No. of Transactions", "Ripple No. of Transactions"]), 
    ( "averageTransactionFee.html", ["Date", "Bitcoin Avg. Transaction Fee", "Ethereum Avg. Transaction Fee", "Ripple Avg. Transaction Fee"])]
    y_Values = [ ("marketCapitalization.html", ["Date", "Bitcoin Market Cap", "Ethereum Market Cap", "Ripple Market Cap"] ) ]
    
    dfs = []
    for filename, columns in x_Values: 
        result = getData( filename, columns )
        dfs.append( result )

    for filename, columns in y_Values: 
    	y_df = getData( filename, columns )
    	y_df.set_index( "Date", inplace = True )

    #merging all features into one df 
    dataset = dfs[0]
    for i in range(1, len(dfs)): 
        dataset = dataset.merge(dfs[i], on='Date', how="outer")

    return dataset, y_df

def runData(): 
    dataset, y_df = cleanData() 

    #runBTCLinReg( dataset, y_df )
    runETHLinReg( dataset, y_df )
#     fitXRPDataLinReg( dataset, y_df )
#     fitBTCDataDecisionTrees( dataset, y_df )
#     fitETHDataDecisionTrees( dataset, y_df )
#     fitXRPDataDecisionTrees( dataset, y_df )
#     #creating a dataframe with all Ethereum data 
    

#     #creating a dataframe with all Ripple data

    


runData()
