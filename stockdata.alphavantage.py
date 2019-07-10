import pandas as pd
import quandl
import time
import requests
import json
import datetime

#Anonymous users have a limit of 20 calls per 10 minutes and 50 calls per day.
#Authenticated users have a limit of 300 calls per 10 seconds, 2,000 calls per 10 minutes and a limit of 50,000 
#quandl.ApiConfig.api_key = "es8HpUC1YMTgqixzAh6F"

#Jared TODO
#DONE....Data for all Stocks, for whole year appending to csv files in data
#Once we have all the data, run cointegration for all possible pairs

class Stock:
    def __init__(self,ticker="",start="",end="",prcs=[],data={}):
        self.__start=start
        self.__end=end
        self.__ticker=ticker
        self.__prcs=prcs
        self.__data=data
        self.__df=pd.DataFrame()
        
    def getPrcs(self): return self.__prcs
    def setPrcs(self, prcs): self.__prcs = prcs
    
    def getData(self): return self.__data

    def runQuery(self):
        request = requests.get('http://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol='+self.__ticker+'&outputsize=full&apikey=NG9C9EPVYBMQT0C8')
        #request = requests.get("https://api.polygon.io/v1/historic/trades/AAPL/2018-2-2?limit=100")
        print("request: ",request.text)
        data = json.loads(request.text)
        #print(data)
        data_array=[]
        for key, val in data['Time Series (Daily)'].items():
            data_array.append([pd.to_datetime(key),float(val['4. close'])])
        df = pd.DataFrame(data_array)
        df.columns=['Date','prc']
        df.index=df['Date']
        df=df.drop('Date',axis=1)
        df=df.sort_index(ascending=True)
        print(self.__ticker)
        #print(df.head())
        start=datetime.datetime(2013,1,1)
        end=datetime.datetime(2018,12,31)
        with open("data_5yr/"+self.__ticker+".csv","w") as file:
            df[start:end].to_csv(file)
        
def main():
    #s1 = Stock("FB","20170101","20171231")
    #s1.runQuery()
    #print(s1.getData())
    #print(s1.getPrcs())
    
    tickersdf = pd.read_csv('/home/ec2-user/environment/pairstrading/home/workspace/jared/analysis/tickers.sp500.1.txt')
    tickers = list(tickersdf['ticker'])
    i=0
    for tick in tickers:
        s = Stock(tick,"20130101","20181231")
        try:
            s.runQuery()
        except:
            print(tick," data failed")
        i+=1
        if i%5==0 and i != 0:
            time.sleep(60)
    
main()
    
    

        
    