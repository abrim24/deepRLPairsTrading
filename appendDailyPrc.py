import bitmex
import datetime
import math
import pandas as pd
import time
import os
import json

'''
#Credentials with no order permissions
id="953y9OnxrISH4ejxqLWCkrk5"
secret="7ZnDirOvf6h77o_J-6ObW6rJ6ED_w6f-j_KS4LDP7U1_GhMQ"
'''

class AppendDailyPrc:
    def __init__(self,id="0UlX4dexMMMfIb7InUB22z0y",secret="elU5yWwALe_nPxn6Wf-rgNd0WE7ZBzWxJ_ZQFqLWXb-MPdRn"):
        
        self.id=id
        self.secret=secret
        self.coins = ["ADA","BCH","ETH","LTC","XRP"]#"EOS" out for now, check again September
        self.client = bitmex.bitmex(test=True, api_key=id, api_secret=secret)
        self.today=datetime.datetime.today()
        self.runDir = "/home/ubuntu/Cryptos/db"
        self.dataPath ="/home/ubuntu/Cryptos/db/data/"
        self.file = "combined.csv"
        self.cols =["Date","ADAUSD","BCHUSD","ETHUSD","LTCUSD","XBTUSD","XRPUSD"]
        
  
    def appendUSDPrcs(self):
        cl = self.client
        cns = self.coins
        dp = self.dataPath
        tod = datetime.datetime.today()
        t = datetime.datetime(tod.year,tod.month,tod.day)
        #t = datetime.datetime(2018,10,3)#rerun for specific date
        
        f = self.file
        cols = self.cols
        
        #saving data to specific dir
        os.chdir(self.runDir)
        
        #prcs = pd.read_csv(dp+f,parse_dates=True, index_col=0, usecols=cols)#no need to load file, just append to it below
        
        row={"Date":t.strftime('%Y-%m-%d')}
        xbtusd = cl.Trade.Trade_get(symbol="XBTUSD",startTime=t).result()[0][0]['price']
        
        print(xbtusd)
        row["XBTUSD"]=xbtusd
        for c in cns:#all coins minus XBT
            #sym = "."+c+"XBT30M"#changed 10/6/2018 to accomodate new Bitmex Index
            sym = ".B"+c+"XBT"
            try: p = cl.Trade.Trade_get(symbol=sym,startTime=t).result()[0][0]['price']
            except: p = -1
            #row[c+"XBT"] = p
            row[c+"USD"] = p * xbtusd
            
        newrow = pd.DataFrame([row])
        print(newrow)
        newrow = newrow.reindex(columns=cols)
        newrow.to_csv(dp+f,mode='a',index=False,header=False)
    
        
def main():
    bmc = AppendDailyPrc()
    bmc.appendUSDPrcs()
    
    
if __name__ == "__main__": main()