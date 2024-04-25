import pandas as pd
import yfinance as yf
from datetime import date,time
import numpy as np
from sklearn.linear_model import LogisticRegression
mindate=date(2023,1,1)
maxdate=date(2024,3,1)
ril=pd.read_csv("ril.csv")
df=pd.DataFrame(ril)
#generates OHLCV data into csv file
def data():
    ril=yf.download("RELIANCE.NS",mindate,maxdate)
    df=pd.DataFrame(ril)
    df.to_csv("ril.csv",index=False)
#gets todays cuurent price
def get_curr_price():
    ticker=yf.Ticker("RELIANCE.NS")
    current_price = ticker.history(period='1d')['Close'][0]
    return current_price
#calucualtes sma for given period
def sma(ril,period):
    ril[f'SMA{period}']=0
    for i in range(len(ril)-1):
        #days before given period is assigned closing price as sma
        if i<period:
            ril[f'SMA{period}'][i]=ril["Close"][i]
        else:
            ril[f'SMA{period}'][i]=ril["Close"][i-5:i].sum()/5
    ril.to_csv("ril.csv",index=False)
#calculates ema for given period            
def ema(ril,period):
    ril[f'EMA{period}']=0
    for i in range(len(ril)-1):
        #days before given period is assigned closing price as ema
        if i<period:
            ril[f'EMA{period}'][i]=ril["Close"][i]
        else:
            ril[f'EMA{period}'][i]=(((ril["Close"][i]-ril["EMA"][i-1])*(2/(period+1)))+(ril["EMA"][i-1]))
    ril.to_csv("ril.csv",index=False)
#calculates macd line
def macdline(ril):
    for i in range(len(ril)-1):
        ril["MACDLine"][i]=ril["EMA12"]-ril["EMA26"]
    ril.to_csv("ril.csv",index=False)
#calculates macd signal
def macdsignal(ril):
    for i in range(len(ril)-1):
        if i>=12: 
            ril["MACDSignal"][i]=sum(ril["MACDLine"][i:i+8])/9
        else:
            ril["MACDSignal"][i]=0
    ril.to_csv("ril.csv",index=False)

#calculates macdhistogram
def macdhisto(ril):
    for i in range(len(ril)-1):
        ril["MACDHisto"][i]=ril["MACDLine"][i]-ril["MACDSignal"][i]
    ril.to_csv("ril.csv",index=False)
#calculates middle band fir bollingerbands
def middleband(ril):
    ril["MiddleBand"]=0
    for i in range(len(ril)-1):
        ril["MiddleBand"][i]=ril["EMA20"][i]
    ril.to_csv("ril.csv",index=False)
#calculates standarddeviation for given period
def standard_deviation(ril,period):
    ril[f"StandardDeviation{period}"]=0
    for i in range(len(ril)-1):
        variance=0
        if i<period:
            ril[f"StandarDeviation{period}"][i]=0
        else:
            for j in range(period):
                variance=variance+((ril["Close"][i-period+j]-ril["SMA20"][i-period+j])**2)
        ril[f"StandarDeviation{period}"][i]=(variance/period)**0.5
    ril.to_csv("ril.csv",index=False)
#calculates upperband for bollingerbands
def upperband(ril):
    ril["UpperBand"]=0
    for i in range(len(ril)-1):
        ril["UpperBand"][i]=ril["MiddleBand"][i]+2*ril["StandarDeviation20"][i]
    ril.to_csv("ril.csv",index=False)
#calculates lowerbandfor bollingerbands
def lowerband(ril):
    ril["LowerBand"]=0
    for i in range(len(ril)-1):
        ril["LowerBand"][i]=ril["MiddleBand"][i]-2*ril["StandarDeviation20"][i]
    ril.to_csv("ril.csv",index=False)


#1 for Buy
#0 for Hold
#-1 for Sell
#According to the rule of respective indicators call either 1,0 or -1 is given
def smacall(ril):
    for i in range(len(ril)-1):
        call=0
        diff=abs((ril["Close"][i]-ril["SMA"][i])*100/ril["SMA"][i])
        #if diff is approx 1% call=0
        if diff==1:
            call=0
        elif ril["SMA"][i]>ril["Close"][i]:
            call=1
        elif ril["SMA"][i]<ril["Close"][i]:
            call=-1
        ril["SMACall"][i]=call
    ril.to_csv("ril.csv",index=False)

def emacall(ril):
    for i in range(len(ril)-1):
        call=0
        #if diff is approx 1% call=0
        diff=abs((ril["Close"][i]-ril["EMA"][i])*100/ril["EMA"][i])
        if diff==1:
            call=0
        elif ril["EMA"][i]>ril["Close"][i]:
            call=1
        elif ril["EMA"][i]<ril["Close"][i]:
            call=-1
        ril["EMACall"][i]=call
    ril.to_csv("ril.csv",index=False)

def macdcall(ril):
    for i in range(len(ril)-1):
        call=0
        diff=abs((ril["MACDHisto"][i]/ril["MACDLine"][i])*100)
        if diff==1:
            call=0
        elif ril["MACDHisto"][i]>0:
            call=1
        elif ril["MACDHisto"][i]<0:
            call=-1
        ril["MACDCall"][i]=call
    ril.to_csv("ril.csv",index=False)


def bollingercall(ril):
    for i in range(len(ril)-1):
        call=0
        if ril["Close"][i]<ril["LowerBand"][i]:
            call=1
        elif ril["Close"][i]>ril["UpperBand"][i]:
            call=-1
        elif ril["Close"][i]<ril["LowerBand"][i] and ril["Close"][i]>ril["UpperBand"][i]:
            call=0
        ril["BollingerCall"][i]=call
    ril.to_csv("ril.csv",index=False)

#calculates relative strengthindex
def rsi(ril, period):
    ril["RSI"] = 0.0  # Initialize RSI column
    for i in range(0, len(ril)):
        if i >= period:
            upclose = 0
            upcount = 0
            downclose = 0
            downcount = 0
            for j in range(period):
                price_change = ril["Close"][i - j] - ril["Close"][i - 1 - j]
                if price_change > 0:
                    upclose += price_change
                    upcount += 1
                else:
                    downclose += abs(price_change)
                    downcount += 1
            if downcount != 0 and upcount!=0:
                upclose /= upcount
                downclose /= downcount
                try:
                    rsi = 100 - (100 / (1 + (upclose / downclose)))
                except ZeroDivisionError:
                    rsi = 0
                ril.loc[i, "RSI"] = rsi
        else:
            pass
    ril.to_csv("ril.csv", index=False)


def rsicall(ril):
    for i in range(len(ril)-1):
        if ril["RSI"][i]>70:
            call=-1
        elif ril["RSI"][i]<30:
            call=1
        else:
            call=0
        ril["RSICall"][i]=call
    ril.to_csv("ril.csv",index=False)
#calculates OBV
def calculate_obv(ril):
    ril.loc[0,"OBV"] = 0
    for i in range(1,len(ril)-1):
        if ril.loc[i,"Close"] > ril.loc[i-1,"Close"]:
            ril["OBV"][i]=ril["Volume"][i]+ril["OBV"][i-1]
        elif ril["Close"][i] > ril["Close"][i - 1]:
            ril["OBV"][i]=ril["OBV"][i-1]-ril["Volume"][i]
        else:
            ril.loc[i,"OBV"]=ril.loc[i-1,"OBV"]
    ril.to_csv("ril.csv")

def obvcall(ril):
    ril["OBVCall"]=0
    for i in range(1,len(ril)-1):
        if ril.loc[i,"OBV"]>ril.loc[i-1,"OBV"]:
            ril.loc[i,"OBVCall"]=1
        elif ril.loc[i,"OBV"]<ril.loc[i-1,"OBV"]:
            ril.loc[i,"OBVCall"]=-1
        else:
            ril.loc[i,"OBVCall"]=0
    ril.to_csv("ril.csv")
# calculates overall call
def call(ril):
    ril["Call"]=0
    for i in range(len(ril)):
        if abs(ril.loc[i,"EMA"]-ril.loc[i,"SMA"])<5:
            ril.loc[i,"Call"]=0
        elif ril.loc[i,"EMA"]>ril.loc[i,"SMA"]:
            ril.loc[i,"Call"]=1
        else:
            ril.loc[i,"Call"]=1

    ril.to_csv("ril.csv",index=False)


#mean normalisation for given indiactors

def fe(ril,feature):
    mean=np.mean(ril[feature])
    std=np.std(ril[feature])
    for i in range(len(ril)):
        ril.loc[i,f"{feature}FE"]=(ril.loc[i,feature]-mean)/std
    ril.to_csv("ril.csv")

#encodes and pushes indices for respective calls into buy,sell,hold

def encode(ril):
    buy=[]
    sell=[]
    hold=[]
    for i in range(len(ril)):
        if ril.loc[i,"Call"]==0:
            hold.append(i)
        elif ril.loc[i,"Call"]==1:
            buy.append(i)
        elif ril.loc[i,"Call"]==-1:
            sell.append(i)
    return buy,sell,hold

#calculates loss
def loss(ril,target,w,b):
    loss=0
    for i in target:
        z=0
        x=np.array((ril.loc[i,"EMAFE"],ril.loc[i,"SMAFE"],ril.loc[i,"MACDHistoFE"],ril.loc[i,"OBVFE"],ril.loc[i,"RSIFE"]))
        for j in range(5):
            z=w[j]*x[j]
        z+=b
        y=1/(1+np.exp(-z))
        loss+=(-1/len(target))*(y*np.log(y)+(1-y)*np.log(1-y))
    return loss
#backwardpropogation or gradient descent
def logisticregression(ril,target):
    w=[2,2,2,2,2]
    b=2
    for j in range(1000):
        for i in range(5):
            x=np.array((ril.loc[i,"EMAFE"],ril.loc[i,"SMAFE"],ril.loc[i,"MACDHistoFE"],ril.loc[i,"OBVFE"],ril.loc[i,"RSIFE"]))
            alpha=1
            w[i]-=alpha*loss(ril,target,w,b)*(x[i])
            b-=alpha*loss(ril,target,w,b)
    return w,b
#forward propogation
def predict(ril,i):
    buy,sell,hold=encode(ril)
    #wi,bi=logisticregrsession(ril,buy or sell or )
    w1,b1=[2.1254832821723877, 2.4087410615849083, 2.3425610888926562, 7.637232965705242, 9.2358522704187],-16.071312217200585
    w2,b2=[2.3276037691508655, 3.0812849295724765, 2.9203769801911603, 17.426327795292423, 22.225449578147924], -46.869652920170374
    w3,b3=[2.230128508616027, 2.7571395506642333, 2.6419759760787866, 12.703663208988258, 15.932484456611121], -32.003625727592095
    x=np.array((ril.loc[i,"EMAFE"],ril.loc[i,"SMAFE"],ril.loc[i,"MACDHistoFE"],ril.loc[i,"OBVFE"],ril.loc[i,"RSIFE"]))
    z1=np.dot(x,w1)+b1
    y1=1/(1+np.exp(-z1))
    z2=np.dot(x,w2)+b2
    y2=1/(1+np.exp(-z2))
    z3=np.dot(x,w3)+b3
    y3=1/(1+np.exp(-z3))
    call=[y1,y2,y3]
    if max(call)==call[0]:return 1
    elif max(call)==call[1]:return -1
    elif  max(call)==call[2]: return 0

#given mlcalls is stored in csv
def mlcall(ril):
    ril["MLCall"]=0
    for i in range(len(ril)):
        ril.loc[i,"MLCall"]=predict(ril,i)
    ril.to_csv("ril.csv")

#calculates and prints accuracy aand f1scores
def accuracy_and_f1_score(ril):
    correct = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for i in range(len(ril)):
        if ril.loc[i, "MLCall"] == ril.loc[i, "Call"]:
            correct += 1
            if ril.loc[i, "MLCall"] == 1:
                true_positives += 1
        else:
            if ril.loc[i, "MLCall"] == 1:
                false_positives += 1
            else:
                false_negatives += 1

    accuracy = correct / len(ril)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("Accuracy:", accuracy)
    print("F1 Score:", f1_score)  




