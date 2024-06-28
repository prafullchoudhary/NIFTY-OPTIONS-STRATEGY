import tkinter as tk
from tkinter import ttk
import threading
from multiprocessing import Manager
import concurrent.futures
import talib
import login as c
import pandas as pd
from datetime import datetime, timedelta
from pandas_market_calendars import get_calendar
import json
import os
from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws
import asyncio
from asyncio import Semaphore
from tkinter import messagebox
import math
import requests
from  time import sleep
from urllib.parse import parse_qs,urlparse
import base64
import pyotp
import pickle

def login():
    def getEncodedString(string):
        string = str(string)
        base64_bytes = base64.b64encode(string.encode("ascii"))
        return base64_bytes.decode("ascii")

    URL_SEND_LOGIN_OTP="https://api-t2.fyers.in/vagator/v2/send_login_otp_v2"
    res = requests.post(url=URL_SEND_LOGIN_OTP, json={"fy_id":getEncodedString(FY_ID),"app_id":"2"}).json()   
    # print(res)

    if datetime.now().second % 30 > 27 : sleep(5)
    URL_VERIFY_OTP="https://api-t2.fyers.in/vagator/v2/verify_otp"
    cotp=pyotp.TOTP(TOTP_KEY).now()
    res2 = requests.post(url=URL_VERIFY_OTP, json= {"request_key":res["request_key"],"otp":cotp}).json()  
    # print(cotp,res2)

    ses = requests.Session()
    URL_VERIFY_OTP2="https://api-t2.fyers.in/vagator/v2/verify_pin_v2"
    payload2 = {"request_key": res2["request_key"],"identity_type":"pin","identifier":getEncodedString(PIN)}
    res3 = ses.post(url=URL_VERIFY_OTP2, json= payload2).json()  

    ses.headers.update({
        'authorization': f"Bearer {res3['data']['access_token']}"
    })

    TOKENURL="https://api-t1.fyers.in/api/v3/token"
    payload3 = {"fyers_id":FY_ID,
            "app_id":client_id[:-4],
            "redirect_uri":redirect_uri,
            "appType":"100","code_challenge":"",
            "state":"None","scope":"","nonce":"","response_type":"code","create_cookie":True}

    res3 = ses.post(url=TOKENURL, json= payload3).json()  

    url = res3['Url']

    parsed = urlparse(url)
    auth_code = parse_qs(parsed.query)['auth_code'][0]

    grant_type = "authorization_code" 

    response_type = "code"  

    session = fyersModel.SessionModel(
        client_id=client_id,
        secret_key=secret_key, 
        redirect_uri=redirect_uri, 
        response_type=response_type, 
        grant_type=grant_type
    )

    # Set the authorization code in the session object
    session.set_token(auth_code)

    # Generate the access token using the authorization code
    response = session.generate_token()

    fyers = fyersModel.FyersModel(client_id=client_id, is_async=False, token=response['access_token'], log_path=os.getcwd())

    return fyers,client_id,response['access_token']

def rsi(close):
    return talib.RSI(close,timeperiod=10).round(2)

def macd(close):
    macd,macd_signal,macd_hist=talib.MACD(close) 
    return macd.round(2),macd_signal.round(2)

def adx(high,low,close):
    return talib.ADX(high=high,low=low,close=close).round(2)

def atr(high,low,close):
    return talib.ATR(high=high,low=low,close=close).round(2)

def ema(close,t):
    return talib.EMA(close,timeperiod=t).round(2)

def bbands(close):
    bbu,bbm,bbl=talib.BBANDS(close,timeperiod=20)
    return bbu.round(2),bbm.round(2), bbl.round(2)

def plusdi(high,low,close):
    return talib.PLUS_DI(high=high,low=low,close=close).round(2)

def minusdi(high,low,close):
    return talib.MINUS_DI(high=high,low=low,close=close).round(2)

def check_internet_connection(url='http://www.google.com/', timeout=5):
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except requests.ConnectionError:
        return False

def show_info(info):
    messagebox.showinfo("Scanner", info)

def strikePrice(obj):
    nclosedata = obj.history(data={
                                "symbol":"NSE:NIFTY50-INDEX",
                                "resolution":'3',
                                "date_format":0,
                                "range_from":int((datetime.now()-timedelta(minutes=15)).timestamp()),
                                "range_to":int((datetime.now()-timedelta(minutes=3)).replace(second=0).timestamp()),
                                "cont_flag":1
                                })
    nifty_ltp=nclosedata['candles'][-1][-2]
    return round(nifty_ltp/50)*50,nifty_ltp

async def VIX(obj,semaphore):
    async with semaphore:
        nclosedata = await obj.history(data={
                                    "symbol":"NSE:INDIAVIX-INDEX",
                                    "resolution":'3',
                                    "date_format":0,
                                    "range_from":int((datetime.now()-timedelta(minutes=15)).timestamp()),
                                    "range_to":int((datetime.now()-timedelta(minutes=3)).replace(second=0).timestamp()),
                                    "cont_flag":1
                                    })
        vix_ltp=nclosedata['candles'][-1][-2]
        return vix_ltp

def place_order(obj,symbol,qty):
    bundle=math.ceil(qty/1800)
    for i in range(bundle):
        if qty>1800:
            response = obj.place_order(data={
                "symbol":symbol,
                "qty":1800,
                "type":2,
                "side":1,
                "productType":"MARGIN",
                "limitPrice":0,
                "stopPrice":0,
                "validity":"DAY",
                "disclosedQty":0,
                "offlineOrder":False
            })
            qty-=1800
        else:
            response = obj.place_order(data={
                "symbol":symbol,
                "qty":qty,
                "type":2,
                "side":1,
                "productType":"MARGIN",
                "limitPrice":0,
                "stopPrice":0,
                "validity":"DAY",
                "disclosedQty":0,
                "offlineOrder":False
            })
    if response['s']!='ok':
        return response['message']
    elif response['s']=='ok':
        return response['id']

def funds_available(ltp,qty,obj):
    bundle=math.ceil(qty/1800)
    for i in obj.funds()['fund_limit']:
        if i['title']=='Available Balance':
            balance=i['equityAmount']
            break
    margin_required=(ltp*qty) + (bundle*20)

    if margin_required<=balance:
        return True
    else:
        return False
    
def is_position(obj,id):
    position=False
    p=0
    entry=0
    buyVal=0
    try:
        order=obj.orderbook(data={"id":id})['orderBook'][0]
        while p<50:
            p+=1
            if order['status']!=2:
                sleep(0.105)
                order=obj.orderbook(data={"id":id})['orderBook'][0]
            elif order['status']==2:
                entry=order['tradedPrice']
                buyVal=entry*order['qty']
                position=True
                break   
    except:
        print('Error in fetching order book')

    return position,entry,buyVal

def market_closed_open(calendar,today_date):
    open=datetime.strptime("09:15", "%H:%M").time()
    close=datetime.strptime("15:06:02", "%H:%M:%S").time()
    if (calendar.valid_days(start_date=today_date, end_date=today_date).size == 1) and (open < datetime.now().time() < close) :
        return True
    else:
        return False

async def dfmake(session,symbol,timeperiod,range,semaphore):
    async with semaphore:
        candles=await session.history(data={
            "symbol":symbol,
            "resolution":timeperiod,
            "date_format":1,
            "range_from":str((datetime.now()-timedelta(days=range)).date()),
            "range_to":str(datetime.now().date()),
            "cont_flag":1
        })
        candles=candles['candles']
        candles=pd.DataFrame(candles,columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        ctime=datetime.now()
        if ((ctime.minute % int(timeperiod)) ==0) and (candles['timestamp'].iloc[-1]==int(ctime.replace(second=0).timestamp())):
            return candles.iloc[:-1]
        else:
            return candles

def weekday(day):
    if day == 'Friday':
        return 1
    elif day == 'Monday':
        return 2
    elif day == 'Tuesday':
        return 3
    elif day == 'Wednesday':
        return 4
    elif day == 'Thursday':
        return 5

def time_to_range(time_str):
    start_time = pd.Timestamp('09:21:00')
    end_time = pd.Timestamp('15:09:00')
    interval = pd.Timedelta(minutes=3)

    time = pd.Timestamp(time_str)

    if time < start_time or time >= end_time:
        return "Time is out of range"

    range_num = 1 + int((time - start_time) / interval)
    return range_num

def pstrike_diff(diff,value,ce_pe):
    if ce_pe==0:
        for i in range(7000,30001,diff):
            if i>value:
                return round((i-value),2)
    else:
        for i in range(300000,6999,-diff):
            if i<value:
                return round((value-i),2)
        
def nstrike_diff(diff,value,ce_pe):
    if ce_pe==0:
        for i in range(30000,6999,-diff):
            if i<value:
                return round((value-i),2)
    else:
        for i in range(7000,30001,diff):
            if i>value:
                return round((i-value),2)
            
def profit_points(day):
    if day == 1 or day == 2 :
        return 30
    elif day == 3 or day == 4:
        return 35
    elif day == 5:
        return 40

def position_maker(model,sl_model,entry_time,ce_pe,strike_price,nifty_value,vix,macd15, macds15, macd3, macds3, close3, ema203, open3, pclose3, rsi3, adx3, pdi3, ppdi3, uband3, lband3, close15, open15, rsi15, rsiema15, ema2,ema5,mdi3,mband3,adx15,atr3,atr15):
    if (macd15 > macds15 and
            macd3 > macds3 and
            close3 > ema203 and
            close3 > open3 and
            close3 > pclose3 and
            pdi3 > ppdi3 and
            close15 > open15 and
            rsi15 > rsiema15 and
            ema2 > ema5 and
            mdi3 < pdi3 ):
        print('yes')
        target=model.predict([entry_time.date().day,entry_time.month,weekday(entry_time.strftime("%A")),time_to_range(str(entry_time.time())),
                              ce_pe,nifty_value,vix,strike_price,pstrike_diff(500,nifty_value,ce_pe),pstrike_diff(1000,nifty_value,ce_pe),
                              nstrike_diff(500,nifty_value,ce_pe),nstrike_diff(1000,nifty_value,ce_pe),
                              close3,adx3,macd15,macds15,macd15-macds15,macds3,macd3-macds3,ema203,close3-ema203,open3,pclose3,
                              close3-pclose3,adx15,pdi3,uband3,lband3,open15,close3-open15,rsiema15,rsi15-rsiema15,ema2,
                              atr3,atr15])
        sl=sl_model.predict([weekday(entry_time.strftime("%A")),time_to_range(str(entry_time.time())),
                             strike_price,nstrike_diff(200,nifty_value,ce_pe),nstrike_diff(500,nifty_value,ce_pe),
                             close3,rsi3,(uband3-lband3)/abs(lband3),macd15,macds15,macd15-macds15,macds3,macd3-macds3,ema203,close3-ema203,
                             open3,close3-open3,pclose3,close3-pclose3,adx15,pdi3,pdi3-ppdi3,uband3,lband3,mband3,open15,close3-open15,
                             rsiema15,rsi15-rsiema15,ema2,ema5,ema2-ema5,mdi3,atr3,atr15])
        print(ce_pe,target,close3,sl,entry_time)
        if ((target-close3)/(close3-sl)) > 2:
            target=target-14
            sl=sl-6.5
            if sl<0:
                sl=0
            if (target-close3)>profit_points(weekday(entry_time.strftime("%A"))):
                if ((close3-sl)/close3)<0.11:
                    return [target,sl,1]
                else:
                    return [target,sl,0.11/((close3-sl)/close3)]
            else:
                return None
        else:
            return None
    else:
        return None

stop_flag=False
def exit_button_clicked():
    global stop_flag
    global position
    if position==True:
        stop_flag=True

def closews(fyers):
    fyers.close_connection()

def autos_time(day):
    if day=='Monday':
        return '15:15:00'
    elif day=='Tuesday':
        return '15:21:00'
    elif day=='Wednesday':
        return '15:29:55'
    elif day=='Thursday':
        return '15:15:00'
    elif day=='Friday':
        return '15:15:00'

ltp=0
def sltp(entry,symbol, sl, tp, qty,obj,client_id,access_token,label_LTP_v,label_PNL_v,name):
    global stop_flag
    def onmessage(message):
        global ltp
        try:
            ltp=message['ltp']
            label_LTP_v.config(text=ltp)
            label_LTP_v.update_idletasks()
            label_PNL_v.config(text=round(((ltp-entry)*qty),1),foreground=("green" if (ltp-entry)>=0 else "red"))
            label_PNL_v.update_idletasks()
        except:
            print("Response:", message)

    def onerror(message):
        print("Error:", message)

    def onclose(message):
        global ltp
        ltp=0
        print("Connection closed:", message)

    def onopen():
        data_type = "SymbolUpdate"
        symbols = [symbol]
        fyers.subscribe(symbols=symbols, data_type=data_type)
        fyers.keep_running()
        
    access_token = f"{client_id}:{access_token}"

    fyers = data_ws.FyersDataSocket(
        access_token=access_token,       # Access token in the format "appid:accesstoken"
        log_path="",                     # Path to save logs. Leave empty to auto-create logs in the current directory.
        litemode=True,                  # Lite mode disabled. Set to True if you want a lite response.
        write_to_file=False,              # Save response in a log file instead of printing it.
        reconnect=True,                  # Enable auto-reconnection to WebSocket on disconnection.
        on_connect=onopen,               # Callback function to subscribe to data upon connection.
        on_close=onclose,                # Callback function to handle WebSocket connection close events.
        on_error=onerror,                # Callback function to handle WebSocket errors.
        on_message=onmessage             # Callback function to handle incoming messages from the WebSocket.
    )

    fyers.connect()
    
    ast=autos_time(datetime.now().strftime("%A"))
    if not os.path.isfile("Position.json"):
        data = {"date":str(datetime.now().date()),'position':symbol,"name": name,"QTY": qty,"sl":sl,"tp":tp,"entry":entry,"Buy_Amount":round(entry*qty) }
        with open("Position.json", "w") as pos:
            json.dump(data, pos)
    while True:
        if ltp!=0:
            break
    last_ltp = -1
    last_minute=(datetime.now() - timedelta(minutes=1)).time().minute
    while True:
        current_time = datetime.now().time()
        if stop_flag:
            obj.exit_positions(data={'id': f'{symbol}-MARGIN'})
            stop_flag=False
            concurrent.futures.ThreadPoolExecutor().submit(show_info,"Position Exited")
            data = {"Time":str(datetime.now().time()),"Exited_due_to": "EXIT Strategy",'Exit_price':ltp,'Exit_amount':round(ltp*qty),'PNL':round(((ltp-entry)*qty),1)}
            with open(f"Scanner_logs/{datetime.now().date()}.json", "a") as f:
                json.dump(data, f,indent=4)
            concurrent.futures.ThreadPoolExecutor().submit(closews,fyers)
            os.remove("Position.json")
            break
        elif ltp != last_ltp:
            last_ltp = ltp
            if (ltp < sl) or (ltp > tp) or (datetime.now().time()>=datetime.strptime(ast, "%H:%M:%S").time()):
                obj.exit_positions(data={'id': f'{symbol}-MARGIN'})
                concurrent.futures.ThreadPoolExecutor().submit(show_info,"Position Exited")
                data = {"Time":str(datetime.now().time()),"Exited_due_to": "SL_TP_Auto-Square-off",'Exit_price':ltp,'Exit_amount':round(ltp*qty),'PNL':round(((ltp-entry)*qty),1)}
                with open(f"Scanner_logs/{datetime.now().date()}.json", "a") as f:
                    json.dump(data, f,indent=4)
                concurrent.futures.ThreadPoolExecutor().submit(closews,fyers)
                os.remove("Position.json")
                break
            elif ((current_time.minute) % 3 == 0) and (last_minute!=current_time.minute):
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    last_minute=current_time.minute
                    df15min = pd.DataFrame(obj.history(data={
                        "symbol":symbol,
                        "resolution":'15',
                        "date_format":1,
                        "range_from":str((datetime.now()-timedelta(days=10)).date()),
                        "range_to":str(datetime.now().date()),
                        "cont_flag":1
                    })['candles'],columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    
                    if ((last_minute % 15) ==0) and (df15min['timestamp'].iloc[-1]==int(datetime.now().replace(second=0).timestamp())):
                        df15min=df15min.iloc[:-1]
                    elif (last_minute % 15) !=0:
                        # print(datetime.now().time())
                        df15min_pclose = obj.history(data={
                        "symbol":symbol,
                        "resolution":'3',
                        "date_format":1,
                        "range_from":str((datetime.now()-timedelta(days=1)).date()),
                        "range_to":str(datetime.now().date()),
                        "cont_flag":1
                        })['candles'][-2][-2]
                        df15min.loc[len(df15min)-1, 'close']=df15min_pclose
                        # print(df15min['close'].iloc[-1])

                    if stop_flag:
                        obj.exit_positions(data={'id': f'{symbol}-MARGIN'})
                        stop_flag=False
                        concurrent.futures.ThreadPoolExecutor().submit(show_info,"Position Exited")
                        data = {"Time":str(datetime.now().time()),"Exited_due_to": "EXIT Strategy",'Exit_price':ltp,'Exit_amount':round(ltp*qty),'PNL':round(((ltp-entry)*qty),1)}
                        with open(f"Scanner_logs/{datetime.now().date()}.json", "a") as f:
                            json.dump(data, f,indent=4)
                        concurrent.futures.ThreadPoolExecutor().submit(closews,fyers)
                        os.remove("Position.json")
                        executor.shutdown(wait=False)
                        break

                    df15min_rsi15_result = executor.submit(rsi, df15min['close'])
                    df15min_ema2_result= executor.submit(ema, df15min['close'], 2)
                    df15min_ema5_result= executor.submit(ema, df15min['close'], 5)

                    df15min_rsi15 = df15min_rsi15_result.result()
                    df15min_ema50rsi15_result = executor.submit(ema, df15min_rsi15, 50)
                    df15min_ema2=df15min_ema2_result.result().iloc[-1]
                    df15min_ema5=df15min_ema5_result.result().iloc[-1]
                    df15min_ema50rsi15 = df15min_ema50rsi15_result.result().iloc[-1]

                    if stop_flag:
                        obj.exit_positions(data={'id': f'{symbol}-MARGIN'})
                        stop_flag=False
                        concurrent.futures.ThreadPoolExecutor().submit(show_info,"Position Exited")
                        data = {"Time":str(datetime.now().time()),"Exited_due_to": "EXIT Strategy",'Exit_price':ltp,'Exit_amount':round(ltp*qty),'PNL':round(((ltp-entry)*qty),1)}
                        with open(f"Scanner_logs/{datetime.now().date()}.json", "a") as f:
                            json.dump(data, f,indent=4)
                        concurrent.futures.ThreadPoolExecutor().submit(closews,fyers)
                        os.remove("Position.json")
                        break

                    cond={'Time': str(datetime.now().time()),"15min_ema2":df15min_ema2,"15min_ema5":df15min_ema5,"15minclose":df15min['close'].iloc[-1],"15minopen":df15min['open'].iloc[-1],"15min_rsi":df15min_rsi15.iloc[-1],"15min_ema50rsi": df15min_ema50rsi15}
                    with open(f"Data/SLTP_{datetime.now().date()}.json", "a") as c:
                        json.dump(cond, c,indent=4)

                    if stop_flag:
                        obj.exit_positions(data={'id': f'{symbol}-MARGIN'})
                        stop_flag=False
                        concurrent.futures.ThreadPoolExecutor().submit(show_info,"Position Exited")
                        data = {"Time":str(datetime.now().time()),"Exited_due_to": "EXIT Strategy",'Exit_price':ltp,'Exit_amount':round(ltp*qty),'PNL':round(((ltp-entry)*qty),1)}
                        with open(f"Scanner_logs/{datetime.now().date()}.json", "a") as f:
                            json.dump(data, f,indent=4)
                        concurrent.futures.ThreadPoolExecutor().submit(closews,fyers)
                        os.remove("Position.json")
                        break
                    elif (df15min_ema2 < df15min_ema5) and (df15min['close'].iloc[-1] < df15min['open'].iloc[-1]):
                        obj.exit_positions(data={'id': f'{symbol}-MARGIN'})
                        concurrent.futures.ThreadPoolExecutor().submit(show_info,"Position Exited")
                        data = {"Time":str(datetime.now().time()),"Exited_due_to": "EMA Condition",'Exit_price':ltp,'Exit_amount':round(ltp*qty),'PNL':round(((ltp-entry)*qty),1)}
                        with open(f"Scanner_logs/{datetime.now().date()}.json", "a") as f:
                            json.dump(data, f,indent=4)
                        concurrent.futures.ThreadPoolExecutor().submit(closews,fyers)
                        os.remove("Position.json")
                        break
                    elif (df15min_rsi15.iloc[-1] < df15min_ema50rsi15) and (df15min['close'].iloc[-1] < df15min['open'].iloc[-1]):
                        obj.exit_positions(data={'id': f'{symbol}-MARGIN'})
                        concurrent.futures.ThreadPoolExecutor().submit(show_info,"Position Exited")
                        data = {"Time":str(datetime.now().time()),"Exited_due_to": "RSI Condition",'Exit_price':ltp,'Exit_amount':round(ltp*qty),'PNL':round(((ltp-entry)*qty),1)}
                        with open(f"Scanner_logs/{datetime.now().date()}.json", "a") as f:
                            json.dump(data, f,indent=4)
                        concurrent.futures.ThreadPoolExecutor().submit(closews,fyers)
                        os.remove("Position.json")
                        break
        sleep(0.001)
    label_LTP_v.config(text="-")
    label_LTP_v.update_idletasks()
    label_PNL_v.config(text="-",foreground='black')
    label_PNL_v.update_idletasks()


position=False
def pro(shared_obj, label_scanner, entry_fund,label_LTP_v,label_PNL_v):
    try:
        shared_obj.scanner_text = f"Loading..."
        label_scanner.config(text=shared_obj.scanner_text)
        label_scanner.update_idletasks()
        global position
        is_login=False
        if os.path.isfile("Position.json"):
            with open("Position.json", 'r') as pf:
                symbol = json.load(pf)
            if datetime.strptime(symbol['date'], "%Y-%m-%d").date()==datetime.now().date():
                obj,client_id,access_token = login()
                asyn_fyers= fyersModel.FyersModel(client_id=client_id, is_async=True, token=access_token)
                is_login=True
                shared_obj.scanner_text = f"Position activated in {symbol['name']} QTY {symbol['QTY']} SL {symbol['sl']} TGT {symbol['tp']} BUY PRICE {symbol['entry']} BUY AMOUNT {symbol['Buy_Amount']}"
                label_scanner.config(text=shared_obj.scanner_text)
                label_scanner.update_idletasks()
                position=True
                sltp(symbol['entry'],symbol['position'], symbol['sl'], symbol['tp'], symbol['QTY'],obj,client_id,access_token,label_LTP_v,label_PNL_v,symbol['name'])
                position=False
            else:
                os.remove("Position.json")


        if not os.path.exists("Scanner_logs"):
            os.makedirs("Scanner_logs")
        if not os.path.exists("Data"):
            os.makedirs("Data")
        
        calendar = get_calendar("NSE")
        today_date=pd.Timestamp.now().date()
        stretegy_start_time=datetime.strptime("09:20", "%H:%M").time()

        url = 'https://public.fyers.in/sym_details/NSE_FO.csv'
        df = pd.read_csv(url,header=None)
        df.columns =  ['FyersToken', 'Name', 'Instrument', 'lot', 'tick','ISIN','TradingSession', 'Lastupdatedate', 'Expirydate', 'symbol', 'Exchange', 'Segment','ScripCode','ScripName','14','strike','CE_PE','17','18','19','20']
        df =df[(df['Instrument']==14) & (df['ScripName'] == 'NIFTY')]
        df['Expirydate'] = pd.to_datetime(df['Expirydate'],unit='s').apply(lambda x: x.date())
        df=df[df['Expirydate']>=datetime.today().date()].sort_values(by=['Expirydate'])
        df['strike'] = df['strike'].astype('Int64')

        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('sl_model.pkl', 'rb') as f:
            sl_model = pickle.load(f)

        last_minute=(datetime.now() - timedelta(minutes=1)).time().minute
        while True:
            if market_closed_open(calendar,today_date)==True:
            # if True:
                shared_obj.scanner_text = "Scanner Is Running"
                label_scanner.config(text=shared_obj.scanner_text)
                label_scanner.update_idletasks()

                if is_login==False:
                    obj,client_id,access_token = login()
                    asyn_fyers= fyersModel.FyersModel(client_id=client_id, is_async=True, token=access_token)
                    is_login=True
                current_time=datetime.now()
                if ((current_time.minute)%3==0) and (current_time.time()>stretegy_start_time) and (last_minute!=current_time.minute):
                # if True:
                    last_minute=current_time.minute
                    strike_price,nifty_value=strikePrice(obj)
                    strikes = df[df['strike'].isin([strike_price])]
                    strikes=strikes.iloc[:2]
                    CE=strikes[strikes['CE_PE']=='CE']
                    PE=strikes[strikes['CE_PE']=='PE']
                    
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        async def main():
                            tasks = []
                            tasks.append(asyncio.ensure_future(dfmake(asyn_fyers,CE['symbol'].iloc[0],"15",10,Semaphore(5))))
                            tasks.append(asyncio.ensure_future(dfmake(asyn_fyers,CE['symbol'].iloc[0],"3",5,Semaphore(5))))
                            tasks.append(asyncio.ensure_future(dfmake(asyn_fyers,PE['symbol'].iloc[0],"15",10,Semaphore(5))))
                            tasks.append(asyncio.ensure_future(dfmake(asyn_fyers,PE['symbol'].iloc[0],"3",5,Semaphore(5))))
                            tasks.append(asyncio.ensure_future(VIX(asyn_fyers,Semaphore(5))))
                            await asyncio.gather(*tasks)
                            
                            CE15=tasks[0].result()
                            CE3=tasks[1].result()
                            PE15=tasks[2].result()
                            PE3=tasks[3].result()
                            vixr=tasks[4].result()
                            return CE15,CE3,PE15,PE3,vixr
                        
                        CE_15min,CE_3min,PE_15min,PE_3min,vix=loop.run_until_complete(main())

                        if (last_minute % 15) !=0:
                            CE_15min.loc[len(CE_15min)-1, 'close']=CE_3min['close'].iloc[-1]
                            PE_15min.loc[len(PE_15min)-1, 'close']=PE_3min['close'].iloc[-1]

                        CE_rsi15_result = executor.submit(rsi, CE_15min['close'])
                        CE_rsi_result = executor.submit(rsi, CE_3min['close']) 
                        CE_macd_result = executor.submit(macd, CE_3min['close'])
                        CE_macd15_result = executor.submit(macd, CE_15min['close'])
                        CE_adx_result = executor.submit(adx, CE_3min['high'], CE_3min['low'], CE_3min['close'])
                        CE_adx15_result = executor.submit(adx, CE_15min['high'], CE_15min['low'], CE_15min['close'])
                        CE_ema_result = executor.submit(ema,CE_3min['close'],20)
                        CE_bbands_result = executor.submit(bbands,CE_3min['close'])
                        CE_minusdi_result = executor.submit(minusdi,CE_3min['high'],CE_3min['low'],CE_3min['close'])
                        CE_plusdi_result = executor.submit(plusdi,CE_3min['high'],CE_3min['low'],CE_3min['close'])
                        CE_ema2_result = executor.submit(ema,CE_15min['close'],2)
                        CE_ema5_result = executor.submit(ema,CE_15min['close'],5)
                        CE_atr3_result = executor.submit(atr, CE_3min['high'], CE_3min['low'], CE_3min['close'])
                        CE_atr15_result = executor.submit(atr, CE_15min['high'], CE_15min['low'], CE_15min['close'])
                        PE_rsi15_result = executor.submit(rsi, PE_15min['close'])
                        PE_rsi_result = executor.submit(rsi, PE_3min['close']) 
                        PE_macd_result = executor.submit(macd, PE_3min['close'])
                        PE_macd15_result = executor.submit(macd, PE_15min['close'])
                        PE_adx_result = executor.submit(adx, PE_3min['high'], PE_3min['low'], PE_3min['close'])
                        PE_adx15_result = executor.submit(adx, PE_15min['high'], PE_15min['low'], PE_15min['close'])
                        PE_ema_result = executor.submit(ema,PE_3min['close'],20)
                        PE_bbands_result = executor.submit(bbands,PE_3min['close'])
                        PE_plusdi_result = executor.submit(plusdi,PE_3min['high'],PE_3min['low'],PE_3min['close'])
                        PE_minusdi_result = executor.submit(minusdi,PE_3min['high'],PE_3min['low'],PE_3min['close'])
                        PE_ema2_result = executor.submit(ema,PE_15min['close'],2)
                        PE_ema5_result = executor.submit(ema,PE_15min['close'],5)
                        PE_atr3_result = executor.submit(atr, PE_3min['high'], PE_3min['low'], PE_3min['close'])
                        PE_atr15_result = executor.submit(atr, PE_15min['high'], PE_15min['low'], PE_15min['close'])
                        
                        CE_rsi15 = CE_rsi15_result.result()
                        CE_ema50rsi15_result = executor.submit(ema,CE_rsi15,50)
                        PE_rsi15 = PE_rsi15_result.result()
                        PE_ema50rsi15_result = executor.submit(ema,PE_rsi15,50)
                        CE_macd15,CE_macd_signal15 = CE_macd15_result.result()
                        CE_macd3,CE_macd_signal3  = CE_macd_result.result()
                        CE_ema203= CE_ema_result.result().iloc[-1]
                        CE_rsi3 = CE_rsi_result.result().iloc[-1]
                        CE_adx3 = CE_adx_result.result().iloc[-1]
                        CE_bbands_u3,CE_bbands_m3,CE_bbands_l3 = CE_bbands_result.result()
                        CE_plusdi3 = CE_plusdi_result.result()
                        CE_ema2 = CE_ema2_result.result().iloc[-1]
                        CE_ema5 = CE_ema5_result.result().iloc[-1]
                        CE_minusdi=CE_minusdi_result.result().iloc[-1]
                        CE_adx15 = CE_adx15_result.result().iloc[-1]
                        CE_atr3 = CE_atr3_result.result().iloc[-1]
                        CE_atr15 = CE_atr15_result.result().iloc[-1]
                        CE_ema50rsi15 = CE_ema50rsi15_result.result().iloc[-1]
                        PE_macd15,PE_macd_signal15 = PE_macd15_result.result()
                        PE_macd3,PE_macd_signal3  = PE_macd_result.result()
                        PE_ema203= PE_ema_result.result().iloc[-1]
                        PE_rsi3 = PE_rsi_result.result().iloc[-1]
                        PE_adx3 = PE_adx_result.result().iloc[-1]
                        PE_bbands_u3,PE_bbands_m3,PE_bbands_l3 = PE_bbands_result.result()
                        PE_plusdi3 = PE_plusdi_result.result()
                        PE_ema2 = PE_ema2_result.result().iloc[-1]
                        PE_ema5 = PE_ema5_result.result().iloc[-1]
                        PE_minusdi=PE_minusdi_result.result().iloc[-1]
                        PE_adx15 = PE_adx15_result.result().iloc[-1]
                        PE_atr3 = PE_atr3_result.result().iloc[-1]
                        PE_atr15 = PE_atr15_result.result().iloc[-1]
                        PE_ema50rsi15 = PE_ema50rsi15_result.result().iloc[-1]
                        CE_pos=executor.submit(position_maker,model,sl_model,datetime.now(),0,strike_price,nifty_value,vix,CE_macd15.iloc[-1],CE_macd_signal15.iloc[-1],CE_macd3.iloc[-1],CE_macd_signal3.iloc[-1],CE_3min['close'].iloc[-1],CE_ema203,CE_3min['open'].iloc[-1],CE_3min['close'].iloc[-2],CE_rsi3,CE_adx3,CE_plusdi3.iloc[-1],CE_plusdi3.iloc[-2],CE_bbands_u3.iloc[-1],CE_bbands_l3.iloc[-1],CE_15min['close'].iloc[-1],CE_15min['open'].iloc[-1],CE_rsi15.iloc[-1],CE_ema50rsi15,CE_ema2,CE_ema5,CE_minusdi,CE_bbands_m3.iloc[-1],CE_adx15,CE_atr3,CE_atr15)
                        PE_pos=executor.submit(position_maker,model,sl_model,datetime.now(),1,strike_price,nifty_value,vix,PE_macd15.iloc[-1],PE_macd_signal15.iloc[-1],PE_macd3.iloc[-1],PE_macd_signal3.iloc[-1],PE_3min['close'].iloc[-1],PE_ema203,PE_3min['open'].iloc[-1],PE_3min['close'].iloc[-2],PE_rsi3,PE_adx3,PE_plusdi3.iloc[-1],PE_plusdi3.iloc[-2],PE_bbands_u3.iloc[-1],PE_bbands_l3.iloc[-1],PE_15min['close'].iloc[-1],PE_15min['open'].iloc[-1],PE_rsi15.iloc[-1],PE_ema50rsi15,PE_ema2,PE_ema5,PE_minusdi,PE_bbands_m3.iloc[-1],PE_adx15,PE_atr3,PE_atr15)
                        ct=datetime.now().time()
                        CE_pos_result=CE_pos.result()
                        PE_pos_result=PE_pos.result()
                        if PE_pos_result!=None:
                            # ltpc=obj.quotes(data={"symbols":PE['symbol'].iloc[0]})['d'][0]['v']['lp']
                            ltpc=PE_3min['close'].iloc[-1]
                            qty=int(((int(entry_fund.get())*PE_pos_result[2])/ltpc)/25)*25
                            if funds_available(ltpc,qty,obj):
                                oid=place_order(obj,PE['symbol'].iloc[0],qty)
                            elif (qty-25)>0:
                                oid=place_order(obj,PE['symbol'].iloc[0],qty-25)
                            pt=datetime.now().time()
                            position,ltpc,buyVal=is_position(obj,oid)
                            if position:
                                executor.submit(show_info,"Entry Activated")
                                shared_obj.scanner_text = f"Position activated in {PE['Name'].iloc[0]} QTY {qty} SL {round(PE_pos_result[1],2)} TGT {round(PE_pos_result[0],2)} BUY PRICE {ltpc} BUY AMOUNT {buyVal}"
                                label_scanner.config(text=shared_obj.scanner_text)
                                label_scanner.update_idletasks()
                                data = {"Time":str(pt),"Position_activated_in": PE['Name'].iloc[0],"QTY": qty,"Buy_Price":ltpc,"SL":PE_pos_result[1],"TP":PE_pos_result[0],"Buy_Amount":buyVal}
                                with open(f"Scanner_logs/{datetime.now().date()}.json", "a") as f:
                                    json.dump(data, f,indent=4)
                                sltp(ltpc,PE['symbol'].iloc[0],PE_pos_result[1],PE_pos_result[0],qty,obj,client_id,access_token,label_LTP_v,label_PNL_v,PE['Name'].iloc[0])
                                position=False
                            else:
                                executor.submit(show_info,"Something Went Wrong\nPosition Not Activated\n",oid)
                        elif CE_pos_result!=None:
                            # ltpc=obj.quotes(data={"symbols":CE['symbol'].iloc[0]})['d'][0]['v']['lp']
                            ltpc=CE_3min['close'].iloc[-1]
                            qty=int(((int(entry_fund.get())*CE_pos_result[2])/ltpc)/25)*25
                            if funds_available(ltpc,qty,obj):
                                oid=place_order(obj,CE['symbol'].iloc[0],qty)
                            elif (qty-25)>0:
                                oid=place_order(obj,CE['symbol'].iloc[0],qty-25)
                            pt=datetime.now().time()
                            position,ltpc,buyVal=is_position(obj,oid)
                            if position:
                                executor.submit(show_info,"Entry Activated")
                                shared_obj.scanner_text = f"Position activated in {CE['Name'].iloc[0]} QTY {qty} SL {round(CE_pos_result[1],2)} TGT {round(CE_pos_result[0],2)} BUY PRICE {ltpc} BUY AMOUNT {buyVal}"
                                label_scanner.config(text=shared_obj.scanner_text)
                                label_scanner.update_idletasks()
                                data = {"Time":str(pt),"Position_activated_in": CE['Name'].iloc[0],"QTY": qty,"Buy_Price":ltpc,"SL":CE_pos_result[1],"TP":CE_pos_result[0],"Buy_Amount":buyVal}
                                with open(f"Scanner_logs/{datetime.now().date()}.json", "a") as f:
                                    json.dump(data, f,indent=4)
                                sltp(ltpc,CE['symbol'].iloc[0],CE_pos_result[1],CE_pos_result[0],qty,obj,client_id,access_token,label_LTP_v,label_PNL_v,CE['Name'].iloc[0])
                                position=False
                            else:
                                executor.submit(show_info,"Something Went Wrong\nPosition Not Activated\n",oid)
                        conditionsi = {
                            'Time': str(ct),
                            'Vix':vix,
                            'CE_Name': CE['Name'].iloc[0],
                            'CE_macd15': CE_macd15.iloc[-1],
                            'CE_macd_signal15': CE_macd_signal15.iloc[-1],
                            'CE_macd3': CE_macd3.iloc[-1],
                            'CE_mac_signal3': CE_macd_signal3.iloc[-1],
                            'CE_3min_close_prev': CE_3min['close'].iloc[-1],
                            'CE_ema203': CE_ema203,
                            'CE_3min_open_prev': CE_3min['open'].iloc[-1],
                            'CE_3min_close_prev2': CE_3min['close'].iloc[-2],
                            'CE_rsi3': CE_rsi3,
                            'CE_adx3': CE_adx3,
                            'CE_plusdi3_prev': CE_plusdi3.iloc[-1],
                            'CE_plusdi3_prev2': CE_plusdi3.iloc[-2],
                            'CE_bbands_u3_prev': CE_bbands_u3.iloc[-1],
                            'CE_bbands_l3_prev': CE_bbands_l3.iloc[-1],
                            'CE_15min_close_prev': CE_15min['close'].iloc[-1],
                            'CE_15min_open_prev': CE_15min['open'].iloc[-1],
                            'CE_rsi15': CE_rsi15.iloc[-1],
                            'CE_ema50rsi15': CE_ema50rsi15,
                            'CE_ema2': CE_ema2,
                            'CE_ema5': CE_ema5,
                            'CE_minusdi': CE_minusdi,
                            'CE_adx15':CE_adx15,
                            'CE_atr3':CE_atr3,
                            'CE_atr15':CE_atr15,
                            'PE_Name': PE['Name'].iloc[0],
                            'PE_macd15': PE_macd15.iloc[-1],
                            'PE_macd_signal15': PE_macd_signal15.iloc[-1],
                            'PE_macd3': PE_macd3.iloc[-1],
                            'PE_macd_signal3': PE_macd_signal3.iloc[-1],
                            'PE_3min_close_prev': PE_3min['close'].iloc[-1],
                            'PE_ema203': PE_ema203,
                            'PE_3min_open_prev': PE_3min['open'].iloc[-1],
                            'PE_3min_close_prev2': PE_3min['close'].iloc[-2],
                            'PE_rsi3': PE_rsi3,
                            'PE_adx3': PE_adx3,
                            'PE_plusdi3_prev': PE_plusdi3.iloc[-1],
                            'PE_plusdi3_prev2': PE_plusdi3.iloc[-2],
                            'PE_bbands_u3_prev': PE_bbands_u3.iloc[-1],
                            'PE_bbands_l3_prev': PE_bbands_l3.iloc[-1],
                            'PE_15min_close_prev': PE_15min['close'].iloc[-1],
                            'PE_15min_open_prev': PE_15min['open'].iloc[-1],
                            'PE_rsi15': PE_rsi15.iloc[-1],
                            'PE_ema50rsi15': PE_ema50rsi15,
                            'PE_ema2': PE_ema2,
                            'PE_ema5': PE_ema5,
                            'PE_minusdi': PE_minusdi,
                            'PE_adx15':PE_adx15,
                            'PE_atr3':PE_atr3,
                            'PE_atr15':PE_atr15
                        }
                        with open(f"Data/conditions_{datetime.now().date()}.json", 'a') as d:
                            json.dump(conditionsi, d, indent=4)
                    executor.shutdown()
                else:
                    sleep(0.001)
            else:
                is_login=False
                shared_obj.scanner_text = "Scanner Is Closed At This Time!!"
                label_scanner.config(text=shared_obj.scanner_text)
                label_scanner.update_idletasks()
                sleep(0.1)
    except Exception as e:
        print(f"An error occurred: {e}")
        while check_internet_connection()==False:
            print('Not connected to internet')
            sleep(1)
        pro(shared_obj, label_scanner, entry_fund,label_LTP_v,label_PNL_v)


if __name__ == "__main__":

    loop = asyncio.get_event_loop()

    redirect_uri = c.redirect_uri
    client_id=c.client_id
    secret_key = c.secret_key
    FY_ID = c.FY_ID  # Your fyers ID
    TOTP_KEY = c.TOTP_KEY  # TOTP secret is generated when we enable 2Factor TOTP from myaccount portal
    PIN = c.PIN

    root = tk.Tk()
    root.title("Scanner")
    root.geometry("410x250")

    entry_fund = ttk.Entry(root, width=10, font=('Helvetica', 12, 'normal'), validate="key", validatecommand=(root.register(lambda x: x.isdigit()), '%S'))
    entry_fund.insert(0, 0)
    entry_fund.place(x=210,y=17)

    label_fund = ttk.Label(root, text="Funds To Trade",font= ('Goudy old style', 15))
    label_fund.place(x=15,y=15)

    label_LTP = ttk.Label(root, text="LTP : ",font= ('Goudy old style', 13))
    label_LTP.place(x=25,y=70)

    label_PNL = ttk.Label(root, text="PNL : ",font= ('Goudy old style', 13))
    label_PNL.place(x=25,y=120)

    label_LTP_v = ttk.Label(root, text="-",font= ('Goudy old style', 13))
    label_LTP_v.place(x=70,y=70)

    label_PNL_v = ttk.Label(root, text="-",font= ('Goudy old style', 13))
    label_PNL_v.place(x=70,y=120)

    label_frame = ttk.LabelFrame(root,height=180, width=225)
    shared_obj = Manager().Namespace()
    label_scanner = ttk.Label(label_frame, text="",font= ('Goudy old style', 12),wraplength=170)
    label_scanner.place(relx=0.5, rely=0.45,anchor='center')
    label_frame.place(x=160, y=50)

    style = ttk.Style()
    style.configure('TButton', font=('Goudy old style', 14,'bold'), padding=('10', '2'))
    button_exit = ttk.Button(root, width=7, text="EXIT", command=exit_button_clicked, style='TButton')
    button_exit.place(x=25,y=170)

    t = threading.Thread(target=pro, args=(shared_obj, label_scanner, entry_fund,label_LTP_v,label_PNL_v))
    t.start()

    root.mainloop()

    os._exit(0)