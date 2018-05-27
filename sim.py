import random
import numpy as np
from queue import Queue
import time
import matplotlib.pyplot as plt
import math

timestamp = 0
timestamp_MAX = 600
car_list = []
maxask = 0
minbid = 200
buyers_now = 0
sellers_now = 0
leftover = []
lastamount = 1
lastdeal = 100
gridprice = 0

# Customized
buyer_TOTAL = 400
seller_TOTAL = 400

ask = np.zeros((200,), dtype=int)
bid = np.zeros((200,), dtype=int)

trade_history = []

class Car:
    def __init__(self,st,ed,vo):
        self.st = st
        self.ed = ed
        self.vo = vo
        self.volume = vo
        self.next = 0
        self.freezed = False
        self.money = 0
        self.last = np.zeros((200,), dtype=int)
        if isinstance(self,Buyer): 
            self.eco = random.betavariate(1.1,1.1)
            #self.eco = random.random() * 0.1 + 0.45
            #self.eco = 0.5
        elif isinstance(self, Seller): self.eco = random.betavariate(0.8,2)
        else: assert 0
        self.out = False

    def abfromuv(self,u,v):
        """
        v = alpha + beta, called 'sample size', referring to en wiki page
        """
        alpha = u * v
        beta = (1 - u) * v
        return alpha, beta

    def pricehit(self, price):
        if self.freezed: return False
        return self.last[price] > 0

    def freeze(self,price):
        #assert 0, "freezed"
        assert self.freezed == False
        assert self.last[price] > 0
        self.last = np.zeros((200,), dtype=int)
        self.money += price + 300
        self.freezed = True
        self.vo -= 1
        self.next += 1

    def unfreeze(self):
        if self.ed < timestamp and not self.out:
            self.out = True
            if isinstance(self, Buyer): leftover.append(self.vo)
        if self.next < timestamp:
            self.next = 0
            self.freezed = False

    def policy(self):
        pass

    def submit(self):
        self.last = np.zeros((200,), dtype=int)
        if self.freezed or self.st > timestamp or self.ed <= timestamp or self.vo == 0:
            return self.last
        self.last = self.policy()
        return self.last


    
class Buyer(Car):
    def policy(self):
        #global buyers_now
        #buyers_now += 1
        if self.vo >= self.ed - timestamp and np.sum(bid) == 0:
            self.money += 200
            self.vo -= 1
            self.freezed = True
            self.next = timestamp + 1
            print("!")
            return np.zeros((200,), dtype=int)
        
        # how do the orders occupy the range(ask + spread)
        beta_mean = min(self.vo / (self.ed - timestamp) * 1.1 / (math.e ** (self.eco - 0.5)), 0.999)
        # indicates the greedy factor. needs refining
        beta_var = 8 - 6 * self.eco

        beta_a, beta_b = self.abfromuv(beta_mean, beta_var)

        ret = np.zeros((200,), dtype=int)
        tmp = np.random.beta(beta_a, beta_b, self.vo)

        # generate order sequence from beta distribution sample

        direct_order = 0
        for _ in tmp:
            idx = int(round(_ * (minbid + 20)))
            idx = min(idx, 199, gridprice)
            if idx < minbid: ret[idx] += 1
            else: direct_order += 1
        # direct_order = (lastamount + 1) * beta_mean
        it = minbid
        direct_order = min(direct_order, (lastamount + 1) * beta_mean)
        while direct_order > 0 and it < min(minbid + 20, gridprice):
            if (bid[it] > 0): 
                ret[it] += 1
                direct_order -= 1
            it += 1

        #print("buyer with volume " + str(self.vo))
        return ret   

class Seller(Car):
    def policy(self):
        if self.vo >= self.ed - timestamp and np.sum(ask) == 0:
            # different from buyer, do nothing
            return np.zeros((200,), dtype=int)
        
        #global sellers_now
        #sellers_now += 1

        beta_mean = 1 - min(self.vo / (self.ed - timestamp) * 1.1 / (math.e ** (self.eco - 0.5)), 0.999)
        beta_var = 8 - 6 * self.eco

        beta_a, beta_b = self.abfromuv(beta_mean, beta_var)
        #print("got seller beta distribution params: ", beta_mean, beta_var, beta_a, beta_b, self.vo, self.eco, self.ed)
        #input("push to continue")

        ret = np.zeros((200,), dtype=int)
        tmp = np.random.beta(beta_a, beta_b, self.vo)

        # generate order sequence from beta distribution sample
        for _ in tmp:
            idx = int(round(_ * (gridprice + 3 - maxask) + maxask - 3))
            idx = min(idx, gridprice, 199)
            idx = max(idx, 1)
            ret[idx] += 1

        #print("seller with volume " + str(self.vo))
        return ret

def generate_seller():
    
    supply = int(random.betavariate(0.8,2)*344) + 18
    while True:
        st = int(random.betavariate(0.8,2)*600)
        ed = int(random.betavariate(2,0.8)*600)
        if(ed - st < 30):
            continue
        print("seller generated ",st,ed,supply)
        return Seller(st,ed,supply)

def generate_buyer(certain_time):
    
    
    # 10,15
    st = random.randint(certain_time, certain_time + 60)
    
    last = time.time()

    while True:
        if(time.time() - last > 0.1):
            return generate_buyer(certain_time)
        need = int(random.betavariate(0.8,2)*344) + 18
        duration = int(need * (1+random.random()))
        if(duration < need + 10):
            continue
        if(st + duration / 2 >= 600 or st - duration / 2 <= 0):
            continue
        print("buyer generated ",st - duration/2,st + duration/2,need)
        return Buyer(st - duration/2,st + duration/2,need)

def refresh_gridprice():
    global gridprice
    price_list = [117,157,193,200,189,183,180,182,181,174]
    time_list = [0,60,120,180,240,300,360,420,480,540]
    it = len(price_list) - 1
    while it > 0:
        if timestamp > time_list[it]:
            gridprice = price_list[it]
            return
        it -= 1
    gridprice = price_list[0]
            
def generate_buyer_helper1(number, sometime):
    for _ in range(number):
        car_list.append(generate_buyer(sometime))

def generate_buyer_real():
    percent = [
        0.012195122,
        0.008710801,
        0.034843206,
        0.087108014,
        0.209059233,
        0.12195122,
        0.148083624,
        0.226480836,
        0.139372822,
        0.012195122
    ]

    for _ in range(0, 10):
        generate_buyer_helper1(int(round(buyer_TOTAL * percent[_])), _ * 60)
    
    input("check it...")

def generate_buyer_rushed():
    for _ in range(buyer_TOTAL):
        car_list.append(generate_buyer(int(np.random.beta(100,150) * 600)))

def fairness_index(arr):
    #assert isinstance(arr, np.ndarray)
    d = 0.0
    for _ in arr:
        d += _ ** 2
    d = len(arr) * d
    u = (np.sum(arr)) ** 2
    return u / d

def volatility_timesample(arr):
    res = []
    for i in range(200,len(arr)):
        tmp = []
        for j in range(max(1,i-100),i):
            if arr[j - 1] == 0:
                continue
            tmp.append(math.log(arr[j] / arr[j - 1]))
        res.append(np.std(np.array(tmp)))
    plt.ylim(0, 1)
    plt.plot(res)
    plt.savefig("volatility_timesample.png")

def clear_plt():
    plt.gcf().clear()
    plt.clf()
    plt.cla()
    plt.close()
    time.sleep(0.1)
    plt.pause(0.0001)

if __name__ == '__main__':

    """
    Buyer and Seller Generation
    """
    generate_buyer_real()
    #generate_buyer_rushed()

    for _ in range(0,seller_TOTAL):
        car_list.append(generate_seller())
    
    cnt = 0



    while timestamp < timestamp_MAX:
        timestamp += 0.1
        refresh_gridprice()
        for cars in car_list:
            cars.unfreeze()
        if np.sum(ask) != 0: maxask = ask.nonzero()[0][-1]
        if np.sum(bid) != 0: minbid = bid.nonzero()[0][0]

        for price in range(0,200):
            assert not(ask[price]>0 and bid[price]>0)

        bins = np.linspace(0, 200, 200)


        if 1 < 0:
            
            print("fuck")
            plt.ylim(0, 100)
            plt.bar(np.arange(len(ask)), ask, alpha=0.5, label='ask')
            plt.bar(np.arange(len(bid)), bid, alpha=0.5, label='bid')
            plt.legend(loc='upper right')
            plt.savefig(str(1)+'.png')
            plt.gcf().clear()
            plt.clf()
            plt.cla()
            plt.close()
            time.sleep(0.1)
            plt.pause(0.0001)
        cnt += 1

        

        #realtime = time.time()

        #assert(len(car_list) == buyer_TOTAL + seller_TOTAL)

        nowtrade = []
        buyers_now = 0
        sellers_now = 0
        
        ask_for_drawing = np.zeros((200,), dtype=int)
        bid_for_drawing = np.zeros((200,), dtype=int)

        for cars in car_list:
            
            succ = False
            if isinstance(cars, Buyer):
                ask -= cars.last
                orders = cars.submit()
                ask_for_drawing += orders
                for price in reversed(range(0,200)):
                    if succ: break
                    
                    while (orders[price] > 0 and bid[price] > 0) and not succ:
                        for car2 in car_list:
                            if isinstance(car2, Seller) and car2.pricehit(price):
                                bid -= car2.last    # vital
                                cars.freeze(price)
                                car2.freeze(price)
                                #print("Deal at ts[{0}], price = {1}".format(timestamp, price))
                                nowtrade.append(price)
                                #assert 0, "we have a deal"
                                succ = True
                                break
                if np.sum(cars.last) > 0: buyers_now += 1
                ask += cars.last
            else:
                bid -= cars.last
                orders = cars.submit()
                bid_for_drawing += orders
                flag = False
                for price in range(0,200):
                    if succ: break
                    while (orders[price] > 0 and ask[price] > 0) and not succ:
                        #print("can only once")
                        #input("Press Enter to continue...")

                        #print(price, ask[price])

                        if flag: assert 0, "bid fault"

                        for car2 in car_list:
                            
                            if isinstance(car2, Buyer) and car2.pricehit(price):
                                ask -= car2.last    # vital
                                cars.freeze(price)
                                car2.freeze(price)
                                nowtrade.append(price)
                                #print("Deal at ts[{0}], price = {1}".format(timestamp, price))
                                #assert 0, "we have a deal"
                                succ = True
                                break
                        #print("no more loop")
                        flag = True
                bid += cars.last
                if np.sum(cars.last) > 0: sellers_now += 1

        print("timestep[{4:.2f}](grid:{5}) concluded with {0} buyers, {1} sellers and {2} trades with an average price of {3:.2f}.".format(buyers_now, sellers_now, len(nowtrade), np.average(nowtrade), timestamp, gridprice))
        lastamount = len(nowtrade)
        ave = np.average(nowtrade)
        try:
            if np.isnan(ave): ave = trade_history[-1]
        except IndexError:
            ave = 0
        trade_history.append(ave)
        #if(len(leftover) > 0):
        #    print(leftover)

    line_p2p = plt.plot(np.arange(0,600,0.1), trade_history, label='p2p price')
    
    line_grid = plt.plot(
        np.array([0,60,120,180,240,300,360,420,480,540]),
        np.array([117,157,193,200,189,183,180,182,181,174]),
        linestyle='dashed',
        label='grid price'
    )
    
    plt.legend()
    plt.savefig('1.1.png')

    x_axis = []
    y_axis = []

    for cars in car_list:
        if isinstance(cars, Buyer): print("[Buyer]", end='')
        else :print("[Seller:]", end='')
        print("eco:",cars.eco,end=',')
        print("vo:", cars.vo,end=',')
        try:
            print("money:",cars.money / (cars.volume - cars.vo),end='\n')
            if isinstance(cars, Buyer):
                x_axis.append(  cars.eco )
                k = cars.money / (cars.volume - cars.vo)
                #k = k / (np.average(trade_history[int(cars.st * 10):int(cars.ed * 10)]) + 300)
                y_axis.append(k)
        except ZeroDivisionError:
            print("no trade")

    clear_plt()

    plt.scatter(x_axis, y_axis)
    plt.savefig('variance.png')

    clear_plt()

    volatility_timesample(trade_history)

    assert False, "should be here"
    #for cars in 

        
