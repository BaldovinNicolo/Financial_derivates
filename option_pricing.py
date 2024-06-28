#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 17:21:38 2024

@author: nicolobaldovin
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import yfinance as yf

class option_pricing:
    def __init__(self, ticker, start_date, end_date, duration, price, strike, r = 0.05):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.T = duration
        self.S = price
        self.K = strike
        self.r = r
        
    def fetch_data(self):
        stock_data = yf.download(self.ticker, self.start_date, self.end_date)
        return stock_data['Adj Close']
    
    def calculate_volatility(self):
        prices = self.fetch_data()
        returns = ((prices-prices.shift(1)) / prices.shift(1)).dropna()
        volatility = returns.std()      # DAILY
        self.volatility = volatility
        self.volatility= self.volatility*np.sqrt(252)
    
    def calculate_drift(self):
        prices = self.fetch_data()
        returns = ((prices-prices.shift(1)) / prices.shift(1)).dropna()
        drift = np.mean(returns)
        self.drift = drift              # DAILY
        self.drift = self.drift*252

    
    def black_scholes_model(self, type_option = "CALL"):
        self.calculate_volatility()
        self.d1 = (math.log(self.S / self.K) + (self.r + 0.5 * self.volatility ** 2) * self.T) / (self.volatility * math.sqrt(self.T))
        self.d2 = self.d1 - self.volatility * math.sqrt(self.T)
        if type_option == 'CALL':
            price = self.S * scipy.stats.norm.cdf(self.d1) - self.K * math.exp(-self.r * self.T) * scipy.stats.norm.cdf(self.d2)
        elif type_option == 'PUT':
            price = self.K * math.exp(-self.r * self.T) * scipy.stats.norm.cdf(-self.d2) - self.S * scipy.stats.norm.cdf(-self.d1)
        return price
    
    def monte_carlo(self, simulation = 200, type_option = "CALL"):
        self.calculate_drift()
        self.calculate_volatility()
        n = 252 * self.T
        stock_price = np.zeros((simulation,n+1))
        stock_price[:,0] = self.S
        dt = self.T / n
        for j in range(simulation):
            for i in range(1, n+1):
                stock_price[j, i] = stock_price[j, i - 1] * np.exp((self.r - 0.5 * self.volatility ** 2) * dt + self.volatility * np.sqrt(dt) * np.random.normal(0,1))
        payoff = np.zeros(n)
        if type_option == "CALL":
            payoff = np.maximum(stock_price[:, -1] - self.K, 0)
        elif type_option == "PUT":
            payoff = np.maximum(self.K - stock_price[:, -1], 0)
        discounted_payoff = np.zeros(simulation)
        discounted_payoff = payoff * np.exp(-self.T*self.r)
        return np.mean(discounted_payoff)
        
        
AAPL = option_pricing("AAPL", "2017-01-01", "2023-01-01", 6, 26, 30)
print("Black-Scholes model price:", AAPL.black_scholes_model())
print("Monte Carlo simulation price:", AAPL.monte_carlo())
