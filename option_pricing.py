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


# Introduction:
# This script provides classes for pricing European and American options using different methods,
# such as the Black-Scholes model, Monte Carlo simulations, and the Barone-Adesi Whaley approximation.
# The classes also include methods for calculating option greeks and fetching historical stock data.


class european_option_pricing:
    def __init__(self, ticker, start_date, end_date, duration, price, strike, r = 0.05, type_option = "CALL"):
        
        """
        Initialize the European option pricing class with required parameters.
        
        Parameters
        ----------
        ticker : str
            Stock ticker symbol.
        start_date : str
            Start date for historical data in 'YYYY-MM-DD' format.
        end_date : str
            End date for historical data in 'YYYY-MM-DD' format.
        duration : float
            Duration of the option in years.
        price : float
            Current stock price.
        strike : float
            Strike price of the option.
        r : float, optional
            Risk-free interest rate. Default is 0.05.
        type_option : str, optional
            Type of option, either "CALL" or "PUT". Default is "CALL".
        """
        
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.T = duration
        self.S = price
        self.K = strike
        self.r = r
        self.type_option = type_option
        
    def fetch_data(self):
        
        """
        Fetch historical stock data for the given ticker and date range.
        
        Returns
        -------
        pandas.Series
            Adjusted closing prices of the stock.
            
        """
        stock_data = yf.download(self.ticker, self.start_date, self.end_date)
        return stock_data['Adj Close']
    
    def calculate_volatility(self):
        
        """
        Calculate annualized volatility of the stock based on historical data.
        """
        
        prices = self.fetch_data()
        returns = ((prices-prices.shift(1)) / prices.shift(1)).dropna()
        volatility = returns.std()      # DAILY
        self.volatility = volatility
        self.volatility= self.volatility*np.sqrt(252)
    
    def calculate_drift(self):
        
        """
        Calculate annualized drift (mean return) of the stock based on historical data.
        """
        
        prices = self.fetch_data()
        returns = ((prices-prices.shift(1)) / prices.shift(1)).dropna()
        drift = np.mean(returns)
        self.drift = drift              # DAILY
        self.drift = self.drift*252

    
    def black_scholes_model(self):
        
        """
        Calculate the option price using the Black-Scholes model.
        
        Returns
        -------
        float
            Option price.
        """
        
        self.calculate_volatility()
        self.d1 = (math.log(self.S / self.K) + (self.r + 0.5 * self.volatility ** 2) * self.T) / (self.volatility * math.sqrt(self.T))
        self.d2 = self.d1 - self.volatility * math.sqrt(self.T)
        if self.type_option == 'CALL':
            price = self.S * scipy.stats.norm.cdf(self.d1) - self.K * math.exp(-self.r * self.T) * scipy.stats.norm.cdf(self.d2)
        elif self.type_option == 'PUT':
            price = self.K * math.exp(-self.r * self.T) * scipy.stats.norm.cdf(-self.d2) - self.S * scipy.stats.norm.cdf(-self.d1)
        return price
    
    def greeks(self):
        
        """
        Calculate the option greeks: vega, gamma, delta, theta, and rho.
        
        Returns
        -------
        tuple
            A tuple containing vega, gamma, delta, theta, and rho.
        """
        
        vega = self.S * scipy.stats.norm.pdf(self.d1) * math.sqrt(self.T) / 100  # 1% change in volatility
        gamma = scipy.stats.norm.pdf(self.d1) / (self.S * self.volatility * math.sqrt(self.T))
        if self.type_option == "CALL":
            delta = scipy.stats.norm.cdf(self.d1)
            theta = (-self.S * scipy.stats.norm.pdf(self.d1) * self.volatility / (2 * math.sqrt(self.T)) - self.r * self.K * math.exp(-self.r * self.T) * scipy.stats.norm.cdf(self.d2))
            rho = self.K * self.T * math.exp(-self.r * self.T) * scipy.stats.norm.cdf(self.d2) / 100  # 1% change in interest rate
        elif self.type_option == "PUT":
            delta = scipy.stats.norm.cdf(self.d1) - 1
            theta = (-self.S * scipy.stats.norm.pdf(self.d1) * self.sigma / (2 * math.sqrt(self.T)) + self.r * self.K * math.exp(-self.r * self.T) * scipy.stats.norm.cdf(-self.d2))
            rho = -self.K * self.T * math.exp(-self.r * self.T) * scipy.stats.norm.cdf(-self.d2) / 100  # 1% change in interest rate
        return vega, gamma, delta, theta, rho

    def monte_carlo(self, simulation = 200):
        
        """
        Calculate the option price using Monte Carlo simulation.
        
        Parameters
        ----------
        simulation : int, optional
            Number of simulation paths (default is 200).
        
        Returns
        -------
        float
            Option price based on Monte Carlo simulation.
        """
        
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
        if self.type_option == "CALL":
            payoff = np.maximum(stock_price[:, -1] - self.K, 0)
        elif self.type_option == "PUT":
            payoff = np.maximum(self.K - stock_price[:, -1], 0)
        discounted_payoff = np.zeros(simulation)
        discounted_payoff = payoff * np.exp(-self.T*self.r)
        return np.mean(discounted_payoff)
        
class american_option_pricing:
    def __init__(self, ticker, start_date, end_date, duration, price, strike, r = 0.05, type_option = "CALL"):
        
        """
        Initialize the American option pricing class with required parameters.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol.
        start_date : str
            Start date for historical data in 'YYYY-MM-DD' format.
        end_date : str
            End date for historical data in 'YYYY-MM-DD' format.
        duration : float
            Duration of the option in years.
        price : float
            Current stock price.
        strike : float
            Strike price of the option.
        r : float, optional
            Risk-free interest rate. Default is 0.05.
        type_option : str, optional
            Type of option, either "CALL" or "PUT". Default is "CALL".
        """
        
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.T = duration
        self.S = price
        self.K = strike
        self.r = r
        self.type_option = type_option
        
    def fetch_data(self):
        
        """
        Fetch historical stock data for the given ticker and date range.
        
        Returns
        -------
        pandas.Series
            Adjusted closing prices of the stock.
        """
        
        stock_data = yf.download(self.ticker, self.start_date, self.end_date)
        return stock_data['Adj Close']
        
    def calculate_volatility(self):
        
        """
        Calculate annualized volatility of the stock based on historical data.
        """
        
        prices = self.fetch_data()
        returns = ((prices-prices.shift(1)) / prices.shift(1)).dropna()
        volatility = returns.std()      # DAILY
        self.volatility = volatility
        self.volatility= self.volatility*np.sqrt(252)
        
    def calculate_drift(self):
        
        """
        Calculate annualized drift (mean return) of the stock based on historical data.
        """
        
        prices = self.fetch_data()
        returns = ((prices-prices.shift(1)) / prices.shift(1)).dropna()
        drift = np.mean(returns)
        self.drift = drift              # DAILY
        self.drift = self.drift*252
        
    def barone_adesi_whaley(self):
        
        """
        Calculate the American option price using the Barone-Adesi Whaley model.
        
        Returns
        -------
        float
            American option price.
        """
        
        underlying = european_option_pricing(self.ticker, self.start_date, self.end_date, self.T, self.S, self.K) 
        european_price = underlying.black_scholes_model()
        self.calculate_drift()
        self.calculate_volatility()
        m = 2*self.drift/self.volatility**2
        n = 2*self.drift/(self.volatility**2*(1-np.exp(-self.T*self.drift)))
        critical_price = (self.K/(1-1/m)) * (1-np.exp((n-1)*np.log(self.K/(self.K+european_price))))
        if self.S >= critical_price:
            premium = self.S-self.K
        else:
            premium = (critical_price-self.K)*(self.S/critical_price)**m
        return premium + european_price
        
if __name__ == '__main__':        
    AAPL = european_option_pricing("AAPL", "2017-01-01", "2023-01-01", 6, 100, 130)
    print("Black-Scholes model price:", AAPL.black_scholes_model())
    print("Monte Carlo simulation price:", AAPL.monte_carlo())
    greeks = AAPL.greeks()
    print("Vega:", f"{greeks[0]:.4f}", "Gamma:", f"{greeks[1]:.4f}", "Delta:", f"{greeks[2]:.4f}", "Theta:", f"{greeks[3]:.4f}", "Rho:", f"{greeks[4]:.4f}")
    AAPL2 = american_option_pricing("AAPL", "2017-01-01", "2023-01-01", 6, 100, 130)
    print(AAPL2.barone_adesi_whaley())
    
    
    