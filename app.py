from flask import Flask, request, render_template, redirect, url_for, session, send_file
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimization
import json
import io
import os
from common import cache
from datetime import date

app = Flask(__name__)
cache.init_app(app=app, config={"CACHE_TYPE": "simple"})
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# on average there are 252 trading days in a year
NUM_TRADING_DAYS = 252
# we will generate random w (different portfolios)
NUM_PORTFOLIOS = 10000

def download_data(stocks, start_date,end_date):
    # name of the stock (key) - stock values (in the date range) as the values
    stock_data = {}

    for stock in stocks:
        # closing prices from yahoo finance api
        ticker = yf.Ticker(stock)
        stock_data[stock] = ticker.history(start=start_date, end=end_date)['Close']

    return pd.DataFrame(stock_data)


#dynamically generate the returns plot to flask
@app.route('/plot.png', methods=['GET'])
def show_data():
    dataset.plot(figsize=(10, 5))
    #matplotlib chart to byte array 
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)

    #send the file to flask
    return send_file(bytes_image,mimetype='image/png', max_age=0, as_attachment=True)

#dynamically generate the efficient frontier plot to flask
@app.route('/portfolio.png', methods=['GET'])
def show_optimal_portfolio():
    bytes_image = io.BytesIO()
    plt.figure(figsize=(10, 6))
    plt.scatter(risks, means, c=means / risks, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.plot(statistics(optimum['x'], log_daily_returns)[1], statistics(optimum['x'], log_daily_returns)[0], 'g*', markersize=20.0)
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)

    #send the dynamic file to flash
    bytes_obj = bytes_image;
    return send_file(bytes_obj,mimetype='image/png', max_age=0)

#Log distribution is assumed for computational finance
def calculate_return(data):
    # NORMALIZATION - to measure all variables in comparable metric
    log_return = np.log(data / data.shift(1))
    return log_return[1:]


def show_statistics(returns):
    # instead of daily metrics we are after annual metrics
    # mean of annual return
    print(returns.mean() * NUM_TRADING_DAYS)
    print(returns.cov() * NUM_TRADING_DAYS)


def show_mean_variance(returns, weights):
    # we are after the annual return so we multiply by the number of trading days
    portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov()
                                                            * NUM_TRADING_DAYS, weights)))
    print("Expected portfolio mean (return): ", portfolio_return)
    print("Expected portfolio volatility (standard deviation): ", portfolio_volatility)

def generate_portfolios(returns):
    portfolio_means = []
    portfolio_risks = []
    portfolio_weights = []

    #generate NUM_PORTFOLIOS with random weights
    for _ in range(NUM_PORTFOLIOS):
        #array of weights equal to the list of stocks provided
        w = np.random.random(len(stocks))
        w /= np.sum(w)
        #append weights of portfolio to total
        portfolio_weights.append(w)
        #append returns of portfolio to total
        portfolio_means.append(np.sum(returns.mean() * w) * NUM_TRADING_DAYS)
        #append risks of portfolio to total
        portfolio_risks.append(np.sqrt(np.dot(w.T, np.dot(returns.cov()
                                                          * NUM_TRADING_DAYS, w))))

    return np.array(portfolio_weights), np.array(portfolio_means), np.array(portfolio_risks)

#return the returns, volatility, and sharpe ratio for the portfolio
def statistics(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov()
                                                            * NUM_TRADING_DAYS, weights)))
    return np.array([portfolio_return, portfolio_volatility,
                     portfolio_return / portfolio_volatility])


# scipy optimize module can find the minimum of a given function
# the maximum of a f(x) is the minimum of -f(x)
# 0 optimizes by return
# 1 optimizes by volitility
# 2 optimizes by sharpe ratio
def min_function_sharpe(weights, returns):
    return -statistics(weights, returns)[2]


# what are the constraints? The sum of weights = 1
# f(x)=0 this is the function to minimize
def optimize_portfolio(weights, returns):
    # the sum of weights is 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # the weights can be 1 at most: 1 when 100% of money is invested into a single stock
    bounds = tuple((.05, 1) for _ in range(len(stocks)))
    # minimize the min function with scipy
    return optimization.minimize(fun=min_function_sharpe, x0=weights[0], args=returns
                                 , method='SLSQP', bounds=bounds, constraints=constraints)


def print_optimal_portfolio(optimum, returns):
    return optimum['x'].round(3), statistics(optimum['x'].round(3), returns)

#route for home
@app.route("/", methods = ['POST', 'GET'])
def home():
    #get the ajax post from jquery
    if request.method == 'POST':
        #store the users stocks as a flask session cookie
        #session['stocks'] = request.json
        #if the stocks cookie is not empty find redirect to the portfolio template
        cache.set("stocks", request.json)
        return redirect("/portfolio")
    else:
        #render the home page on get request
        return render_template("home.html")

dataset = pd.DataFrame()

#route for portfolio page
@app.route("/portfolio", methods=['GET'])
def portfolio():
    #get the sessions stocks cookie and make it a global variable
    global stocks
    #stocks = session.get('stocks', None)
    stocks = cache.get("stocks")
    print(stocks)
    # historical data - define START and END dates
    start_date = '2011-01-01'
    end_date = date.today().strftime("%Y-%m-%d")
    
    global dataset
    #download the dataset and make it a global variable
    if stocks is not None:
        dataset = download_data(stocks,start_date,end_date)

    if dataset.empty == False:
        #calculate the log returns and make it a global variable
        global log_daily_returns
        log_daily_returns = calculate_return(dataset)
        # show_statistics(log_daily_returns)

        #Calculate the weights, returns and vol of all the portfolios and make them global
        #global pweights
        #global means
        #global risks

        pweights, means, risks = generate_portfolios(log_daily_returns)

        # find and print the optimal portfolio weights
        global optimum
        optimum = optimize_portfolio(pweights, log_daily_returns)
        weights, facts = print_optimal_portfolio(optimum, log_daily_returns)

        #render the portfolio template with the calcuated data
        return render_template('portfolio.html', data=stocks, weights=weights, returns=facts[0], vol=facts[1], sharpe=facts[2])
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=True)