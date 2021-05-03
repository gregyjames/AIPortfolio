![Logo](https://i.imgur.com/955FRvs.png)
Use AI, Modern Portfolio Theory, and Monte Carlo simulation's to generate a optimized stock portfolio that minimizes risk while maximizing returns.


## How does it work?
The app works by pulling the stock close data from the yahoo finance api. We then calculate the log returns and the volitility of the data to see what the overall trend for the stocks look like. We then generate random portfolio weights and use scipy to maximize a function that calculates the the best portfolio weights for a portfolio with a maximum return to volitility ration (this is known as the sharpe ratio). This is effectivly a monte carlo simulation to find the optimal stock portfolio.


## Resources and Readings

- [Why Log Returns](https://quantivity.wordpress.com/2011/02/21/why-log-returns/)
- [Sharpe Ratio](https://www.investopedia.com/terms/s/sharperatio.asp)
- [Efficient Frontier](https://www.investopedia.com/terms/e/efficientfrontier.asp)
- [Markowitz Modern Portfolio Theory](https://www.investopedia.com/terms/m/modernportfoliotheory.asp)
- [Monte Carlo Simulation](https://www.investopedia.com/terms/m/montecarlosimulation.asp)


## License
MIT License

Copyright (c) 2021 Greg James

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## DISCLAIMER
This project and it's generated portfolios are NOT investment advice. This is purly educational.
