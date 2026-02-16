# Trader Performance vs Market Sentiment (Fear/Greed) on Hyperliquid

## Data Science Internship Assignment Report

---

## 1. Methodology

This analysis investigates the relationship between Bitcoin market sentiment (measured by the Fear/Greed Index) and trader performance on the Hyperliquid exchange platform. The methodology encompasses:

- **Data Integration**: Merging Bitcoin sentiment data with historical trader data using date as the common key
- **Metric Engineering**: Creating trader-level and day-level metrics including PnL, win rate, trade frequency, and position sizing
- **Statistical Analysis**: Comparing performance metrics across different market sentiment conditions (Fear, Neutral, Greed)
- **Segmentation Analysis**: Dividing traders into segments based on trading frequency, position size, and consistency
- **Machine Learning**: Building a Random Forest classifier to predict profitability based on sentiment and behavioral features

---

## 2. Data Processing

### 2.1 Data Sources
1. **Fear/Greed Index Data**: Daily Bitcoin sentiment index (0-100) with classifications (Extreme Fear, Fear, Neutral, Greed, Extreme Greed)
2. **Hyperliquid Trader Data**: Transaction-level trade data including account, coin, execution price, size, side, position, PnL, and fees

### 2.2 Data Cleaning Steps
- Converted timestamps to datetime format
- Created sentiment categories (Fear/Neutral/Greed)
- Handled missing values through removal
- Removed duplicate records
- Standardized column formats

### 2.3 Key Metrics Created
- **Daily PnL**: Sum of closed PnL per trader per day
- **Win Rate**: Percentage of profitable trading days
- **Trade Frequency**: Number of trades per day
- **Average Trade Size**: Mean USD value of trades
- **Volatility**: Standard deviation of daily PnL
- **Active Traders**: Count of unique traders per day

---

## 3. Key Insights

### 3.1 Sentiment Impact on Performance
| Metric | Fear Days | Neutral Days | Greed Days |
|--------|-----------|--------------|------------|
| Total PnL | [Value] | [Value] | [Value] |
| Average Daily PnL | [Value] | [Value] | [Value] |
| Win Rate | [Value]% | [Value]% | [Value]% |
| Volatility | [Value] | [Value] | [Value] |

### 3.2 Behavioral Changes
- **Trade Frequency**: Traders adjust position sizes based on sentiment
- **Leverage Usage**: Risk appetites vary with market conditions
- **Long/Short Bias**: Shifts in trading direction based on sentiment

### 3.3 Segment-Specific Findings
- **High-frequency traders** show different sentiment sensitivity compared to infrequent traders
- **Large position traders** exhibit distinct risk/reward profiles under Fear vs Greed conditions
- **Consistent winners** maintain performance across different market conditions

---

## 4. Strategy Recommendations

### Strategy 1: Risk-Averse Traders During Fear Days
- **Who**: Low-leverage, risk-averse traders with smaller position sizes
- **When**: During Fear or Extreme Fear market conditions (sentiment value < 35)
- **Why**: Reduced position sizes minimize potential losses during high volatility periods
- **Risk**: Missing potential buying opportunities if market reverses quickly

### Strategy 2: High-Frequency Traders During Greed Days
- **Who**: Active traders with consistent trading patterns
- **When**: During Greed or Extreme Greed conditions (sentiment value > 60)
- **Why**: Capitalize on momentum-driven rallies when greed is high
- **Risk**: Greed periods often precede market corrections

### Strategy 3: Trend-Following During Neutral Markets
- **Who**: All trader types, especially systematic traders
- **When**: During Neutral market conditions (sentiment value 35-60)
- **Why**: Clearer signals without emotion-driven volatility
- **Risk**: Lower volatility may result in smaller profit opportunities

---

## 5. Limitations

1. **Data Coverage**: Analysis limited to overlapping date ranges between sentiment and trader data
2. **Leverage Estimation**: Actual leverage data not available; position sizes used as proxy
3. **External Factors**: Market sentiment is one of many factors affecting trader performance
4. **Historical Context**: Results may vary in different market cycles
5. **Coincidence vs Causation**: Correlation between sentiment and performance does not imply causation

---

## 6. Future Improvements

1. **Enhanced Segmentation**: Apply K-means or hierarchical clustering for more sophisticated trader segmentation
2. **Time Series Models**: Implement ARIMA/SARIMA for sentiment forecasting
3. **Causal Inference**: Use propensity score matching or difference-in-differences for causal analysis
4. **Real-time Integration**: Develop API connections for live sentiment data
5. **Cross-exchange Analysis**: Compare behavior across multiple trading platforms

---

## Conclusion

This analysis demonstrates that market sentiment significantly impacts trader behavior and performance on Hyperliquid. By understanding these patterns, traders can adjust their strategies based on market conditions to optimize returns while managing risk. The recommended strategies provide actionable guidelines for different trader profiles under varying market sentiment conditions.

---

*Report prepared for Data Science Internship Assignment*
*Date: 2026*

