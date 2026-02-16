# Trader Performance vs Market Sentiment (Fear/Greed) on Hyperliquid

## Data Science Internship Assignment

This project analyzes how Bitcoin market sentiment (Fear/Greed index) affects trader behavior and performance on the Hyperliquid exchange platform.

## Project Overview

### Objective
Analyze the relationship between Bitcoin market sentiment (Fear/Greed Index) and trader performance metrics on Hyperliquid, including:
- Profit/Loss (PnL) patterns
- Win rates
- Trade frequency
- Position sizing
- Leverage usage
- Trader segmentation

### Datasets
1. **Bitcoin Market Sentiment (fear_greed_index.csv)**
   - Date, Fear/Greed index value (0-100)
   - Classification: Extreme Fear, Fear, Neutral, Greed, Extreme Greed

2. **Historical Trader Data (historical_data.csv)**
   - Account, Coin, Execution Price, Size Tokens, Size USD
   - Side, Timestamp, Start Position, Direction
   - Closed PnL, Transaction Hash, Order ID, Fee, Trade ID

## Folder Structure

```
.
├── analysis.py           # Main analysis script
├── analysis.ipynb        # Jupyter Notebook (optional)
├── README.md            # This file
├── report.md            # Detailed analysis report
├── fear_greed_index.csv # Bitcoin sentiment data
├── historical_data.csv  # Hyperliquid trader data
├── output/              # Generated charts and tables
│   ├── sentiment_performance_overview.png
│   ├── segment_analysis.png
│   ├── time_series_analysis.png
│   ├── distribution_analysis.png
│   ├── summary_table_1.csv
│   ├── summary_table_2.csv
│   └── summary_table_3.csv
└── .gitignore
```

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Dependencies
Install the required Python packages:

```
bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Running the Analysis

1. **Run as Python script:**
   
```
bash
   python analysis.py
   
```

2. **Run in Jupyter Notebook:**
   - Open Jupyter: `jupyter notebook`
   - Open `analysis.ipynb`
   - Run cells sequentially

## Key Results

### Findings Summary

The analysis reveals significant insights into how market sentiment affects trader behavior:

1. **Sentiment Impact on PnL:**
   - Total PnL varies significantly between Fear, Neutral, and Greed days
   - Average daily PnL shows distinct patterns based on sentiment

2. **Win Rate Analysis:**
   - Win rates differ between Fear and Greed market conditions
   - Trader profitability is sentiment-dependent

3. **Behavioral Changes:**
   - Trade frequency changes with market sentiment
   - Position sizing and leverage usage vary by sentiment

4. **Trader Segments:**
   - High vs Low frequency traders respond differently to sentiment
   - Position sizing strategies differ across market conditions

### Strategy Recommendations

Based on the analysis, three actionable strategies are recommended:

1. **Risk-Averse Traders During Fear Days**
   - Who: Low-leverage, risk-averse traders
   - When: During Fear or Extreme Fear (index < 35)
   - Why: Reduced position sizes minimize potential losses during high volatility

2. **High-Frequency Traders During Greed Days**
   - Who: Active traders with consistent patterns
   - When: During Greed or Extreme Greed (index > 60)
   - Why: Capitalize on momentum-driven rallies

3. **Trend-Following During Neutral Markets**
   - Who: All trader types, especially systematic traders
   - When: During Neutral conditions (index 35-60)
   - Why: Clearer signals without emotion-driven volatility

## Output Files

The analysis generates the following outputs:

### Charts (PNG)
- `sentiment_performance_overview.png` - Main performance metrics by sentiment
- `segment_analysis.png` - Trader segment analysis
- `time_series_analysis.png` - PnL and sentiment over time
- `distribution_analysis.png` - PnL distribution by sentiment

### Summary Tables (CSV)
- `summary_table_1.csv` - Overall performance by sentiment
- `summary_table_2.csv` - Win rate analysis
- `summary_table_3.csv` - Segment performance

### ML Model Output
- `feature_importance.csv` - Feature importance for profitability prediction

## Methodology

### Data Processing
1. Load and clean both datasets
2. Handle missing values and duplicates
3. Convert timestamps to datetime
4. Merge datasets on date
5. Create sentiment categories (Fear/Neutral/Greed)

### Metrics Calculated
- Daily PnL per trader
- Win rate (profitable days / total days)
- Average trade size
- Trade frequency
- Volatility (PnL standard deviation)
- Active traders per day

### Analysis Performed
- Statistical comparison of Fear vs Greed days
- Trader segmentation (frequency, size, consistency)
- Segment-specific sentiment analysis
- Time series visualization
- Distribution analysis

### ML Model (Bonus)
- Random Forest classifier for profitability prediction
- Features: sentiment, trade metrics, lag values
- Evaluation: Accuracy, ROC-AUC score

## Limitations

1. **Data Coverage:** Analysis limited to date overlap between sentiment and trader data
2. **Proxy Metrics:** Leverage estimated from position sizes rather than actual leverage data
3. **External Factors:** Market sentiment is one of many factors affecting trader performance
4. **Historical Context:** Results may vary in different market conditions

## Future Improvements

1. **Enhanced Segmentation:** More sophisticated trader clustering
2. **Time Series Models:** ARIMA/SARIMA for sentiment forecasting
3. **Causal Analysis:** Propensity score matching for causal inference
4. **Real-time Integration:** Live sentiment data for trading signals
5. **Cross-exchange Analysis:** Compare behavior across multiple exchanges

## Author

Data Science Intern
Date: 2024

## License

This project is for educational purposes as part of a Data Science internship assignment.
#
