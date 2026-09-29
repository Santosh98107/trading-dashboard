trading_system/
│
├── streamlit_app.py
├── fastapi_server.py
├── config.py
├── auth.py
├── requirements.txt
│
├── core/
│   ├── __init__.py
│   ├── utils.py
│   ├── feature_engine.py
│   ├── backtest_engine.py
│   ├── paper_trader.py
│   ├── recommendation_engine.py
│   ├── probability_engine.py
│   ├── macro_engine.py
│   ├── market_context_engine.py
│   ├── options_sentiment_engine.py
│   ├── volume_engine.py
│   ├── regime_engine.py
│   ├── signal_fusion_engine.py
│   └── live_signal_engine.py
│
└── data/
    ├── price.csv
    ├── macro.csv
    ├── breadth.csv
    ├── banks.csv
    └── options.csv
