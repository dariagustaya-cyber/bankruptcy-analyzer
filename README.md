# Bankruptcy Risk Analyzer

Bankruptcy Risk Analyzer is a local Streamlit application for preliminary counterparty bankruptcy risk assessment. The service was developed as part of a master's thesis on the development of an agent-based system for counterparty verification.

The application combines financial data processing, machine learning prediction, rule-based financial diagnostics, FNS company verification, KAD Arbitr legal context, and an integrated risk summary.

## Quick Start

Follow these steps to run the service locally.

### 1. Clone the repository

```bash
git clone https://github.com/dariagustaya-cyber/bankruptcy-analyzer.git
cd bankruptcy-analyzer
```

### 2. Create and activate a virtual environment

Python 3.11 is recommended.

For macOS / Linux:

```bash
python3.11 -m venv venv
source venv/bin/activate
```

For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in the browser. If it does not open automatically, use the local URL shown in the terminal:

```text
http://localhost:8501
```

## What the Service Does

The service supports preliminary counterparty screening before cooperation with a client or supplier begins.

Main functions:

- financial data input and editing;
- bankruptcy probability estimation using a trained Random Forest model;
- yearly bankruptcy risk dynamics;
- rule-based financial diagnostics;
- client / supplier role selection;
- FNS-based company verification;
- KAD Arbitr legal context analysis;
- integrated risk summary.

## Repository Structure

```text
bankruptcy-analyzer/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── runtime.txt            # Python runtime version
├── README.md              # Launch instructions and project description
└── model_cache/           # Created automatically after model download
```

The `model_cache/` folder is not uploaded to GitHub.

## Required Files

Before running the service, make sure the repository contains:

```text
app.py
requirements.txt
runtime.txt
README.md
```

The model file is downloaded from Google Drive after launch.

## Input Options

The application supports several input scenarios:

1. automatic company data retrieval where available;
2. file upload with prepared financial data;
3. manual editing of financial indicators before running the model.

## Analytical Logic

The service follows a modular agent-based workflow:

1. data input and validation;
2. feature engineering;
3. machine learning prediction;
4. rule-based financial interpretation;
5. external verification through FNS and KAD Arbitr;
6. final aggregation into an integrated risk profile.

## Important Notes

- The service is intended for local demonstration and research purposes.
- Python 3.11 is recommended for compatibility with the trained model bundle.
- The model is loaded from Google Drive because the `.pkl` file is too large for GitHub.
- API keys are required for some external data modules.
- The application is not a production credit scoring system.
- Final decisions should involve additional financial, legal, and managerial review.

## Troubleshooting

### The model does not load

Check that:

- the Google Drive link is inserted into `app.py`;
- the file is shared as `Anyone with the link can view`;
- the file is accessible in the browser;
- the model file is the correct `bankruptcy_rf_bundle.pkl`.

### Streamlit does not start

Try reinstalling dependencies:

```bash
pip install -r requirements.txt
streamlit run app.py
```

### There is a Python version issue

Use Python 3.11:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### External modules do not work

Check API keys:

```bash
echo $PARSER_API_KEY
echo $ARBITR_API_KEY
```

## Thesis Context

The project was developed as an applied part of a master's thesis on bankruptcy risk assessment and counterparty verification. The empirical model was trained on financial data of Russian mining and manufacturing companies, and the final Random Forest model was implemented in the analytical service as the core predictive component.

## Author

Daria Gustaia  
Master's Programme: Business Analytics and Big Data Systems  
National Research University Higher School of Economics
