# Bankruptcy Risk Analyzer

Bankruptcy Risk Analyzer is an agent-based system for counterparty verification focused on preliminary bankruptcy risk assessment. The service combines financial data processing, machine learning prediction, rule-based financial diagnostics, official company verification, arbitration case analysis, and an integrated risk summary.

The project was developed as part of a master's thesis on corporate bankruptcy risk assessment and the development of an agent-based system for counterparty verification.

## Project Purpose

The service is designed to support preliminary counterparty screening before cooperation with a client or supplier begins. It helps structure the assessment of financial distress risk by combining:

- financial statement indicators;
- machine learning-based bankruptcy probability;
- rule-based interpretation of financial ratios;
- FNS-based company verification;
- KAD Arbitr legal context;
- role-sensitive interpretation for clients and suppliers.

The system should be treated as a decision-support tool. It does not replace professional financial, legal, or credit analysis.

## Main Features

- Editable financial data table
- Bankruptcy probability estimation using a trained Random Forest model
- Risk classification by probability level
- Yearly bankruptcy risk dynamics
- Rule-based financial diagnostics
- Client / supplier role selection
- FNS data module
- KAD Arbitr legal context module
- Integrated Summary page with selected priority factors
- Local execution through Streamlit

## Repository Structure

```text
bankruptcy-analyzer/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── runtime.txt            # Python runtime version for deployment compatibility
├── README.md              # Project description and launch instructions
└── .gitignore             # Files excluded from Git tracking
