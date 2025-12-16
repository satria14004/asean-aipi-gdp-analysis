# AI Preparedness Index and GDP Analysis - ASEAN Region

## Overview
This project analyzes the relationship between the IMF AI Preparedness Index (AIPI) 
and GDP per capita across 10 ASEAN countries (2020-2023). Using Ordinary Least Squares (OLS) regression analysis to quantify how AI preparedness correlates with economic productivity (GDP).

## Quick Start
```bash
# Clone repository
git clone https://github.com/YOUR-USERNAME/aipi-gdp-analysis.git
cd aipi-gdp-analysis

# Install dependencies
pip install -r requirements.txt

# Run analysis
python src/analysis.py
```

## Key Findings
- Countries with higher AIPI scores demonstrate significantly higher GDP per capita
- The relationship shows a statistically significant positive correlation
- Interaction effects suggest differential impacts by AI preparedness level

## Data Sources
- **AI Preparedness Index**: IMF DataMapper API
- **GDP per Capita**: IMF DataMapper API (NGDPDPC indicator)
- **Time Period**: 2020-2023
- **Countries**: Indonesia, Singapore, Malaysia, Thailand, Vietnam, Philippines, 
  Cambodia, Laos, Myanmar, Brunei

## Project Structure
- `src/analysis.py` - Main analysis script
- `notebooks/exploratory_analysis.ipynb` - Jupyter notebook with step-by-step analysis
- `data/` - Raw and processed datasets
- `results/` - Visualizations and regression output
- `METHODOLOGY.md` - Detailed explanation of methods
- `FINDINGS.md` - Detailed findings and interpretation

## Methodology
See [METHODOLOGY.md](METHODOLOGY.md) for detailed explanation of:
- Data collection process
- Regression specifications
- Interaction term rationale
- Statistical interpretation

## Requirements
- Python 3.8+
- See `requirements.txt` for full list

## License
MIT License - see LICENSE file

## Author
[Satria Mahesya Muhammad]

## Contact
[muhammadsatria96@gmail.com]
