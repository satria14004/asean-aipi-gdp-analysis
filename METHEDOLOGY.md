# Methodology

## Data Sources
### AI Preparedness Index (AIPI)
- Source: IMF DataMapper API (`AI_PI` endpoint)
- Measures a country's readiness to adopt and implement AI technologies
- Includes factors like human capital, infrastructure, institutional framework

### GDP Per Capita
- Source: IMF DataMapper API (`NGDPDPC` endpoint)
- Current prices in US dollars
- Proxy for economic productivity and development

## Analysis Approach

### 1. Data Collection
- Fetched AIPI data for 10 ASEAN countries (2023)
- Fetched GDP per capita for same countries and period (2023)
- Merged datasets on ISO3 country code and year

### 2. Data Transformation
- Log-transformed GDP per capita to normalize distribution
- Formula: `log(GDP) = ln(GDP per capita)`
- Rationale: Captures diminishing returns to scale ()

### 3. Baseline OLS Regression
**Model Specification:**

log(GDP_i,t) = β₀ + β₁ × AIPI_i,t + ε_i,t

Where:
- `i` = country
- `t` = year
- `ε` = error term

**Interpretation:**
- β₁ coefficient = percentage change in GDP per capita for 1-unit increase in AIPI
- Example: if β₁ = 0.15, then 1-unit increase in AIPI → 15% increase in GDP

### 4. Interaction Model
**Model Specification:**

log(GDP_i,t) = β₀ + β₁ × AIPI_i,t + β₂ × Low_AIPI_i,t + β₃ × (AIPI × Low_AIPI)_i,t + ε_i,t

Where:
- `Low_AIPI` = dummy variable (1 if AIPI < median, 0 otherwise)
- Allows AIPI effect to differ for high vs. low AI-prepared countries

**Rationale:** Tests whether AI preparedness has differential impacts depending 
on initial level—does AI help more when countries are already prepared?

## Statistical Tests
- **T-statistics**: Test if individual coefficients differ from zero
- **F-statistic**: Test overall model fit
- **R-squared**: Proportion of variance explained by model
- **Robust standard errors**: Account for potential heteroskedasticity

## Limitations
- Small sample size 
- Cross-sectional variation dominates time variation
- Omitted variables (education, infrastructure, etc.) may bias results
- AIPI is relatively new index with limited historical data (only 2023)
- Correlation ≠ causation: GDP may drive AIPI investment, not vice versa