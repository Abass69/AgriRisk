# Dataset card and attribution

## Sources and rights

1. **The World Bank: Commodity Price Data (The Pink Sheet).** Monthly Prices sheet, Wheat US HRW, Maize, and Crude oil Brent. Original definitions and third-party source notes are retained in the workbook's Description sheet. [Dataset landing page](https://www.worldbank.org/en/research/commodity-markets). [Dataset terms](https://www.worldbank.org/ext/en/legal/terms-conditions/datasets). [Data licensing](https://datacatalog.worldbank.org/public-licenses). The default World Bank dataset licence is CC BY 4.0 with additional terms, unless metadata states otherwise; third-party material may have separate restrictions. Do not interpret the code's MIT licence as relicensing source data.
2. **Bank of Canada: FXUSDCAD**, Canadian dollars per US dollar, daily indicative exchange rates. [Valet documentation](https://www.bankofcanada.ca/valet/docs/), [terms and disclaimers](https://www.bankofcanada.ca/terms/), [rate definitions](https://www.bankofcanada.ca/rates/exchange/daily-exchange-rates/). Used under Bank of Canada permission to reproduce with attribution, subject to its exceptions and terms; not represented as CC BY. Underlying FX data are sourced from LSEG. These are indicative analytical rates, not executable transaction prices.

Changes made: select columns and dates, convert numeric values, average daily FX by calendar month, align monthly dates, derive CAD wheat-price proxy, and generate lag/volatility features. Neither source endorses this project. Retain this attribution when sharing derived data.

## Scope and schema

| Column | Unit | Interpretation |
|---|---|---|
| month | First day of month | Period label, not publication timestamp |
| wheat_usd_t | USD / metric tonne | US hard red winter wheat benchmark |
| maize_usd_t | USD / metric tonne | Related agricultural market input |
| brent_usd_bbl | USD / barrel | Energy-market context proxy |
| cad_per_usd | CAD / USD | Arithmetic mean of observed daily rates |
| wheat_cad_t | CAD / metric tonne | Monthly wheat price multiplied by monthly mean FX |

Sample: January 2017–December 2025. Daily FX observations are business-day observations; weekends and holidays are not imputed. The product of two monthly averages is a benchmark proxy, not an exact average of daily converted wheat prices. There are no firm invoices, customer records or personal data.

## Quality and missingness

Require a unique, ordered, complete monthly index, positive finite source values, and at least 15 FX observations per month. Source ellipses/non-numeric prices become missing and fail validation in the study window. No future interpolation, backward filling or synthetic replacement is performed. Expected warm-up missing feature rows are dropped only from training. Class imbalance is not applicable to continuous regression; rare shock periods are evaluated separately but are too few for strong tail-risk conclusions.

## Reproducibility and availability

The raw XLSX and API JSON are cached. `data/manifest.json` records exact input hashes, URLs, processed range and audit counts. The workbook was published in November 2026; the analytical cutoff is December 2025. This is a revised snapshot, not a vintage dataset. Monthly features are conservatively delayed two period labels, but actual historical release timestamps/revisions are not reconstructed.
