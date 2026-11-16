# Five-minute demonstration

1. Launch `streamlit run app.py`. Explain the Canadian processor case and the historical sample.
2. Set wheat to USD 300/t, FX to CAD 1.35/USD, purchase volume to 1,000 tonnes, sales to CAD 600,000, other costs to CAD 120,000 and freight to CAD 30/t.
3. With no shocks, materials cost CAD 405,000; total costs CAD 555,000; profit CAD 45,000; margin 7.5%.
4. Set wheat +20% and FX +10%. Materials rise 32%, not 30%, to CAD 534,600. Total cost is CAD 684,600; profit is −CAD 84,600; margin is −14.1%. These are illustrative assumptions, not observed firm results.
5. Inspect the sensitivity grid. Discuss fixed revenue, freight and volume assumptions; no demand response, hedging, tax or financing is modelled.
6. Open Forecast evidence. Explain why validation selected persistence and why a later small XGBoost advantage is insufficient to claim superiority.
7. Open ablations. FX and other price context are tested by adding feature groups to the same XGBoost setup. Target-only already includes past FX effects embedded in CAD wheat cost; it is not a pure USD commodity model.
8. Download predictions and scenario grids. Show the test suite and report.

## Research defence prompts

- Why chronological splits? Future information must not predict the past.
- Why t−2? Monthly prices for t−1 may not be released at the start of t.
- Why no class balancing? This is a continuous price regression problem.
- Does an imposed scenario have measurable forecast accuracy? No; only historical model predictions can be scored. Scenario arithmetic can be verified, while empirical business accuracy needs invoices and realised margins.
- What does the Food Science graduate contribute? Ingredient specifications, processing yields, storage losses, supply-chain constraints and validation of operational assumptions.
- What next? Longer FX history with harmonisation, release-vintage data, multi-commodity/multi-country evaluation, and prospective firm-level validation.
