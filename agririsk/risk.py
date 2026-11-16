"""Transparent accounting scenarios; shocks are assumptions, not probabilities."""
import math

def scenario(wheat_usd_t, cad_per_usd, tonnes=1000., revenue=600000.,
             other_cost=120000., freight_cad_t=30., wheat_shock=0., fx_shock=0.):
    vals = [wheat_usd_t, cad_per_usd, tonnes, revenue, other_cost, freight_cad_t, wheat_shock, fx_shock]
    if not all(math.isfinite(v) for v in vals): raise ValueError('Inputs must be finite')
    if min(wheat_usd_t, cad_per_usd, revenue) <= 0: raise ValueError('Prices, FX and revenue must be positive')
    if min(tonnes, other_cost, freight_cad_t) < 0 or min(wheat_shock, fx_shock) <= -1:
        raise ValueError('Invalid quantity, costs or shock')
    material = tonnes*wheat_usd_t*cad_per_usd*(1+wheat_shock)*(1+fx_shock)
    total = material + tonnes*freight_cad_t + other_cost
    return {'material_cost_cad': material, 'total_cost_cad': total,
            'profit_cad': revenue-total, 'margin_pct': 100*(revenue-total)/revenue}
