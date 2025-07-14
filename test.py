import pandas as pd
import pytest
from modif_data import generate_positions, generate_history, generate_tonnage, generate_fill_levels

def test_generate_positions_length_and_columns():
    df = generate_positions(10)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 10
    assert set(df.columns) == {"sensor_id", "lat", "lon", "capacity_tons"}

def test_generate_history_timestamps_and_values():
    df_pos = generate_positions(3)
    df_hist = generate_history(df_pos, days=2)
    # on doit avoir 3 capteurs × 12 points par jour × 2 jours = 72 lignes
    assert df_hist.shape[0] == 3 * (24//2) * 2
    assert df_hist["fill_level"].between(0, 1).all()

def test_generate_tonnage_totals_to_city_total():
    df_ton = generate_tonnage(5)
    total = df_ton["annual_tons"].sum()
    # tolérance à 1% d'arrondi
    assert abs(total - 146_949_553.36) / 146_949_553.36 < 0.01

def test_generate_fill_levels_range_and_length():
    levels = generate_fill_levels(20)
    assert isinstance(levels, list)
    assert len(levels) == 20
    assert all(15 <= lvl <= 100 for lvl in levels)
