from config.schema import Stock
from data_ingestion.news import build_aliases, match_headlines_to_symbols, normalize_name


def test_normalize_name_strips_suffixes():
    assert normalize_name("Reliance Industries Ltd") == "reliance"
    assert normalize_name("Larsen & Toubro") == "larsen toubro"


def test_matches_full_name_short_name_and_symbol():
    stocks = [
        Stock("RELIANCE", "Oil & Gas", name="Reliance Industries"),
        Stock("TCS", "IT", name="Tata Consultancy Services"),
        Stock("LT", "Infrastructure", name="Larsen & Toubro"),
        Stock("M&M", "Auto", name="Mahindra & Mahindra"),
    ]
    heads = [
        {"title": "Reliance Industries posts record profit", "summary": ""},
        {"title": "Markets gain", "summary": "TCS leads the rally"},
        {"title": "L&T wins order", "summary": "Larsen & Toubro bags Rs 5000 cr contract"},
        {"title": "Mahindra & Mahindra unveils SUV", "summary": ""},
    ]
    m = match_headlines_to_symbols(heads, stocks)
    assert {"RELIANCE", "TCS", "LT", "M&M"} <= set(m)


def test_two_letter_symbol_not_falsely_matched():
    stocks = [Stock("LT", "Infrastructure", name="Larsen & Toubro")]
    heads = [{"title": "It felt lt random", "summary": "nothing relevant"}]
    # 'LT' (2 chars) should not be used as a bare-symbol alias
    assert "lt" not in {a for a in build_aliases(stocks[0]) if a == "lt"}
