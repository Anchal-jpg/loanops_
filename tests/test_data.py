from src.validate_data import validate

def test_schema():
    assert len(validate()) > 0
