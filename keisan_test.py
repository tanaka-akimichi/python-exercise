import pytest

import keisan


def test_add_number_plus_double():
    cal = keisan.Keisan()
    assert cal.add_number_plus_double(20, 10) == 60


def test_add_number_plus_double2():
    cal = keisan.Keisan()
    with pytest.raises(ValueError):
        cal.add_number_plus_double('a', 10)


