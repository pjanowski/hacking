import pytest
import e1

@pytest.mark.parametrize("a, t", [([3, 5, -8, -10, 10, 12, -2, -3], 22)])
def test_e1(a, t):
    assert e1.main1(a) == t

