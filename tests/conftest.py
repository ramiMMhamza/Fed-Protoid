import pytest

from cpu_round import make_market


@pytest.fixture(scope="session")
def market_root(tmp_path_factory):
    """A synthetic Market-1501 tree: 12 identities x 6 cameras x 6 images."""
    return make_market(tmp_path_factory.mktemp("data"))

