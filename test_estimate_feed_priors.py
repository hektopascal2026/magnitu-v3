"""WP4: Dirichlet blend of census labels with empirical labeled priors."""
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import estimate_feed_priors as efp  # noqa: E402


def test_census_tag_detects_prefix():
    assert efp.is_census_reasoning("[census]")
    assert efp.is_census_reasoning("[census] nzz packaging")
    assert not efp.is_census_reasoning("human note")
    assert not efp.is_census_reasoning("")


def test_blend_pulls_toward_census_when_n_grows():
    labeled = ["noise"] * 80 + ["background"] * 16 + ["important"] * 4
    census = ["investigation_lead"] * 10 + ["noise"] * 10
    pi, n, hat = efp.blend_feed_priors(census, labeled, alpha=20.0)
    assert n == 20
    assert hat["noise"] > 0.7
    assert pi["investigation_lead"] > hat["investigation_lead"]
    assert abs(sum(pi.values()) - 1.0) < 1e-9
    assert efp.CENSUS_TAG == "[census]"


def test_empty_census_equals_labeled_empirical():
    labeled = ["noise"] * 3 + ["important"] * 1
    pi, n, hat = efp.blend_feed_priors([], labeled, alpha=20.0)
    assert n == 0
    for c in hat:
        assert abs(pi[c] - hat[c]) < 1e-9
    assert abs(sum(efp.labeled_empirical(labeled).values()) - 1.0) < 1e-9
