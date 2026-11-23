import pytest
import gameplay.authority_move_reconcile as mod

CASES = [
    ("frame_ack", (1)),
    ("error_smooth", (1)),
    ("shared_collision", (1)),
    ("dash_validate", (1)),
    ("rtt_smooth", (1)),
    ("rollback_present", (1)),
    ("predict_step", (1)),
    ("validation", (1)),
    ("audit", (1)),
    ("fallback", (1)),
    ("snap_threshold", (1)),
    ("metrics", (1)),
    ("lag_comp", (1)),
    ("authority_state", (1)),
    ("determinism", (1)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
