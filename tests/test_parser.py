"""Event-parser unit tests on hand-written Retrosheet fragments."""
from wp import retro_parse as R


def run(events, bases=(None, None, None, None), outs=0):
    st = R.GameState(bases=list(bases), outs=outs)
    for e in events:
        R.apply_play(st, e, "bat")
    return st


def test_single_and_advances():
    st = run(["S8/G.2-H;1-3"], bases=(None, "a", "b", None))
    assert st.score[0] == 1 and st.base_bits() == 0b101 and st.outs == 0


def test_force_out_batter_safe():
    st = run(["54(1)/FO/G5"], bases=(None, "a", None, None))
    assert st.outs == 1 and st.base_bits() == 0b001


def test_double_play():
    st = run(["64(1)3/GDP"], bases=(None, "a", None, None))
    assert st.outs == 2 and st.base_bits() == 0


def test_walk_forces_runners():
    st = run(["W"], bases=(None, "a", "b", "c"))
    assert st.score[0] == 1 and st.base_bits() == 0b111


def test_strikeout_with_steal_and_error_negating_out():
    st = run(["K+SB2"], bases=(None, "a", None, None))
    assert st.outs == 1 and st.base_bits() == 0b010
    st = run(["CS2(2E4).1-3"], bases=(None, "a", None, None))
    assert st.outs == 0 and st.base_bits() == 0b100
    # a clean putout group alongside an error group: runner is out
    st = run(["S8.BXH(E8)(82)"])
    assert st.outs == 1 and st.score[0] == 0


def test_dropped_third_strike_and_home_run():
    st = run(["K23+WP.B-1"])
    assert st.outs == 0 and st.base_bits() == 1
    st = run(["HR/F7.2-H;1-H"], bases=(None, "a", "b", None))
    assert st.score[0] == 3 and st.base_bits() == 0


def test_pa_start_states_and_radj():
    lines = [
        "info,tiebreaker,2",
        "play,1,0,a,00,,S8",
        "play,1,0,b,00,,SB2",       # non-batting: same PA continues
        "play,1,0,b,00,,K",
        "play,1,0,c,00,,8/F",
        "play,1,0,d,00,,63/G",
        "play,1,1,e,00,,HR",
        "play,1,1,f,00,,K",
        "play,1,1,g,00,,K",
        "play,1,1,h,00,,K",
        "radj,z,2",
        "play,2,0,a,00,,K",
    ]
    g = R.parse_game(lines, "TST202501010")
    assert g.error is None
    # (inning, half, outs, bases, vis, home)
    assert g.states[:4] == [(1, 0, 0, 0, 0, 0), (1, 0, 0, 1, 0, 0), (1, 0, 1, 2, 0, 0), (1, 0, 2, 2, 0, 0)]
    assert g.states[-1] == (2, 0, 0, 2, 0, 1)   # tie-breaker runner placed on 2B
    assert g.final == (0, 1)


def test_error_group_with_clean_putout_is_out():
    st = run(["S1.BX3(35)(E1/TH1)"])
    assert st.outs == 1 and st.base_bits() == 0
    st = run(["OA.1X3(E5)"], bases=(None, "a", None, None))
    assert st.outs == 0 and st.base_bits() == 0b100
