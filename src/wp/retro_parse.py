"""Pure-Python parser for Retrosheet event files (.EVA / .EVN).

Why not Chadwick: the build environment could not compile/run third-party C
code, so this module re-implements the subset of Chadwick's game-state logic
that win-probability work needs: inning, half, outs, base occupancy and score
*before every plate appearance*, plus the final score of every game.

Correctness is checked empirically: every parsed game's final score is
compared with the Retrosheet game log (see ``wp.games``); games that do not
match are excluded and counted.

Event-file grammar reference: Retrosheet "The Event File" specification,
https://www.retrosheet.org/eventfile.htm  (sections "play", "radj", "info").
Key rules implemented here (all from that document):
  * play,<inning>,<0=visitor bats|1=home bats>,<batter>,<count>,<pitches>,<event>
  * event = basic[/modifiers...][.advances]; advances separated by ';'
    ``r-d`` = runner r reaches d,  ``rXd`` = runner r thrown out going to d,
    unless a parenthesised fielding group contains an error ``E`` (then safe).
  * Batter advances implied by the basic play (S=1B, D/DGR=2B, T=3B, H/HR=home,
    W/IW/I/HP/C/E/FC = 1B, K and fielded outs = out) unless overridden by an
    explicit ``B-x`` / ``BXx``.
  * Fielded out ``54(1)``: the parenthesised runner is out; if the last
    fielding group names a runner and no batter out is given, the batter is
    safe at first (force play).
  * ``radj,<player>,<base>``: runner placed on base (2020+ extra innings rule).
  * Characters ``#`` (uncertain), ``!`` (exceptional) and ``?`` are annotations.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

BASE_NUM = {"B": 0, "1": 1, "2": 2, "3": 3, "H": 4}
NON_BATTING = ("SB", "CS", "POCS", "PO", "WP", "PB", "BK", "OA", "DI", "NP", "FLE", "OBS")


class ParseError(Exception):
    pass


def split_top(s: str, sep: str) -> list[str]:
    """Split on ``sep`` outside parentheses."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return out


_ADV_RE = re.compile(r"^([B123])([-X])([123H])((?:\([^)]*\))*)")
_PAREN_RE = re.compile(r"\(([^)]*)\)")
_FIELD_RE = re.compile(r"^\d*(E\d)?\d*(/[A-Z]+\d*)*$")


def _error_negates(paren_text: str) -> bool:
    """True if the *last* fielding group in ``paren_text`` contains an error."""
    groups = [g for g in _PAREN_RE.findall(paren_text) if g and _FIELD_RE.match(g)]
    return bool(groups) and "E" in groups[-1]


def parse_advances(adv: str) -> dict[int, int | None]:
    """Return {runner_base: dest} with dest None meaning out (0=batter, 4=home)."""
    res: dict[int, int | None] = {}
    if not adv:
        return res
    for a in adv.split(";"):
        a = a.strip()
        if not a:
            continue
        m = _ADV_RE.match(a)
        if not m:
            raise ParseError(f"bad advance {a!r}")
        r, kind, d, parens = m.groups()
        dest = BASE_NUM[d]
        if kind == "X":
            # An error in the fielding sequence negates the out, unless a later
            # fielding sequence records the out anyway, e.g. 1XH(E8)(82).
            safe = _error_negates(parens)
            res[BASE_NUM[r]] = dest if safe else None
        else:
            res[BASE_NUM[r]] = dest
    return res


_OUT_GROUP_RE = re.compile(r"(\d+)(?:\(([B123])\))?")


def parse_fielded_out(ev: str) -> tuple[bool, set[int], bool]:
    """Parse a fielded out such as ``64(1)3``.

    Returns (batter_out, runners_out, batter_safe_on_error).
    """
    if "E" in ev:  # e.g. "6E3": throwing error on a play at first -> batter safe
        return False, set(), True
    pos = 0
    runners_out: set[int] = set()
    batter_out = False
    last_had_runner = False
    while pos < len(ev):
        m = _OUT_GROUP_RE.match(ev, pos)
        if not m:
            raise ParseError(f"bad out {ev!r}")
        _, r = m.groups()
        if r is None:
            batter_out = True
            last_had_runner = False
        elif r == "B":
            batter_out = True
            last_had_runner = True
        else:
            runners_out.add(int(r))
            last_had_runner = True
        pos = m.end()
    if not batter_out and not last_had_runner:
        batter_out = True
    return batter_out, runners_out, False


def _non_batting(tok: str, implied: dict[int, int | None]) -> None:
    """Apply a non-batting event (SB2, CS3(25), PO1(13), POCS2(..), WP, ...)."""
    for t in tok.split(";"):
        t = t.strip()
        if not t:
            continue
        if t.startswith("SB"):
            d = BASE_NUM[t[2]]
            implied[d - 1] = d
        elif t.startswith("POCS") or t.startswith("CS"):
            k = 4 if t.startswith("POCS") else 2
            d = BASE_NUM[t[k]]
            implied[d - 1] = d if _error_negates(t) else None
        elif t.startswith("PO"):
            r = BASE_NUM[t[2]]
            if not _error_negates(t):
                implied[r] = None
        elif t.startswith(("WP", "PB", "BK", "OA", "DI", "NP", "FLE", "OBS")):
            pass
        elif t.startswith("E") and t[1:2].isdigit():
            pass  # error on a dropped third strike etc. (K+E2); advances are explicit
        else:
            raise ParseError(f"unknown non-batting event {t!r}")


def classify(event: str) -> tuple[bool, dict[int, int | None]]:
    """Return (is_batting_event, implied moves) for the basic part of a play.

    implied maps base->dest (None = out); key 0 is the batter.
    """
    implied: dict[int, int | None] = {}
    main, *extras = split_top(event, "+")
    m = main
    batting = True
    if m.startswith(NON_BATTING):
        batting = False
        _non_batting(m, implied)
    elif m.startswith("HP"):
        implied[0] = 1
    elif m.startswith("HR") or (m.startswith("H") and not m.startswith("HP")):
        implied[0] = 4
    elif m.startswith("FC"):
        implied[0] = 1
    elif m.startswith("IW") or m.startswith("I") or m.startswith("W"):
        implied[0] = 1
    elif m.startswith("K"):
        implied[0] = None
    elif m.startswith("C"):
        implied[0] = 1
    elif m.startswith("E"):
        implied[0] = 1
    elif m.startswith("S"):
        implied[0] = 1
    elif m.startswith("DGR") or m.startswith("D"):
        implied[0] = 2
    elif m.startswith("T"):
        implied[0] = 3
    elif m[:1].isdigit():
        bout, rout, err = parse_fielded_out(m)
        implied[0] = 1 if (err or not bout) else None
        for r in rout:
            implied[r] = None
    else:
        raise ParseError(f"unknown event {main!r}")
    for x in extras:
        _non_batting(x, implied)
    return batting, implied


@dataclass
class GameState:
    inning: int = 1
    half: int = 0  # 0 top (visitor bats), 1 bottom
    outs: int = 0
    bases: list[str | None] = field(default_factory=lambda: [None, None, None, None])  # idx 1..3
    score: list[int] = field(default_factory=lambda: [0, 0])  # [visitor, home]

    def base_bits(self) -> int:
        return sum(1 << (b - 1) for b in (1, 2, 3) if self.bases[b] is not None)


def apply_play(st: GameState, event: str, batter: str) -> bool:
    """Mutate state for one play string; return True if it was a batting event."""
    ev = event.replace("#", "").replace("!", "").replace("?", "").strip()
    basic, _, adv = ev.partition(".")
    first = split_top(basic, "/")[0]
    batting, implied = classify(first)
    explicit = parse_advances(adv)

    # final fate of each occupant: base -> dest (None = out, absent = stays)
    fate: dict[int, int | None] = {}
    fate.update(implied)
    fate.update(explicit)
    if not batting:
        fate.pop(0, None)
    # runners must exist (ignore stale implied moves for empty bases)
    for b in (1, 2, 3):
        if b in fate and st.bases[b] is None:
            if b in explicit:
                raise ParseError(f"advance from empty base {b} in {event!r}")
            fate.pop(b)
    # forced advances when the batter takes first
    if batting and fate.get(0) == 1:
        need = 1
        while need <= 3 and st.bases[need] is not None and need not in fate:
            fate[need] = need + 1
            need += 1
    new = [None, None, None, None]
    stayers: list[tuple[int, str]] = []
    for b in (3, 2, 1, 0):
        if b == 0:
            if not batting:
                continue
            occ = batter
        else:
            occ = st.bases[b]
            if occ is None:
                continue
        if b != 0 and b not in fate:
            stayers.append((b, occ))
            continue
        d = fate[b]
        if d is None:
            st.outs += 1
        elif d == 4:
            st.score[st.half] += 1
        else:
            if d < b and b != 0:
                raise ParseError(f"runner moved backward in {event!r}")
            if new[d] is not None:
                raise ParseError(f"two runners on base {d} in {event!r}")
            new[d] = occ
    # Runners not mentioned stay put; if their base was taken (an unrecorded
    # forced advance, seen in some early-era files) push them forward.
    for b, occ in sorted(stayers, reverse=True):
        d = b
        while d <= 3 and new[d] is not None:
            d += 1
        if d == 4:
            st.score[st.half] += 1
        else:
            new[d] = occ
    st.bases = new
    return batting


@dataclass
class ParsedGame:
    game_id: str
    info: dict
    states: list[tuple]  # (inning, half, outs, bases, vis, home)
    final: tuple[int, int]
    last_inning: int
    last_half: int
    last_outs: int
    error: str | None = None


def parse_game(lines: list[str], game_id: str) -> ParsedGame:
    info: dict[str, str] = {}
    st = GameState()
    states: list[tuple] = []
    pa_open = False
    cur_half = (1, 0)
    started = False
    err = None
    pending_radj: list[tuple[str, int]] = []
    for line in lines:
        rec, _, rest = line.partition(",")
        if rec == "info":
            k, _, v = rest.partition(",")
            info[k] = v.strip().strip('"')
        elif rec == "radj":
            p = rest.split(",")
            pending_radj.append((p[0], int(p[1])))
        elif rec == "play":
            if err:
                continue
            p = rest.split(",")
            inn, team, batter, event = int(p[0]), int(p[1]), p[2], p[5]
            if (inn, team) != cur_half or not started:
                if started and (inn, team) < cur_half:
                    err = f"half-inning went backwards at {inn},{team}"
                    continue
                cur_half = (inn, team)
                st.inning, st.half, st.outs = inn, team, 0
                st.bases = [None, None, None, None]
                pa_open = False
                started = True
            for pid, base in pending_radj:
                st.bases[base] = pid
            pending_radj = []
            ev = event.strip()
            if ev.startswith("NP") and "." not in ev:
                continue
            if st.outs >= 3:
                err = f"play after third out [{inn},{team},{event}]"
                continue
            if not pa_open:
                states.append((st.inning, st.half, st.outs, st.base_bits(), st.score[0], st.score[1]))
                pa_open = True
            try:
                if apply_play(st, ev, batter):
                    pa_open = False
            except (ParseError, KeyError, IndexError, ValueError) as e:
                err = f"{e} [{inn},{team},{event}]"
            if st.outs >= 3:
                pa_open = False
                if st.outs > 3:
                    err = f"more than 3 outs [{inn},{team},{event}]"
    return ParsedGame(game_id, info, states, (st.score[0], st.score[1]),
                      st.inning, st.half, st.outs, err)


def iter_games(path: Path):
    gid, buf = None, []
    with open(path, encoding="latin1") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line.startswith("id,"):
                if gid is not None:
                    yield gid, buf
                gid, buf = line[3:].strip(), []
            elif gid is not None:
                buf.append(line)
    if gid is not None:
        yield gid, buf


def parse_file(path: Path) -> list[ParsedGame]:
    return [parse_game(lines, gid) for gid, lines in iter_games(path)]
