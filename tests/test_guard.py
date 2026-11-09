import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from screenguard import Guard, is_frontal


def run(g, seq, t0=1000.0, dt=0.25):
    out = []
    for i, (owner, stranger, anyf) in enumerate(seq):
        out.append(g.update(t0 + i * dt, owner, stranger, anyf))
    return out


def test_owner_with_others_stays_on():
    g = Guard(); g.last_owner = 1000.0
    assert not any(run(g, [(True, True, True)] * 40))


def test_stranger_alone_blanks_after_delay():
    g = Guard(stranger_delay=1.5, owner_grace=2.0); g.last_owner = 1000.0
    res = run(g, [(False, True, True)] * 30)
    assert not res[0] and res[-1]
    first = res.index(True) * 0.25
    assert 2.0 <= first <= 4.0  # grace + delay


def test_owner_return_unblanks_after_confirm():
    g = Guard(); g.last_owner = 1000.0
    run(g, [(False, True, True)] * 30)
    assert g.blanked
    res = run(g, [(True, True, True)] * 3, t0=1100.0)
    assert res == [True, False, False]  # needs 2 consecutive owner frames


def test_absent_disabled_by_default():
    g = Guard(); g.last_owner = 1000.0
    assert not any(run(g, [(False, False, False)] * 200))


def test_absent_delay():
    g = Guard(absent_delay=5); g.last_owner = 1000.0
    assert run(g, [(False, False, False)] * 40)[-1]


def test_stranger_not_looking_ignored():
    g = Guard(); g.last_owner = 1000.0
    assert not any(run(g, [(False, False, True)] * 40))


def test_frontal():
    # x,y,w,h, right eye, left eye, nose, mouth r, mouth l, score
    front = np.array([0,0,100,100, 30,40, 70,40, 50,62, 35,80, 65,80, 0.9])
    side = np.array([0,0,100,100, 30,40, 70,40, 72,62, 35,80, 65,80, 0.9])
    assert is_frontal(front) and not is_frontal(side)
