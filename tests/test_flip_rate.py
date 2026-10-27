"""Toy 3-prompt x 4-candidate checks for the best-of-n metrics, with answers computed by hand."""
import numpy as np
import pytest

from src.eval.best_of_n import flip_rate, mean_kendall_tau, n_ties, winners

A = np.array([[0.1, 0.9, 0.3, 0.2],
              [2.0, 1.0, 0.0, -1.0],
              [0.0, 0.5, 1.5, 1.0]])


def test_identical_matrices():
    assert flip_rate(A, A.copy()) == 0.0
    tau, n_nan = mean_kendall_tau(A, A.copy())
    assert tau == pytest.approx(1.0) and n_nan == 0


def test_reversed_scores_give_tau_minus_one():
    tau, _ = mean_kendall_tau(A, -A)
    assert tau == pytest.approx(-1.0)


def test_one_differing_argmax_out_of_three():
    B = A.copy()
    B[2] = [3.0, 0.5, 1.5, 1.0]          # prompt 2 winner moves from #2 to #0; prompts 0 and 1 unchanged
    assert list(winners(B)) == [1, 0, 0]
    assert flip_rate(A, B) == pytest.approx(1 / 3)


def test_tie_goes_to_lowest_index_and_is_counted():
    T = np.array([[0.2, 0.7, 0.7, 0.1],   # tie between #1 and #2 -> #1
                  [1.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0]])  # all tied -> #0
    assert list(winners(T)) == [1, 0, 0]
    assert n_ties(T) == 2


def test_constant_row_tau_is_skipped_and_counted():
    C = A.copy()
    C[1] = 0.0                            # tau undefined on prompt 1
    tau, n_nan = mean_kendall_tau(A, C)
    assert n_nan == 1 and tau == pytest.approx(1.0)


def test_flip_gap_mean_is_cross_minus_within():
    from src.eval.best_of_n import flip_gap
    B = A.copy()
    B[2] = [3.0, 0.5, 1.5, 1.0]
    models = {"h0": {"scores": A}, "h1": {"scores": A.copy()}, "u0": {"scores": B}}
    groups = {"within_hh": [("h0", "h1")], "cross_hh_uf": [("h0", "u0"), ("h1", "u0")]}
    d = flip_gap(models, groups, ["cross_hh_uf"], ["within_hh"])
    assert list(d) == [0.0, 0.0, 1.0]                 # only prompt 2 flips, and only across configs
    assert d.mean() == pytest.approx(1 / 3 - 0.0)


def test_readme_block_is_replaced_from_file(tmp_path, monkeypatch):
    from src.eval.readme_tables import refresh
    monkeypatch.chdir(tmp_path)
    (tmp_path / "t.md").write_text("| a |\n|---|\n| 1 |\n")
    text = "intro\n<!-- BEGIN t.md -->\nstale\n<!-- END t.md -->\noutro\n"
    new, used = refresh(text)
    assert used == ["t.md"]
    assert new == "intro\n<!-- BEGIN t.md -->\n| a |\n|---|\n| 1 |\n<!-- END t.md -->\noutro\n"
