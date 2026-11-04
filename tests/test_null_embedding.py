import argparse

import pytest

from audiocfg.null_embedding import data, sanity, train
from audiocfg.null_embedding.train import LogRow, null_output_path, select_best, write_log
from audiocfg.schema import RESULT_FILES


def test_select_excludes_T_and_is_deterministic():
    cand = [str(i) for i in range(1000)]
    excl = [str(i) for i in range(0, 1000, 2)]  # E2 set T
    a = data.select_clips(cand, excl, 100, seed=0)
    b = data.select_clips(list(reversed(cand)), set(excl), 100, seed=0)
    assert a == b and len(a) == 100
    assert not set(a) & set(excl)
    assert data.select_clips(cand, excl, 10_000, seed=0) == sorted(set(cand) - set(excl))


def test_split_deterministic_and_disjoint(tmp_path):
    ids = [f"fs_{i}" for i in range(500)]
    tr, va = data.train_val_split(ids, 0.1, seed=3)
    tr2, va2 = data.train_val_split(list(reversed(ids)), 0.1, seed=3)
    assert (tr, va) == (tr2, va2)
    assert len(va) == 50 and len(tr) == 450 and not set(tr) & set(va)
    assert data.train_val_split(ids, 0.1, seed=4)[1] != va
    data.write_split(tmp_path / "s.csv", tr, va)
    assert data.read_split(tmp_path / "s.csv") == (tr, va)


def test_read_id_list(tmp_path):
    (tmp_path / "a.csv").write_text("id,license\n12,cc0\n34,cc-by\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("1\n2\n\n", encoding="utf-8")
    assert data.read_id_list(tmp_path / "a.csv") == ["12", "34"]
    assert data.read_id_list(tmp_path / "b.txt") == ["1", "2"]


def _rows():
    return [LogRow("zeros", 0, float("nan"), 1.0, 1.0), LogRow("zeros", 100, 0.9, 0.80, 1.0),
            LogRow("zeros", 200, 0.8, 0.75, 1.0), LogRow("zeros", 300, 0.7, 0.75, 1.0),
            LogRow("zeros", 400, 0.6, 0.78, 1.0)]


def test_checkpoint_selection_min_val_loss_earliest_tie():
    assert select_best(_rows()).step == 200


def test_log_columns_and_sanity(tmp_path):
    path = tmp_path / "e3_null_training.csv"
    write_log(path, _rows())
    train.replace_init_in_log(path, "normal", [LogRow("normal", 0, 0.0, 1.2, 1.0)])
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert tuple(header) == RESULT_FILES["e3_null_training.csv"]
    checks = sanity.check_val_loss(sanity.read_log(path))
    assert checks["zeros"].passed and checks["zeros"].best_step == 200
    assert not checks["normal"].passed


def test_null_output_path():
    assert null_output_path("outputs", "zeros").as_posix() == "outputs/null/sao_learned_zeros.pt"


def test_cli_args():
    p = argparse.ArgumentParser()
    train.add_args(p)
    a = p.parse_args(["--init", "zeros", "--eval-every", "50"])
    assert a.init == "zeros" and a.eval_every == 50


def test_train_null_tiny_frozen_model(tmp_path):
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    L, d = 4, 8
    model = torch.nn.Linear(d, d)
    target_null = torch.randn(L, d)

    def loss_fn(batch, cond):  # stands in for SAO's v-loss; optimum at cond == target_null
        noise = torch.randn_like(batch) * 0.01
        return ((model(cond) - model(target_null) + noise.mean()) ** 2).mean() + 0 * batch.sum()

    batches = iter(lambda: torch.randn(2, 3), None)
    res = train.train_null(loss_fn, model, (L, d), "zeros", batches, [torch.randn(2, 3)],
                           lr=0.05, max_steps=200, eval_every=20)
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    assert res["val_loss"] < res["rows"][0].val_loss_zero_null
    assert res["step"] == select_best(res["rows"]).step
    out = train.save_null(null_output_path(tmp_path, "zeros"), res)
    saved = torch.load(out)
    assert set(saved) >= {"null", "shape", "val_loss", "step"}
    assert tuple(saved["null"].shape) == (L, d)
