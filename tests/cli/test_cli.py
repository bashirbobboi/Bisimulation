import json
import numpy as np
import pytest
from typer.testing import CliRunner
from pathlib import Path

from probisim.cli import app  

runner = CliRunner()

def make_prism_model(tmp_path):
    pm = tmp_path / "model.pm"
    pm.write_text(
        """
        [] s=0 -> 0.5 : (s'=0) + 0.5 : (s'=1);
        [] s=1 -> 1.0 : (s'=1);
        """
    )
    return str(pm)


def test_parse_writes_internal_json(tmp_path):
    model_file = make_prism_model(tmp_path)
    internal = tmp_path / "internal.json"

    result = runner.invoke(app, ["parse", model_file, "prism", "--to", str(internal)])
    assert result.exit_code == 0
    assert "Parsed" in result.stdout

    data = json.loads(internal.read_text())
    # matrix row‐sums, termination flags, and empty labels
    assert data["T"] == [[0.5, 0.5], [0.0, 1.0]]
    assert data["Term"] == [0, 1]
    assert data["labels"] == {}


def test_bisim_and_file_outputs(tmp_path, monkeypatch):
    # Prepare internal JSON
    internal = tmp_path / "int.json"
    internal.write_text(json.dumps({
        "T": [[0.5, 0.5], [0.0, 1.0]],
        "Term": [0, 1],
        "labels": {}
    }))

    # Monkey‐patch Graphviz.Source.render to be a no‐op
    from pathlib import Path

    def fake_render(self, filename, cleanup):
        # Graphviz.render(filename, cleanup) writes out filename.png
        Path(f"{filename}.png").write_bytes(b"")  # create an empty PNG
        return str(Path(f"{filename}.png"))

    monkeypatch.setattr("graphviz.Source.render", fake_render)
    
    # Change into tmp_path so all outputs land there
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["bisim", str(internal)])
    assert result.exit_code == 0

    out = result.stdout
    assert "Number of Equivalence Classes" in out
    assert (tmp_path / "minimized_PTS.png").exists()
    assert (tmp_path / "original_PTS.png").exists()


def test_dist_and_heatmap(tmp_path, monkeypatch):
    internal = tmp_path / "int.json"
    internal.write_text(json.dumps({
        "T": [[1.0, 0.0], [0.0, 1.0]],
        "Term": [0, 0],
        "labels": {}
    }))

    # Change into tmp_path so all outputs land there
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["dist", str(internal)])
    assert result.exit_code == 0
    assert "Distance Matrix:" in result.stdout
    assert (tmp_path / "distance_heatmap.png").exists()


def test_explain_and_class_commands(tmp_path, monkeypatch):
    internal = tmp_path / "int.json"
    internal.write_text(json.dumps({
        "T": [[1.0, 0.0], [0.5, 0.5]],
        "Term": [0, 0],
        "labels": {}
    }))

    # Change into tmp_path so all outputs land there
    monkeypatch.chdir(tmp_path)

    # explain S1 vs S2
    res1 = runner.invoke(app, ["explain", str(internal), "1", "2"])
    assert res1.exit_code == 0
    assert "Distance between S1 and S2" in res1.stdout

    # classof S2
    res2 = runner.invoke(app, ["classof", str(internal), "2"])
    assert res2.exit_code == 0
    assert "equivalence class" in res2.stdout

    # classes
    res3 = runner.invoke(app, ["classes", str(internal)])
    assert res3.exit_code == 0
    assert "Class 1:" in res3.stdout


def test_manual_interactive(tmp_path, monkeypatch):
    out_file = tmp_path / "out.json"
    # simulate user entering: 2 states, non-terminating then terminating, transitions "2 1.0"
    user_input = "\n".join([
        "2",  # number of states
        "n",  # state1 non-terminating
        "y",  # state2 terminating
        "2 1.0",  # from state1: go to state2 with prob 1.0
        "",       # skip state2 transitions
    ]) + "\n"
    
    # Change into tmp_path so all outputs land there
    monkeypatch.chdir(tmp_path)
    
    result = runner.invoke(app, ["manual", "--to", str(out_file)], input=user_input)
    assert result.exit_code == 0
    data = json.loads(out_file.read_text())
    assert data["T"] == [[0.0,1.0],[0.0,0.0]]
    assert data["Term"] == [0,1]


def test_manual_error_parsing(tmp_path, monkeypatch):
    out_file = tmp_path / "err.json"
    # simulate invalid input causing parse error
    user_input = "\n".join([
        "2",  # number of states
        "n",  # state1 non-terminating
        "n",  # state2 non-terminating
        "invalid_input",  # bad format
    ]) + "\n"

    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["manual", "--to", str(out_file)], input=user_input)
    # Parser currently treats bad format as warning but continues, so exit 0
    assert result.exit_code == 0
    assert "Error parsing transitions for state 1:" in result.stdout


def test_simulate_and_compare(tmp_path, monkeypatch):
    internal = tmp_path / "int.json"
    internal.write_text(json.dumps({
        "T": [[0.0, 1.0], [1.0, 0.0]],
        "Term": [0, 1],
        "labels": {}
    }))

    # Monkey-patch np.random.choice to always return state 0
    monkeypatch.setattr(
        np.random,
        "choice",
        lambda a, p=None, *args, **kwargs: 0
    )
    
    # Change into tmp_path so all outputs land there
    monkeypatch.chdir(tmp_path)

    sim = runner.invoke(app, ["simulate", str(internal), "--start-state", "1", "--num-simulations", "5", "--max-steps", "10"])  
    assert sim.exit_code == 0
    assert "Average Steps to Termination:" in sim.stdout

    comp = runner.invoke(app, ["compare-sim", str(internal), "--state1", "1", "--state2", "2", "--num-runs", "5"])  
    assert comp.exit_code == 0
    assert "Comparative Simulation Results" in comp.stdout


def test_simulate_show_runs(tmp_path, monkeypatch):
    internal = tmp_path / "int.json"
    internal.write_text(json.dumps({
        "T": [[0.0, 1.0], [1.0, 0.0]],
        "Term": [0, 1],
        "labels": {}
    }))

    # Monkey-patch np.random.choice to cycle through states predictably
    seq = [1, 0]
    idx = {'i': 0}
    def fake_choice(a, p=None, *args, **kwargs):
        val = seq[idx['i'] % len(seq)]
        idx['i'] += 1
        return val
    monkeypatch.setattr(np.random, "choice", fake_choice)
    
    # Change into tmp_path so all outputs land there
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["simulate", str(internal), "--start-state", "1",
                                 "--num-simulations", "3", "--max-steps", "2", "--show-runs"])
    assert result.exit_code == 0
    assert "Sample Run Sequences" in result.stdout
    assert "Run 1:" in result.stdout
    assert "S1 →" in result.stdout


def test_compare_sim_show_runs(tmp_path, monkeypatch):
    internal = tmp_path / "int.json"
    # simple two-state cyclic model
    internal.write_text(json.dumps({
        "T": [[0.0, 1.0], [1.0, 0.0]],
        "Term": [0, 1],
        "labels": {}
    }))

    # Monkey-patch np.random.choice so runs1 and runs2 are deterministic
    seq = [1, 1, 1]  # Always go to state 1
    idx = {'i': 0}
    def fake_choice(a, p=None, *args, **kwargs):
        val = seq[idx['i'] % len(seq)]
        idx['i'] += 1
        return val
    monkeypatch.setattr(np.random, "choice", fake_choice)

    # Change into tmp_path
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["compare-sim", str(internal), "--state1", "1", "--state2", "2",
                                "--num-runs", "3", "--show-runs"])
    assert result.exit_code == 0
    assert "Comparative Simulation Results" in result.stdout
    assert "Sample Run Sequences" in result.stdout
    assert "State 1 Run 1:" in result.stdout
    assert "State 2 Run 1:" in result.stdout
    # Verify the deterministic runs
    assert "S1 → S2 (Terminated)" in result.stdout  # State 1 always goes to state 2
    assert "S2 (Terminated)" in result.stdout  # State 2 is terminating


