import pytest
import os
import tempfile
from click.testing import CliRunner
import cli as cli

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture
def temp_file():
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("""3
0.5 0.5 0.0
0.0 0.5 0.5
0.0 0.0 1.0
0 0 1
a b c""")
    yield f.name
    os.unlink(f.name)

def test_parse_txt(runner, temp_file):
    """Test parsing a text file."""
    result = runner.invoke(cli, ['parse', temp_file, '--format', 'txt'])
    assert result.exit_code == 0
    assert "Successfully parsed" in result.output

def test_parse_prism(runner):
    """Test parsing a PRISM file."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("""[0] -> 0.5 : (state' = 1) + 0.5 : (state' = 2);
[1] -> 0.5 : (state' = 1) + 0.5 : (state' = 2);
[2] [term];""")
    
    try:
        result = runner.invoke(cli, ['parse', f.name, '--format', 'prism'])
        assert result.exit_code == 0
        assert "Successfully parsed" in result.output
    finally:
        os.unlink(f.name)

def test_parse_json(runner):
    """Test parsing a JSON file."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("""{
    "states": 3,
    "transitions": [
        {"from": 0, "to": 1, "prob": 0.5, "label": "a"},
        {"from": 0, "to": 2, "prob": 0.5, "label": "b"},
        {"from": 1, "to": 1, "prob": 0.5, "label": "c"},
        {"from": 1, "to": 2, "prob": 0.5, "label": "d"}
    ],
    "terminating": [2]
}""")
    
    try:
        result = runner.invoke(cli, ['parse', f.name, '--format', 'json'])
        assert result.exit_code == 0
        assert "Successfully parsed" in result.output
    finally:
        os.unlink(f.name)

def test_bisim(runner, temp_file):
    """Test bisimulation minimization."""
    result = runner.invoke(cli, ['bisim', temp_file, '--format', 'txt'])
    assert result.exit_code == 0
    assert "Compression ratio" in result.output
    assert "Equivalence classes" in result.output

def test_dist(runner, temp_file):
    """Test distance computation."""
    result = runner.invoke(cli, ['dist', temp_file, '--format', 'txt'])
    assert result.exit_code == 0
    assert "Distance matrix" in result.output
    assert "Most similar pairs" in result.output
    assert "Most different pairs" in result.output

def test_explain(runner, temp_file):
    """Test state difference explanation."""
    result = runner.invoke(cli, ['explain', temp_file, '--format', 'txt', '--states', '0', '1'])
    assert result.exit_code == 0
    assert "Distance" in result.output
    assert "Differences" in result.output

def test_classof(runner, temp_file):
    """Test equivalence class query."""
    result = runner.invoke(cli, ['classof', temp_file, '--format', 'txt', '--state', '0'])
    assert result.exit_code == 0
    assert "Equivalence class" in result.output

def test_classes(runner, temp_file):
    """Test listing all equivalence classes."""
    result = runner.invoke(cli, ['classes', temp_file, '--format', 'txt'])
    assert result.exit_code == 0
    assert "Equivalence classes" in result.output

def test_simulate(runner, temp_file):
    """Test simulation."""
    result = runner.invoke(cli, [
        'simulate',
        temp_file,
        '--format', 'txt',
        '--start', '0',
        '--runs', '10',
        '--max-steps', '5'
    ])
    assert result.exit_code == 0
    assert "Simulation results" in result.output
    assert "Average steps" in result.output
    assert "Termination rate" in result.output

def test_compare_sim(runner, temp_file):
    """Test comparative simulation."""
    result = runner.invoke(cli, [
        'compare-sim',
        temp_file,
        '--format', 'txt',
        '--states', '0', '1',
        '--runs', '10',
        '--max-steps', '5'
    ])
    assert result.exit_code == 0
    assert "Comparison results" in result.output
    assert "State 0" in result.output
    assert "State 1" in result.output

def test_manual(runner):
    """Test manual PTS entry."""
    result = runner.invoke(cli, ['manual'], input='2\n0.5 0.5\n0.5 0.5\n0 0\n')
    assert result.exit_code == 0
    assert "Successfully created" in result.output

def test_invalid_file(runner):
    """Test handling of invalid file."""
    result = runner.invoke(cli, ['parse', 'nonexistent.txt', '--format', 'txt'])
    assert result.exit_code != 0
    assert "Error" in result.output

def test_invalid_format(runner, temp_file):
    """Test handling of invalid format."""
    result = runner.invoke(cli, ['parse', temp_file, '--format', 'invalid'])
    assert result.exit_code != 0
    assert "Error" in result.output

def test_invalid_state(runner, temp_file):
    """Test handling of invalid state index."""
    result = runner.invoke(cli, ['explain', temp_file, '--format', 'txt', '--states', '0', '10'])
    assert result.exit_code != 0
    assert "Error" in result.output

def test_output_files(runner, temp_file):
    """Test that output files are created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = runner.invoke(cli, [
            'bisim',
            temp_file,
            '--format', 'txt',
            '--output-dir', tmpdir
        ])
        assert result.exit_code == 0
        assert os.path.exists(os.path.join(tmpdir, 'minimized_pts.png'))
        assert os.path.exists(os.path.join(tmpdir, 'minimized_pts.json')) 