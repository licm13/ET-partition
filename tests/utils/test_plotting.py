import sys
from pathlib import Path
import tempfile
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.plotting import save_figure, print_performance_summary


def test_save_figure_creates_file():
    """Test that save_figure creates a file."""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.png"
        save_figure(fig, output_path)
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0


def test_save_figure_creates_parent_dirs():
    """Test that save_figure creates parent directories."""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "subdir" / "nested" / "test.png"
        save_figure(fig, output_path)
        
        assert output_path.exists()
        assert output_path.parent.exists()


def test_print_performance_summary_runs():
    """Test that print_performance_summary runs without error."""
    data = {
        'method': ['Method_A', 'Method_B'],
        'rmse_T_mean': [1.234, 2.345],
        'rmse_T_std': [0.123, 0.234],
        'rmse_E_mean': [1.567, 2.678],
        'rmse_E_std': [0.156, 0.267],
        'nse_T_mean': [0.876, 0.765],
        'nse_T_std': [0.087, 0.076],
        'kge_T_mean': [0.912, 0.823],
        'kge_T_std': [0.091, 0.082],
        'correlation_T_mean': [0.945, 0.856],
        'correlation_T_std': [0.094, 0.085],
    }
    df = pd.DataFrame(data)
    
    # Should not raise an exception
    print_performance_summary(df)
