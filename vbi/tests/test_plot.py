import unittest
import numpy as np
import pytest
from parameterized import parameterized
import sys
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Add the parent directory to the path to import vbi
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from vbi.plot import (
    _ensure_numpy_no_torch,
    _convert_to_list_of_numpy_no_torch,
    _handle_nan_infs_no_torch,
    _infer_limits_no_torch,
    _prepare_for_plot_no_torch,
    _to_list_string,
    _to_list_kwargs,
    _update_dict,
    _get_default_fig_kwargs_no_torch,
    _get_default_diag_kwargs_no_torch,
    _get_default_offdiag_kwargs_no_torch,
    _hist_1d,
    _kde_1d,
    _scatter_1d,
    _hist_2d,
    _kde_2d,
    _contour_2d,
    _scatter_2d,
    _plot_2d,
    _format_axis,
    plot_1d_distribution,
    plot_2d_distribution,
    pairplot_numpy,
)


@pytest.mark.short
@pytest.mark.fast
class TestUtilityFunctions(unittest.TestCase):
    """Test suite for utility functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.seed = 42
        np.random.seed(self.seed)

    def test_ensure_numpy_no_torch(self):
        """Test _ensure_numpy_no_torch with different inputs."""
        # Test with list
        arr = [1, 2, 3, 4]
        result = _ensure_numpy_no_torch(arr)
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1, 2, 3, 4]))
        
        # Test with numpy array
        arr = np.array([1, 2, 3, 4])
        result = _ensure_numpy_no_torch(arr)
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_convert_to_list_of_numpy_no_torch(self):
        """Test _convert_to_list_of_numpy_no_torch with different inputs."""
        # Test with single array
        arr = np.array([[1, 2], [3, 4]])
        result = _convert_to_list_of_numpy_no_torch(arr)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result[0], arr)
        
        # Test with list of arrays
        arr1 = np.array([[1, 2], [3, 4]])
        arr2 = np.array([[5, 6], [7, 8]])
        result = _convert_to_list_of_numpy_no_torch([arr1, arr2])
        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(result[0], arr1)
        np.testing.assert_array_equal(result[1], arr2)

    def test_handle_nan_infs_no_torch(self):
        """Test _handle_nan_infs_no_torch removes NaNs and Infs."""
        # Test with NaNs
        arr = np.array([[1, 2], [3, np.nan], [5, 6]])
        result = _handle_nan_infs_no_torch([arr])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape[0], 2)  # One row removed
        
        # Test with Infs
        arr = np.array([[1, 2], [np.inf, 4], [5, 6]])
        result = _handle_nan_infs_no_torch([arr])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape[0], 2)  # One row removed

    def test_infer_limits_no_torch(self):
        """Test _infer_limits_no_torch infers correct limits."""
        samples = [np.array([[1, 2], [3, 4], [5, 6]])]
        limits = _infer_limits_no_torch(samples, dim=2)
        
        self.assertEqual(len(limits), 2)
        # Check that limits are extended beyond min/max
        self.assertLess(limits[0][0], 1)
        self.assertGreater(limits[0][1], 5)
        self.assertLess(limits[1][0], 2)
        self.assertGreater(limits[1][1], 6)

    def test_infer_limits_with_points(self):
        """Test _infer_limits_no_torch includes points in limits."""
        samples = [np.array([[1, 2], [3, 4]])]
        points = [np.array([[0, 1], [6, 7]])]
        limits = _infer_limits_no_torch(samples, dim=2, points=points)
        
        # Limits should include the points
        self.assertLessEqual(limits[0][0], 0)
        self.assertGreaterEqual(limits[0][1], 6)
        self.assertLessEqual(limits[1][0], 1)
        self.assertGreaterEqual(limits[1][1], 7)

    def test_prepare_for_plot_no_torch(self):
        """Test _prepare_for_plot_no_torch prepares data correctly."""
        samples = np.random.randn(100, 3)
        samples_list, dim, limits, points = _prepare_for_plot_no_torch(samples)
        
        self.assertIsInstance(samples_list, list)
        self.assertEqual(len(samples_list), 1)
        self.assertEqual(dim, 3)
        self.assertEqual(limits.shape, (3, 2))
        self.assertEqual(len(points), 0)

    def test_to_list_string(self):
        """Test _to_list_string with different inputs."""
        # Single string
        result = _to_list_string("hist", 3)
        self.assertEqual(result, ["hist", "hist", "hist"])
        
        # Already a list
        result = _to_list_string(["hist", "kde", "scatter"], 3)
        self.assertEqual(result, ["hist", "kde", "scatter"])

    def test_to_list_kwargs(self):
        """Test _to_list_kwargs with different inputs."""
        # Single dict
        result = _to_list_kwargs({"color": "red"}, 2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], {"color": "red"})
        
        # Already a list
        result = _to_list_kwargs([{"color": "red"}, {"color": "blue"}], 2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["color"], "red")
        self.assertEqual(result[1]["color"], "blue")

    def test_update_dict(self):
        """Test _update_dict merges dictionaries correctly."""
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        update = {"b": {"c": 4}, "e": 5}
        result = _update_dict(base, update)
        
        self.assertEqual(result["a"], 1)
        self.assertEqual(result["b"]["c"], 4)
        self.assertEqual(result["b"]["d"], 3)
        self.assertEqual(result["e"], 5)

    def test_get_default_fig_kwargs(self):
        """Test _get_default_fig_kwargs_no_torch returns proper dict."""
        kwargs = _get_default_fig_kwargs_no_torch()
        
        self.assertIsInstance(kwargs, dict)
        self.assertIn("legend", kwargs)
        self.assertIn("samples_colors", kwargs)
        self.assertIn("points_colors", kwargs)

    def test_get_default_diag_kwargs(self):
        """Test _get_default_diag_kwargs_no_torch for different plot types."""
        # Test kde
        kwargs = _get_default_diag_kwargs_no_torch("kde", 0)
        self.assertIn("bw_method", kwargs)
        self.assertIn("mpl_kwargs", kwargs)
        
        # Test hist
        kwargs = _get_default_diag_kwargs_no_torch("hist", 0)
        self.assertIn("bin_heuristic", kwargs)
        self.assertIn("mpl_kwargs", kwargs)
        
        # Test scatter
        kwargs = _get_default_diag_kwargs_no_torch("scatter", 0)
        self.assertIn("mpl_kwargs", kwargs)

    def test_get_default_offdiag_kwargs(self):
        """Test _get_default_offdiag_kwargs_no_torch for different plot types."""
        # Test kde
        kwargs = _get_default_offdiag_kwargs_no_torch("kde", 0)
        self.assertIn("bw_method", kwargs)
        
        # Test scatter
        kwargs = _get_default_offdiag_kwargs_no_torch("scatter", 0)
        self.assertIn("mpl_kwargs", kwargs)
        
        # Test contour
        kwargs = _get_default_offdiag_kwargs_no_torch("contour", 0)
        self.assertIn("levels", kwargs)


@pytest.mark.short
@pytest.mark.fast
class TestPlottingPrimitives(unittest.TestCase):
    """Test suite for basic plotting functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.seed = 42
        np.random.seed(self.seed)
        self.samples = np.random.randn(100)
        self.limits = np.array([-3, 3])

    def tearDown(self):
        """Clean up after tests."""
        plt.close('all')

    def test_hist_1d(self):
        """Test _hist_1d creates histogram."""
        fig, ax = plt.subplots()
        kwargs = {"mpl_kwargs": {"bins": 20, "color": "blue"}}
        _hist_1d(ax, self.samples, self.limits, kwargs)
        
        # Check that something was plotted
        self.assertGreater(len(ax.patches), 0)

    def test_kde_1d(self):
        """Test _kde_1d creates KDE plot."""
        fig, ax = plt.subplots()
        kwargs = {"bw_method": "scott", "bins": 50, "mpl_kwargs": {"color": "blue"}}
        _kde_1d(ax, self.samples, self.limits, kwargs)
        
        # Check that a line was plotted
        self.assertGreater(len(ax.lines), 0)

    def test_scatter_1d(self):
        """Test _scatter_1d creates vertical lines."""
        fig, ax = plt.subplots()
        kwargs = {"mpl_kwargs": {"color": "red", "alpha": 0.5}}
        samples = np.array([0, 1, 2])
        _scatter_1d(ax, samples, self.limits, kwargs)
        
        # Check that lines were plotted
        self.assertGreater(len(ax.lines), 0)

    def test_hist_2d(self):
        """Test _hist_2d creates 2D histogram."""
        fig, ax = plt.subplots()
        x = np.random.randn(100)
        y = np.random.randn(100)
        limx = np.array([-3, 3])
        limy = np.array([-3, 3])
        kwargs = {"np_hist_kwargs": {"bins": 20}, "mpl_kwargs": {"origin": "lower"}}
        _hist_2d(ax, x, y, limx, limy, kwargs)
        
        # Check that an image was created
        self.assertGreater(len(ax.images), 0)

    def test_kde_2d(self):
        """Test _kde_2d creates 2D KDE plot."""
        fig, ax = plt.subplots()
        x = np.random.randn(100)
        y = np.random.randn(100)
        limx = np.array([-3, 3])
        limy = np.array([-3, 3])
        kwargs = {"bw_method": "scott", "bins": 30, "mpl_kwargs": {"origin": "lower"}}
        _kde_2d(ax, x, y, limx, limy, kwargs)
        
        # Check that an image was created
        self.assertGreater(len(ax.images), 0)

    def test_contour_2d(self):
        """Test _contour_2d creates contour plot."""
        fig, ax = plt.subplots()
        x = np.random.randn(100)
        y = np.random.randn(100)
        limx = np.array([-3, 3])
        limy = np.array([-3, 3])
        kwargs = {
            "bw_method": "scott",
            "bins": 30,
            "levels": [0.68, 0.95],
            "percentile": True,
            "mpl_kwargs": {"colors": "blue"}
        }
        _contour_2d(ax, x, y, limx, limy, kwargs)
        
        # Check that contours were created
        self.assertGreater(len(ax.collections), 0)

    def test_scatter_2d(self):
        """Test _scatter_2d creates scatter plot."""
        fig, ax = plt.subplots()
        x = np.random.randn(100)
        y = np.random.randn(100)
        limx = np.array([-3, 3])
        limy = np.array([-3, 3])
        kwargs = {"mpl_kwargs": {"color": "red", "alpha": 0.5}}
        _scatter_2d(ax, x, y, limx, limy, kwargs)
        
        # Check that scatter plot was created
        self.assertGreater(len(ax.collections), 0)

    def test_plot_2d(self):
        """Test _plot_2d creates line plot."""
        fig, ax = plt.subplots()
        x = np.random.randn(100)
        y = np.random.randn(100)
        limx = np.array([-3, 3])
        limy = np.array([-3, 3])
        kwargs = {"mpl_kwargs": {"color": "blue"}}
        _plot_2d(ax, x, y, limx, limy, kwargs)
        
        # Check that a line was plotted
        self.assertGreater(len(ax.lines), 0)

    def test_format_axis(self):
        """Test _format_axis formats axes correctly."""
        fig, ax = plt.subplots()
        
        # Test hiding both axes
        _format_axis(ax, xhide=True, yhide=True)
        # Check that spines are hidden
        self.assertFalse(ax.spines['bottom'].get_visible())
        self.assertFalse(ax.spines['left'].get_visible())
        
        # Test showing x-axis
        _format_axis(ax, xhide=False, yhide=True, xlabel="X Label")
        self.assertEqual(ax.get_xlabel(), "X Label")
        self.assertTrue(ax.spines['bottom'].get_visible())


@pytest.mark.short
@pytest.mark.fast
class TestPlot1DDistribution(unittest.TestCase):
    """Test suite for plot_1d_distribution function."""

    def setUp(self):
        """Set up test fixtures."""
        self.seed = 42
        np.random.seed(self.seed)
        self.samples = np.random.randn(1000)

    def tearDown(self):
        """Clean up after tests."""
        plt.close('all')

    def test_basic_hist(self):
        """Test basic histogram plot."""
        fig, ax = plt.subplots()
        result = plot_1d_distribution(ax, self.samples, plot_type="hist")
        
        self.assertIsInstance(result, Axes)
        self.assertGreater(len(ax.patches), 0)

    def test_basic_kde(self):
        """Test basic KDE plot."""
        fig, ax = plt.subplots()
        result = plot_1d_distribution(ax, self.samples, plot_type="kde")
        
        self.assertIsInstance(result, Axes)
        self.assertGreater(len(ax.lines), 0)

    def test_basic_scatter(self):
        """Test basic scatter plot."""
        fig, ax = plt.subplots()
        samples = np.array([0, 1, 2, 3])
        result = plot_1d_distribution(ax, samples, plot_type="scatter")
        
        self.assertIsInstance(result, Axes)
        self.assertGreater(len(ax.lines), 0)

    def test_with_limits(self):
        """Test plot with custom limits."""
        fig, ax = plt.subplots()
        plot_1d_distribution(ax, self.samples, plot_type="hist", limits=[-5, 5])
        
        xlim = ax.get_xlim()
        self.assertAlmostEqual(xlim[0], -5, places=1)
        self.assertAlmostEqual(xlim[1], 5, places=1)

    def test_with_label(self):
        """Test plot with label."""
        fig, ax = plt.subplots()
        plot_1d_distribution(ax, self.samples, plot_type="hist", label="Parameter θ")
        
        self.assertEqual(ax.get_xlabel(), "Parameter θ")

    def test_with_points(self):
        """Test plot with reference points."""
        fig, ax = plt.subplots()
        plot_1d_distribution(
            ax, self.samples, plot_type="kde",
            points=0.0,
            points_kwargs={"color": "red", "linewidth": 2},
            points_label="True value"
        )
        
        # Check that vertical line was added
        lines = ax.get_lines()
        self.assertGreaterEqual(len(lines), 2)  # KDE line + reference line

    def test_with_multiple_points(self):
        """Test plot with multiple reference points."""
        fig, ax = plt.subplots()
        plot_1d_distribution(
            ax, self.samples, plot_type="hist",
            points=np.array([0.0, 1.0, -1.0]),
            points_label="Reference points"
        )
        
        # Check that multiple vertical lines were added
        lines = ax.get_lines()
        self.assertGreaterEqual(len(lines), 3)

    def test_with_custom_plot_kwargs(self):
        """Test plot with custom plotting kwargs."""
        fig, ax = plt.subplots()
        plot_1d_distribution(
            ax, self.samples, plot_type="kde",
            plot_kwargs={"mpl_kwargs": {"color": "green", "linewidth": 3}}
        )
        
        # Check that line was plotted
        self.assertGreater(len(ax.lines), 0)

    def test_invalid_plot_type(self):
        """Test that invalid plot type raises error."""
        fig, ax = plt.subplots()
        with self.assertRaises(ValueError):
            plot_1d_distribution(ax, self.samples, plot_type="invalid")

    def test_2d_samples_error(self):
        """Test that 2D samples with multiple columns raise error."""
        fig, ax = plt.subplots()
        samples_2d = np.random.randn(100, 2)
        with self.assertRaises(ValueError):
            plot_1d_distribution(ax, samples_2d, plot_type="hist")

    def test_2d_samples_single_column(self):
        """Test that 2D samples with single column are squeezed."""
        fig, ax = plt.subplots()
        samples_2d = np.random.randn(100, 1)
        result = plot_1d_distribution(ax, samples_2d, plot_type="hist")
        
        self.assertIsInstance(result, Axes)
        self.assertGreater(len(ax.patches), 0)

    def test_list_input(self):
        """Test that list input is converted correctly."""
        fig, ax = plt.subplots()
        samples_list = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = plot_1d_distribution(ax, samples_list, plot_type="scatter")
        
        self.assertIsInstance(result, Axes)


@pytest.mark.short
@pytest.mark.fast
class TestPlot2DDistribution(unittest.TestCase):
    """Test suite for plot_2d_distribution function."""

    def setUp(self):
        """Set up test fixtures."""
        self.seed = 42
        np.random.seed(self.seed)
        self.x = np.random.randn(1000)
        self.y = 0.5 * self.x + np.random.randn(1000) * 0.5

    def tearDown(self):
        """Clean up after tests."""
        plt.close('all')

    def test_basic_scatter(self):
        """Test basic scatter plot."""
        fig, ax = plt.subplots()
        result = plot_2d_distribution(ax, self.x, self.y, plot_type="scatter")
        
        self.assertIsInstance(result, Axes)
        self.assertGreater(len(ax.collections), 0)

    def test_basic_hist2d(self):
        """Test basic 2D histogram."""
        fig, ax = plt.subplots()
        result = plot_2d_distribution(ax, self.x, self.y, plot_type="hist2d")
        
        self.assertIsInstance(result, Axes)
        self.assertGreater(len(ax.images), 0)

    def test_basic_kde2d(self):
        """Test basic 2D KDE."""
        fig, ax = plt.subplots()
        result = plot_2d_distribution(ax, self.x, self.y, plot_type="kde2d")
        
        self.assertIsInstance(result, Axes)
        self.assertGreater(len(ax.images), 0)

    def test_basic_contour(self):
        """Test basic contour plot."""
        fig, ax = plt.subplots()
        result = plot_2d_distribution(ax, self.x, self.y, plot_type="contour")
        
        self.assertIsInstance(result, Axes)
        self.assertGreater(len(ax.collections), 0)

    def test_with_limits(self):
        """Test plot with custom limits."""
        fig, ax = plt.subplots()
        plot_2d_distribution(
            ax, self.x, self.y, plot_type="scatter",
            limits_x=[-5, 5], limits_y=[-5, 5]
        )
        
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        self.assertAlmostEqual(xlim[0], -5, places=1)
        self.assertAlmostEqual(xlim[1], 5, places=1)
        self.assertAlmostEqual(ylim[0], -5, places=1)
        self.assertAlmostEqual(ylim[1], 5, places=1)

    def test_with_labels(self):
        """Test plot with axis labels."""
        fig, ax = plt.subplots()
        plot_2d_distribution(
            ax, self.x, self.y, plot_type="scatter",
            xlabel="Parameter 1", ylabel="Parameter 2"
        )
        
        self.assertEqual(ax.get_xlabel(), "Parameter 1")
        self.assertEqual(ax.get_ylabel(), "Parameter 2")

    def test_with_single_point(self):
        """Test plot with single reference point."""
        fig, ax = plt.subplots()
        plot_2d_distribution(
            ax, self.x, self.y, plot_type="scatter",
            points=[0.0, 0.0],
            points_kwargs={"marker": "x", "markersize": 10, "color": "red"},
            points_label="True value"
        )
        
        # Check that point was added
        lines = ax.get_lines()
        self.assertGreater(len(lines), 0)

    def test_with_multiple_points(self):
        """Test plot with multiple reference points."""
        fig, ax = plt.subplots()
        points = np.array([[0.0, 0.0], [1.0, 0.5], [-1.0, -0.5]])
        plot_2d_distribution(
            ax, self.x, self.y, plot_type="scatter",
            points=points,
            points_label="Reference points"
        )
        
        # Check that points were added
        lines = ax.get_lines()
        self.assertGreater(len(lines), 0)

    def test_with_custom_plot_kwargs(self):
        """Test plot with custom plotting kwargs."""
        fig, ax = plt.subplots()
        plot_2d_distribution(
            ax, self.x, self.y, plot_type="scatter",
            plot_kwargs={"mpl_kwargs": {"color": "green", "alpha": 0.3}}
        )
        
        self.assertGreater(len(ax.collections), 0)

    def test_contour_with_levels(self):
        """Test contour plot with custom levels."""
        fig, ax = plt.subplots()
        plot_2d_distribution(
            ax, self.x, self.y, plot_type="contour",
            plot_kwargs={"levels": [0.68, 0.95], "percentile": True}
        )
        
        self.assertGreater(len(ax.collections), 0)

    def test_invalid_plot_type(self):
        """Test that invalid plot type raises error."""
        fig, ax = plt.subplots()
        with self.assertRaises(ValueError):
            plot_2d_distribution(ax, self.x, self.y, plot_type="invalid")

    def test_mismatched_lengths(self):
        """Test that mismatched array lengths raise error."""
        fig, ax = plt.subplots()
        x = np.random.randn(100)
        y = np.random.randn(50)
        with self.assertRaises(ValueError):
            plot_2d_distribution(ax, x, y, plot_type="scatter")

    def test_2d_input_error(self):
        """Test that 2D input with wrong shape raises error."""
        fig, ax = plt.subplots()
        x = np.random.randn(100, 2)
        y = np.random.randn(100)
        with self.assertRaises(ValueError):
            plot_2d_distribution(ax, x, y, plot_type="scatter")

    def test_2d_input_single_column(self):
        """Test that 2D input with single column is squeezed."""
        fig, ax = plt.subplots()
        x = np.random.randn(100, 1)
        y = np.random.randn(100, 1)
        result = plot_2d_distribution(ax, x, y, plot_type="scatter")
        
        self.assertIsInstance(result, Axes)

    def test_invalid_points_shape(self):
        """Test that invalid points shape raises error."""
        fig, ax = plt.subplots()
        with self.assertRaises(ValueError):
            plot_2d_distribution(
                ax, self.x, self.y, plot_type="scatter",
                points=[1.0, 2.0, 3.0]  # Wrong length
            )


@pytest.mark.short
@pytest.mark.fast
class TestPairplotNumpy(unittest.TestCase):
    """Test suite for pairplot_numpy function."""

    def setUp(self):
        """Set up test fixtures."""
        self.seed = 42
        np.random.seed(self.seed)
        
        # Generate correlated 3D samples
        mean = np.array([0, 1, -1])
        cov = np.array([[1.0, 0.5, 0.2],
                        [0.5, 1.0, 0.3],
                        [0.2, 0.3, 1.0]])
        self.samples_3d = np.random.multivariate_normal(mean, cov, 500)
        
        # Generate 1D samples
        self.samples_1d = np.random.randn(500, 1)

    def tearDown(self):
        """Clean up after tests."""
        plt.close('all')

    def test_basic_pairplot(self):
        """Test basic pairplot with default settings."""
        fig, axes = pairplot_numpy(self.samples_3d)
        
        self.assertIsInstance(fig, Figure)
        self.assertIsNotNone(axes)

    def test_pairplot_with_upper_lower(self):
        """Test pairplot with different upper and lower plots."""
        fig, axes = pairplot_numpy(
            self.samples_3d,
            diag="kde",
            upper="scatter",
            lower="contour"
        )
        
        self.assertIsInstance(fig, Figure)

    def test_pairplot_with_hist(self):
        """Test pairplot with histogram diagonal."""
        fig, axes = pairplot_numpy(
            self.samples_3d,
            diag="hist",
            upper="hist2d"
        )
        
        self.assertIsInstance(fig, Figure)

    def test_pairplot_1d(self):
        """Test pairplot with 1D samples."""
        fig, axes = pairplot_numpy(self.samples_1d, diag="kde")
        
        self.assertIsInstance(fig, Figure)

    def test_pairplot_with_points(self):
        """Test pairplot with reference points."""
        true_params = np.array([[0, 1, -1]])
        fig, axes = pairplot_numpy(
            self.samples_3d,
            points=true_params,
            diag="kde",
            upper="scatter"
        )
        
        self.assertIsInstance(fig, Figure)

    def test_pairplot_with_limits(self):
        """Test pairplot with custom limits."""
        limits = [[-3, 3], [-2, 4], [-4, 2]]
        fig, axes = pairplot_numpy(
            self.samples_3d,
            limits=limits,
            diag="hist"
        )
        
        self.assertIsInstance(fig, Figure)

    def test_pairplot_with_labels(self):
        """Test pairplot with custom labels."""
        labels = ["θ₁", "θ₂", "θ₃"]
        fig, axes = pairplot_numpy(
            self.samples_3d,
            labels=labels,
            diag="kde"
        )
        
        self.assertIsInstance(fig, Figure)

    def test_pairplot_with_subset(self):
        """Test pairplot with subset of dimensions."""
        subset = [0, 2]  # Only plot dimensions 0 and 2
        fig, axes = pairplot_numpy(
            self.samples_3d,
            subset=subset,
            diag="hist"
        )
        
        self.assertIsInstance(fig, Figure)

    def test_pairplot_with_multiple_samples(self):
        """Test pairplot with multiple sample sets."""
        samples1 = self.samples_3d
        samples2 = self.samples_3d + 0.5  # Shifted samples
        
        fig, axes = pairplot_numpy(
            [samples1, samples2],
            diag="kde",
            upper="scatter"
        )
        
        self.assertIsInstance(fig, Figure)

    def test_pairplot_with_custom_figsize(self):
        """Test pairplot with custom figure size."""
        fig, axes = pairplot_numpy(
            self.samples_3d,
            figsize=(12, 12),
            diag="hist"
        )
        
        self.assertIsInstance(fig, Figure)
        # Check that figsize was applied
        self.assertEqual(fig.get_figwidth(), 12)
        self.assertEqual(fig.get_figheight(), 12)

    def test_pairplot_with_fig_kwargs(self):
        """Test pairplot with custom figure kwargs."""
        fig_kwargs = {
            "title": "Test Pairplot",
            "title_format": {"fontsize": 14}
        }
        fig, axes = pairplot_numpy(
            self.samples_3d,
            fig_kwargs=fig_kwargs,
            diag="kde"
        )
        
        self.assertIsInstance(fig, Figure)
        # Check that title was set
        self.assertIsNotNone(fig._suptitle)

    def test_pairplot_with_diag_kwargs(self):
        """Test pairplot with custom diagonal kwargs."""
        diag_kwargs = {
            "mpl_kwargs": {"color": "red", "linewidth": 2}
        }
        fig, axes = pairplot_numpy(
            self.samples_3d,
            diag="kde",
            diag_kwargs=diag_kwargs
        )
        
        self.assertIsInstance(fig, Figure)

    def test_pairplot_with_offdiag_kwargs(self):
        """Test pairplot with custom off-diagonal kwargs."""
        upper_kwargs = {
            "mpl_kwargs": {"alpha": 0.3}
        }
        fig, axes = pairplot_numpy(
            self.samples_3d,
            upper="scatter",
            upper_kwargs=upper_kwargs
        )
        
        self.assertIsInstance(fig, Figure)

    def test_pairplot_offdiag_alias(self):
        """Test that offdiag parameter works as alias for upper."""
        fig, axes = pairplot_numpy(
            self.samples_3d,
            offdiag="scatter",
            diag="hist"
        )
        
        self.assertIsInstance(fig, Figure)

    def test_pairplot_with_list_input(self):
        """Test pairplot with list of lists input."""
        # Convert to numpy first to avoid dimension issues with nested lists
        samples_arr = np.array(self.samples_3d.tolist())
        fig, axes = pairplot_numpy(samples_arr, diag="hist")
        
        self.assertIsInstance(fig, Figure)

    def test_pairplot_exclude_lower(self):
        """Test pairplot with only diagonal and upper."""
        fig, axes = pairplot_numpy(
            self.samples_3d,
            diag="kde",
            upper="scatter",
            lower=None
        )
        
        self.assertIsInstance(fig, Figure)

    def test_pairplot_exclude_upper(self):
        """Test pairplot with only diagonal and lower."""
        fig, axes = pairplot_numpy(
            self.samples_3d,
            diag="kde",
            upper=None,
            lower="contour"
        )
        
        self.assertIsInstance(fig, Figure)


if __name__ == '__main__':
    unittest.main()
