"""
Test unified parameter API for all CuPy models.

This test suite verifies that all CuPy models inherit from BaseCupyModel
and implement the unified parameter management API correctly.
"""

import unittest
import numpy as np
import pytest


# Check if CuPy is available
CUPY_AVAILABLE = True
try:
    import cupy
    if cupy.cuda.is_available():
        ENGINE = "gpu"
    else:
        ENGINE = "cpu"
except ImportError:
    CUPY_AVAILABLE = False
    ENGINE = "cpu"


# Import BaseCupyModel separately to avoid NameError
BASE_AVAILABLE = True
try:
    from vbi.models.cupy.base import BaseCupyModel
except ImportError:
    BASE_AVAILABLE = False


# Import all CuPy models
GHB_AVAILABLE = True
try:
    from vbi.models.cupy.ghb import GHB_sde
except ImportError:
    GHB_AVAILABLE = False

MPR_AVAILABLE = True
try:
    from vbi.models.cupy.mpr import MPR_sde
except ImportError:
    MPR_AVAILABLE = False

WC_AVAILABLE = True
try:
    from vbi.models.cupy.wilson_cowan import WC_sde
except ImportError:
    WC_AVAILABLE = False

WW_AVAILABLE = True
try:
    from vbi.models.cupy.ww import WW_sde
except ImportError:
    WW_AVAILABLE = False

JR_AVAILABLE = True
try:
    from vbi.models.cupy.jansen_rit import JR_sde
except ImportError:
    JR_AVAILABLE = False

KM_AVAILABLE = True
try:
    from vbi.models.cupy.km import KM_sde
except ImportError:
    KM_AVAILABLE = False


class TestCupyUnifiedAPI(unittest.TestCase):
    """Test unified parameter API for CuPy models."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.seed = 42
        np.random.seed(cls.seed)
        
        # Create test connectivity matrices
        cls.weights_2x2 = np.random.rand(2, 2)
        cls.weights_3x3 = np.random.rand(3, 3)
        
        # For Kuramoto model
        cls.omega_3 = np.random.rand(3)

    @pytest.mark.skipif(not GHB_AVAILABLE, reason="GHB model not available")
    def test_ghb_inherits_from_base(self):
        """Test that GHB_sde inherits from BaseCupyModel."""
        par = {'weights': self.weights_2x2, 'seed': self.seed}
        model = GHB_sde(par=par)
        self.assertIsInstance(model, BaseCupyModel)

    @pytest.mark.skipif(not GHB_AVAILABLE, reason="GHB model not available")
    def test_ghb_get_parameters(self):
        """Test GHB_sde.get_parameters() method."""
        par = {'weights': self.weights_2x2, 'seed': self.seed}
        model = GHB_sde(par=par)
        
        params = model.get_parameters()
        self.assertIsInstance(params, dict)
        self.assertGreater(len(params), 0)
        self.assertIn('G', params)
        self.assertIn('eta', params)
        self.assertIn('omega', params)

    @pytest.mark.skipif(not GHB_AVAILABLE, reason="GHB model not available")
    def test_ghb_get_parameter(self):
        """Test GHB_sde.get_parameter() method."""
        par = {'weights': self.weights_2x2, 'G': 30.0, 'seed': self.seed}
        model = GHB_sde(par=par)
        
        G = model.get_parameter('G')
        self.assertEqual(G, 30.0)

    @pytest.mark.skipif(not GHB_AVAILABLE, reason="GHB model not available")
    def test_ghb_get_parameter_descriptions(self):
        """Test GHB_sde.get_parameter_descriptions() method."""
        par = {'weights': self.weights_2x2, 'seed': self.seed}
        model = GHB_sde(par=par)
        
        descriptions = model.get_parameter_descriptions()
        self.assertIsInstance(descriptions, dict)
        self.assertGreater(len(descriptions), 0)
        
        # Check format: each value should be (description, type) tuple
        for key, value in descriptions.items():
            self.assertIsInstance(value, tuple)
            self.assertEqual(len(value), 2)
            self.assertIsInstance(value[0], str)  # description
            self.assertIsInstance(value[1], str)  # type

    @pytest.mark.skipif(not MPR_AVAILABLE, reason="MPR model not available")
    def test_mpr_inherits_from_base(self):
        """Test that MPR_sde inherits from BaseCupyModel."""
        par = {'weights': self.weights_2x2, 'seed': self.seed}
        model = MPR_sde(par=par)
        self.assertIsInstance(model, BaseCupyModel)

    @pytest.mark.skipif(not MPR_AVAILABLE, reason="MPR model not available")
    def test_mpr_get_parameters(self):
        """Test MPR_sde.get_parameters() method."""
        par = {'weights': self.weights_2x2, 'seed': self.seed}
        model = MPR_sde(par=par)
        
        params = model.get_parameters()
        self.assertIsInstance(params, dict)
        self.assertGreater(len(params), 0)
        self.assertIn('G', params)
        self.assertIn('eta', params)
        self.assertIn('tau', params)

    @pytest.mark.skipif(not MPR_AVAILABLE, reason="MPR model not available")
    def test_mpr_get_parameter(self):
        """Test MPR_sde.get_parameter() method."""
        par = {'weights': self.weights_2x2, 'G': 0.8, 'seed': self.seed}
        model = MPR_sde(par=par)
        
        G = model.get_parameter('G')
        self.assertEqual(G, 0.8)

    @pytest.mark.skipif(not MPR_AVAILABLE, reason="MPR model not available")
    def test_mpr_get_parameter_descriptions(self):
        """Test MPR_sde.get_parameter_descriptions() method."""
        par = {'weights': self.weights_2x2, 'seed': self.seed}
        model = MPR_sde(par=par)
        
        descriptions = model.get_parameter_descriptions()
        self.assertIsInstance(descriptions, dict)
        self.assertGreater(len(descriptions), 0)

    @pytest.mark.skipif(not WC_AVAILABLE, reason="Wilson-Cowan model not available")
    def test_wc_inherits_from_base(self):
        """Test that WC_sde inherits from BaseCupyModel."""
        par = {'weights': self.weights_2x2, 'seed': self.seed}
        model = WC_sde(par=par)
        self.assertIsInstance(model, BaseCupyModel)

    @pytest.mark.skipif(not WC_AVAILABLE, reason="Wilson-Cowan model not available")
    def test_wc_get_parameters(self):
        """Test WC_sde.get_parameters() method."""
        par = {'weights': self.weights_2x2, 'seed': self.seed}
        model = WC_sde(par=par)
        
        params = model.get_parameters()
        self.assertIsInstance(params, dict)
        self.assertGreater(len(params), 0)
        self.assertIn('c_ee', params)
        self.assertIn('c_ei', params)
        self.assertIn('tau_e', params)

    @pytest.mark.skipif(not WC_AVAILABLE, reason="Wilson-Cowan model not available")
    def test_wc_get_parameter(self):
        """Test WC_sde.get_parameter() method."""
        par = {'weights': self.weights_2x2, 'c_ee': 18.0, 'seed': self.seed}
        model = WC_sde(par=par)
        
        c_ee = model.get_parameter('c_ee')
        self.assertEqual(c_ee, 18.0)

    @pytest.mark.skipif(not WC_AVAILABLE, reason="Wilson-Cowan model not available")
    def test_wc_get_parameter_descriptions(self):
        """Test WC_sde.get_parameter_descriptions() method."""
        par = {'weights': self.weights_2x2, 'seed': self.seed}
        model = WC_sde(par=par)
        
        descriptions = model.get_parameter_descriptions()
        self.assertIsInstance(descriptions, dict)
        self.assertGreater(len(descriptions), 0)

    @pytest.mark.skipif(not WW_AVAILABLE, reason="Wong-Wang model not available")
    def test_ww_inherits_from_base(self):
        """Test that WW_sde inherits from BaseCupyModel."""
        par = {'weights': self.weights_2x2, 'seed': self.seed}
        model = WW_sde(par=par)
        self.assertIsInstance(model, BaseCupyModel)

    @pytest.mark.skipif(not WW_AVAILABLE, reason="Wong-Wang model not available")
    def test_ww_get_parameters(self):
        """Test WW_sde.get_parameters() method."""
        par = {'weights': self.weights_2x2, 'seed': self.seed}
        model = WW_sde(par=par)
        
        params = model.get_parameters()
        self.assertIsInstance(params, dict)
        self.assertGreater(len(params), 0)
        self.assertIn('a_exc', params)
        self.assertIn('a_inh', params)
        self.assertIn('tau_exc', params)

    @pytest.mark.skipif(not WW_AVAILABLE, reason="Wong-Wang model not available")
    def test_ww_get_parameter(self):
        """Test WW_sde.get_parameter() method."""
        par = {'weights': self.weights_2x2, 'a_exc': 320, 'seed': self.seed}
        model = WW_sde(par=par)
        
        a_exc = model.get_parameter('a_exc')
        self.assertEqual(a_exc, 320)

    @pytest.mark.skipif(not WW_AVAILABLE, reason="Wong-Wang model not available")
    def test_ww_get_parameter_descriptions(self):
        """Test WW_sde.get_parameter_descriptions() method."""
        par = {'weights': self.weights_2x2, 'seed': self.seed}
        model = WW_sde(par=par)
        
        descriptions = model.get_parameter_descriptions()
        self.assertIsInstance(descriptions, dict)
        self.assertGreater(len(descriptions), 0)

    @pytest.mark.skipif(not JR_AVAILABLE, reason="Jansen-Rit model not available")
    def test_jr_inherits_from_base(self):
        """Test that JR_sde inherits from BaseCupyModel."""
        par = {'weights': self.weights_2x2, 'seed': self.seed}
        model = JR_sde(par=par)
        self.assertIsInstance(model, BaseCupyModel)

    @pytest.mark.skipif(not JR_AVAILABLE, reason="Jansen-Rit model not available")
    def test_jr_get_parameters(self):
        """Test JR_sde.get_parameters() method."""
        par = {'weights': self.weights_2x2, 'seed': self.seed}
        model = JR_sde(par=par)
        
        params = model.get_parameters()
        self.assertIsInstance(params, dict)
        self.assertGreater(len(params), 0)
        self.assertIn('A', params)
        self.assertIn('B', params)
        self.assertIn('C0', params)

    @pytest.mark.skipif(not JR_AVAILABLE, reason="Jansen-Rit model not available")
    def test_jr_get_parameter(self):
        """Test JR_sde.get_parameter() method."""
        par = {'weights': self.weights_2x2, 'A': 3.5, 'seed': self.seed}
        model = JR_sde(par=par)
        
        A = model.get_parameter('A')
        self.assertEqual(A, 3.5)

    @pytest.mark.skipif(not JR_AVAILABLE, reason="Jansen-Rit model not available")
    def test_jr_get_parameter_descriptions(self):
        """Test JR_sde.get_parameter_descriptions() method."""
        par = {'weights': self.weights_2x2, 'seed': self.seed}
        model = JR_sde(par=par)
        
        descriptions = model.get_parameter_descriptions()
        self.assertIsInstance(descriptions, dict)
        self.assertGreater(len(descriptions), 0)

    @pytest.mark.skipif(not KM_AVAILABLE, reason="Kuramoto model not available")
    def test_km_inherits_from_base(self):
        """Test that KM_sde inherits from BaseCupyModel."""
        par = {'weights': self.weights_3x3, 'omega': self.omega_3, 'seed': self.seed}
        model = KM_sde(par=par)
        self.assertIsInstance(model, BaseCupyModel)

    @pytest.mark.skipif(not KM_AVAILABLE, reason="Kuramoto model not available")
    def test_km_get_parameters(self):
        """Test KM_sde.get_parameters() method."""
        par = {'weights': self.weights_3x3, 'omega': self.omega_3, 'seed': self.seed}
        model = KM_sde(par=par)
        
        params = model.get_parameters()
        self.assertIsInstance(params, dict)
        self.assertGreater(len(params), 0)
        self.assertIn('G', params)
        self.assertIn('omega', params)
        self.assertIn('noise_amp', params)

    @pytest.mark.skipif(not KM_AVAILABLE, reason="Kuramoto model not available")
    def test_km_get_parameter(self):
        """Test KM_sde.get_parameter() method."""
        par = {'weights': self.weights_3x3, 'omega': self.omega_3, 'G': 2.0, 'seed': self.seed}
        model = KM_sde(par=par)
        
        G = model.get_parameter('G')
        self.assertEqual(G, 2.0)

    @pytest.mark.skipif(not KM_AVAILABLE, reason="Kuramoto model not available")
    def test_km_get_parameter_descriptions(self):
        """Test KM_sde.get_parameter_descriptions() method."""
        par = {'weights': self.weights_3x3, 'omega': self.omega_3, 'seed': self.seed}
        model = KM_sde(par=par)
        
        descriptions = model.get_parameter_descriptions()
        self.assertIsInstance(descriptions, dict)
        self.assertGreater(len(descriptions), 0)

    @pytest.mark.skipif(not GHB_AVAILABLE, reason="GHB model not available")
    def test_ghb_formatted_table(self):
        """Test that GHB_sde returns formatted table via __str__."""
        par = {'weights': self.weights_2x2, 'seed': self.seed}
        model = GHB_sde(par=par)
        
        table = str(model)
        self.assertIsInstance(table, str)
        self.assertIn('Hopf', table)  # Check for "Hopf" from "Generic Hopf Bifurcation"
        self.assertIn('Parameter', table)
        self.assertIn('Description', table)
        self.assertIn('Value/Shape', table)
        self.assertIn('Type', table)

    @pytest.mark.skipif(not MPR_AVAILABLE, reason="MPR model not available")
    def test_mpr_formatted_table(self):
        """Test that MPR_sde returns formatted table via __str__."""
        par = {'weights': self.weights_2x2, 'seed': self.seed}
        model = MPR_sde(par=par)
        
        table = str(model)
        self.assertIsInstance(table, str)
        self.assertIn('MPR', table)
        self.assertIn('Parameter', table)

    def test_all_models_have_consistent_api(self):
        """Test that all available models have consistent API methods."""
        models_to_test = []
        
        if GHB_AVAILABLE:
            models_to_test.append(
                ('GHB', GHB_sde, {'weights': self.weights_2x2, 'seed': self.seed})
            )
        if MPR_AVAILABLE:
            models_to_test.append(
                ('MPR', MPR_sde, {'weights': self.weights_2x2, 'seed': self.seed})
            )
        if WC_AVAILABLE:
            models_to_test.append(
                ('WC', WC_sde, {'weights': self.weights_2x2, 'seed': self.seed})
            )
        if WW_AVAILABLE:
            models_to_test.append(
                ('WW', WW_sde, {'weights': self.weights_2x2, 'seed': self.seed})
            )
        if JR_AVAILABLE:
            models_to_test.append(
                ('JR', JR_sde, {'weights': self.weights_2x2, 'seed': self.seed})
            )
        if KM_AVAILABLE:
            models_to_test.append(
                ('KM', KM_sde, {'weights': self.weights_3x3, 'omega': self.omega_3, 'seed': self.seed})
            )
        
        # Skip if no models available
        if len(models_to_test) == 0:
            self.skipTest("No CuPy models available")
        
        for name, ModelClass, par in models_to_test:
            with self.subTest(model=name):
                model = ModelClass(par=par)
                
                # Check all required methods exist
                self.assertTrue(hasattr(model, 'get_parameters'))
                self.assertTrue(hasattr(model, 'get_parameter'))
                self.assertTrue(hasattr(model, 'get_default_parameters'))
                self.assertTrue(hasattr(model, 'get_parameter_descriptions'))
                
                # Check methods are callable
                self.assertTrue(callable(model.get_parameters))
                self.assertTrue(callable(model.get_parameter))
                self.assertTrue(callable(model.get_default_parameters))
                self.assertTrue(callable(model.get_parameter_descriptions))
                
                # Check methods return correct types
                params = model.get_parameters()
                self.assertIsInstance(params, dict)
                
                defaults = model.get_default_parameters()
                self.assertIsInstance(defaults, dict)
                
                descriptions = model.get_parameter_descriptions()
                self.assertIsInstance(descriptions, dict)
                
                # Check that descriptions match parameters
                self.assertEqual(len(descriptions), len(defaults))

    def test_parameter_consistency(self):
        """Test that get_parameters() and get_parameter() are consistent."""
        models_to_test = []
        
        if GHB_AVAILABLE:
            models_to_test.append(
                (GHB_sde, {'weights': self.weights_2x2, 'seed': self.seed}, 'G')
            )
        if MPR_AVAILABLE:
            models_to_test.append(
                (MPR_sde, {'weights': self.weights_2x2, 'seed': self.seed}, 'G')
            )
        if WC_AVAILABLE:
            models_to_test.append(
                (WC_sde, {'weights': self.weights_2x2, 'seed': self.seed}, 'c_ee')
            )
        if WW_AVAILABLE:
            models_to_test.append(
                (WW_sde, {'weights': self.weights_2x2, 'seed': self.seed}, 'a_exc')
            )
        if JR_AVAILABLE:
            models_to_test.append(
                (JR_sde, {'weights': self.weights_2x2, 'seed': self.seed}, 'A')
            )
        if KM_AVAILABLE:
            models_to_test.append(
                (KM_sde, {'weights': self.weights_3x3, 'omega': self.omega_3, 'seed': self.seed}, 'G')
            )
        
        if len(models_to_test) == 0:
            self.skipTest("No CuPy models available")
        
        for ModelClass, par, param_name in models_to_test:
            with self.subTest(model=ModelClass.__name__):
                model = ModelClass(par=par)
                
                # Get parameter via get_parameters()
                all_params = model.get_parameters()
                value_from_dict = all_params[param_name]
                
                # Get parameter via get_parameter()
                value_from_method = model.get_parameter(param_name)
                
                # Should be equal
                if isinstance(value_from_dict, np.ndarray):
                    np.testing.assert_array_equal(value_from_dict, value_from_method)
                else:
                    self.assertEqual(value_from_dict, value_from_method)


if __name__ == '__main__':
    unittest.main()
