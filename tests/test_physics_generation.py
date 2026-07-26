import unittest

import numpy as np

from physics_generation import (
    estimate_motion_geometry,
    transformation_vector,
    warp_doppler_time_map,
)


class PhysicsGenerationTests(unittest.TestCase):
    def test_geometry_recovers_known_motion(self):
        distance = 1.5
        speed = 1.0
        theta_deg = 60.0
        frame_interval = 0.05
        wavelength = 0.004
        cosine = np.cos(np.deg2rad(theta_deg))
        doppler = 2.0 * speed * cosine / wavelength
        next_distance = np.sqrt(
            distance**2
            + (speed * frame_interval) ** 2
            - 2.0 * distance * speed * frame_interval * cosine
        )

        result = estimate_motion_geometry(
            np.array([doppler]),
            np.array([distance, next_distance]),
            frame_interval,
            wavelength,
        )

        self.assertAlmostEqual(result.speed_m_s[0], speed, places=10)
        self.assertAlmostEqual(result.radial_cosine[0], cosine, places=10)

    def test_transformation_vector_matches_equation_nine(self):
        vector = transformation_vector(np.array([0.0, 30.0]), 60.0)
        expected = np.array([0.5, 0.0])
        np.testing.assert_allclose(vector, expected, atol=1e-12)

    def test_zero_deviation_is_identity(self):
        dtm = np.arange(35, dtype=np.float64).reshape(7, 5)
        axis = np.linspace(-3.0, 3.0, dtm.shape[0])
        vector = transformation_vector(np.zeros(dtm.shape[1]), 0.0)
        generated = warp_doppler_time_map(dtm, axis, vector)
        np.testing.assert_allclose(generated, dtm)

    def test_negative_factor_reverses_doppler_sign(self):
        axis = np.linspace(-2.0, 2.0, 9)
        dtm = np.zeros((axis.size, 1), dtype=np.float64)
        dtm[np.argmin(np.abs(axis - 2.0)), 0] = 1.0
        vector = transformation_vector(0.0, 120.0)

        generated = warp_doppler_time_map(dtm, axis, vector)

        peak_frequency = axis[np.argmax(generated[:, 0])]
        self.assertAlmostEqual(peak_frequency, -1.0)


if __name__ == "__main__":
    unittest.main()
