import numpy as np

from qgapselect.qcollide.text_atlas import (
    control_output,
    control_projection_from_head,
    pairwise_distances,
    residual_hidden,
    standardized_projection,
    text_surface_descriptors,
)


def test_text_surface_descriptors_are_finite_and_label_agnostic_shape():
    values = text_surface_descriptors(
        [
            "Hello WORLD! Version 2 is here.",
            "A second sentence, with punctuation? Yes!",
        ]
    )
    assert values.shape == (2, 8)
    assert np.all(np.isfinite(values))
    assert values[0, 4] > 0.0
    assert values[0, 6] > 0.0


def test_control_projection_and_residual_are_orthogonal():
    coefficients = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
        ]
    )
    projection = control_projection_from_head(coefficients, 2)
    hidden = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 1.0, 0.0, 3.0],
            [0.0, 1.0, 2.0, 1.0],
        ]
    )
    residual = residual_hidden(hidden, projection)
    assert projection.shape == (2, 4)
    assert np.allclose(residual @ projection.T, 0.0, atol=1e-10)


def test_whitened_control_output_and_pairwise_distances_are_finite():
    support = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [-1.0, 0.0, 1.0],
        ]
    )
    projection = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    standardized = standardized_projection(projection, support)
    controls = control_output(support, standardized)
    distances = pairwise_distances(controls, controls)
    assert controls.shape == (4, 2)
    assert np.all(np.isfinite(distances))
    assert np.allclose(np.diag(distances), 0.0, atol=1e-10)
