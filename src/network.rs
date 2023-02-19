use itertools::Itertools;
use ndarray::{s, Array, Array1, Array2, Array3, Array4};

#[derive(Debug, Clone)]
pub struct Conv2d {
    in_channels: usize,
    out_channels: usize,
    weights: Array4<f32>,
    bias: Array1<f32>,
}

impl Conv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        weights: Array4<f32>,
        bias: Array1<f32>,
    ) -> Self {
        Self {
            in_channels,
            out_channels,
            weights,
            bias,
        }
    }

    pub fn apply(&self, x: &Array3<f32>) -> Array3<f32> {
        let x_shape = x.shape();
        let mut y = Array::zeros((self.out_channels, x_shape[1], x_shape[2]));
        let pads = x
            .outer_iter()
            .map(|x| {
                let x_shape = x.shape();
                let dim0 = x_shape[0];
                let dim1 = x_shape[1];

                // padding
                let mut pad: Array2<f32> = Array::zeros((dim0 + 2, dim1 + 2));
                pad.slice_mut(s![1..=dim0, 1..=dim1]).assign(&x);
                pad
            })
            .collect_vec();

        for (mut y, kernel) in y.outer_iter_mut().zip(self.weights.outer_iter()) {
            for (pad, kernel) in pads.iter().zip(kernel.outer_iter()) {
                for (row, mut y) in y.outer_iter_mut().enumerate() {
                    for (col, y) in y.iter_mut().enumerate() {
                        *y += pad[[row, col]] * kernel[[0, 0]];
                        *y += pad[[row, col + 1]] * kernel[[0, 1]];
                        *y += pad[[row, col + 2]] * kernel[[0, 2]];
                        *y += pad[[row + 1, col]] * kernel[[1, 0]];
                        *y += pad[[row + 1, col + 1]] * kernel[[1, 1]];
                        *y += pad[[row + 1, col + 2]] * kernel[[1, 2]];
                        *y += pad[[row + 2, col]] * kernel[[2, 0]];
                        *y += pad[[row + 2, col + 1]] * kernel[[2, 1]];
                        *y += pad[[row + 2, col + 2]] * kernel[[2, 2]];
                    }
                }
            }
        }

        for (mut y, bias) in y.outer_iter_mut().zip(self.bias.iter()) {
            for y in y.iter_mut() {
                *y += *bias;
            }
        }

        y
    }
}

#[cfg(test)]
mod test {
    use ndarray::{array, Array3, Array4};

    use super::Conv2d;

    #[test]
    fn conv2d() {
        let x = Array3::from_shape_vec(
            [2, 4, 4],
            vec![
                0.3745, 0.9507, 0.7320, 0.5987, 0.1560, 0.1560, 0.0581, 0.8662, 0.6011, 0.7081,
                0.0206, 0.9699, 0.8324, 0.2123, 0.1818, 0.1834, 0.3042, 0.5248, 0.4319, 0.2912,
                0.6119, 0.1395, 0.2921, 0.3664, 0.4561, 0.7852, 0.1997, 0.5142, 0.5924, 0.0465,
                0.6075, 0.1705,
            ],
        )
        .unwrap();

        let expected_y = Array3::from_shape_vec(
            [3, 4, 4],
            vec![
                0.3332, 0.2844, 0.6340, 0.4501, 0.5494, 0.6954, 0.6584, 0.5510, 0.5963, 0.2623,
                0.6651, 0.2932, 0.2658, 0.7560, 0.1659, 0.4733, 0.1912, 0.1550, 0.1166, -0.0678,
                -0.2192, -0.1559, -0.1053, -0.2381, 0.0328, -0.2550, 0.1509, -0.0868, -0.2724,
                0.1906, -0.3862, 0.2094, -0.1582, -0.0478, 0.0958, 0.1631, 0.4057, 0.3927, 0.3058,
                0.3384, 0.0475, 0.1810, -0.0103, 0.2418, 0.2456, 0.1627, -0.0401, 0.3557,
            ],
        )
        .unwrap();

        let weights = Array4::from_shape_vec(
            [3, 2, 3, 3],
            vec![
                0.1802, 0.1956, -0.0552, 0.2165, -0.0516, 0.0476, -0.1148, 0.1384, 0.2078, -0.1729,
                0.2049, 0.0441, 0.1741, 0.0319, 0.1137, -0.0333, 0.1817, 0.0348, -0.1100, 0.0601,
                -0.1086, -0.0276, -0.0957, 0.1564, -0.1861, -0.1087, -0.0666, -0.1417, 0.0222,
                -0.2328, 0.2129, -0.2002, 0.1820, 0.0392, -0.0765, 0.1457, 0.0367, 0.1904, 0.0258,
                -0.0743, 0.0633, -0.0639, 0.0992, 0.2104, 0.1362, -0.1030, 0.1361, 0.0422, 0.1197,
                -0.1437, -0.2333, -0.0911, -0.1808, 0.1934,
            ],
        )
        .unwrap();
        let bias = array![0.0679, 0.0976, 0.0745];

        let conv2d = Conv2d::new(2, 3, weights, bias);
        let y = conv2d.apply(&x);

        for (expected, actual) in expected_y.iter().zip(y.iter()) {
            assert!((expected - actual).abs() < 1e-3);
        }
    }
}
