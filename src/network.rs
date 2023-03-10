mod weight;
use itertools::{izip, Itertools};
use ndarray::{s, stack, Array, Array1, Array2, Array3, Array4, Axis};

use self::weight::UNetWeightDict;

pub trait NNModule {
    fn apply(&self, x: &Array3<f32>) -> Array3<f32>;
}

#[derive(Debug, Clone)]
pub struct UNet {
    double_conv1: DoubleConvBlock,
    double_conv2: DoubleConvBlock,
    double_conv3: DoubleConvBlock,
    double_conv4: DoubleConvBlock,
    double_conv5: DoubleConvBlock,
    double_conv6: DoubleConvBlock,
    double_conv7: DoubleConvBlock,
    up_conv1: UpConvBlock,
    up_conv2: UpConvBlock,
    up_conv3: UpConvBlock,
    last_conv: Conv2d,
    maxpool: MaxPoolX2,
    sigmoid: Sigmoid,
}

impl UNet {
    pub fn new() -> Self {
        let dict = UNetWeightDict::new();
        Self {
            double_conv1: DoubleConvBlock::new_dict(2, 4, 1, &dict),
            double_conv2: DoubleConvBlock::new_dict(4, 8, 2, &dict),
            double_conv3: DoubleConvBlock::new_dict(8, 16, 3, &dict),
            double_conv4: DoubleConvBlock::new_dict(16, 32, 4, &dict),
            double_conv5: DoubleConvBlock::new_dict(32, 16, 5, &dict),
            double_conv6: DoubleConvBlock::new_dict(16, 8, 6, &dict),
            double_conv7: DoubleConvBlock::new_dict(8, 4, 7, &dict),
            up_conv1: UpConvBlock::new_dict(32, 16, 1, &dict),
            up_conv2: UpConvBlock::new_dict(16, 8, 2, &dict),
            up_conv3: UpConvBlock::new_dict(8, 4, 3, &dict),
            last_conv: Conv2d::from_vec(4, 1, dict.get("conv1.weight"), dict.get("conv1.bias")),
            maxpool: MaxPoolX2,
            sigmoid: Sigmoid,
        }
    }
}

impl NNModule for UNet {
    fn apply(&self, x: &Array3<f32>) -> Array3<f32> {
        // encoder
        let x = self.double_conv1.apply(x);
        let x1 = x.clone();
        let x = self.maxpool.apply(&x);

        let x = self.double_conv2.apply(&x);
        let x2 = x.clone();
        let x = self.maxpool.apply(&x);

        let x = self.double_conv3.apply(&x);
        let x3 = x.clone();
        let x = self.maxpool.apply(&x);

        // middle
        let x = self.double_conv4.apply(&x);

        // decoder
        let x = self.up_conv1.apply(&x);
        let x = stack![Axis(0), x3, x];
        let x = self.double_conv5.apply(&x);

        let x = self.up_conv2.apply(&x);
        let x = stack![Axis(0), x2, x];
        let x = self.double_conv6.apply(&x);

        let x = self.up_conv3.apply(&x);
        let x = stack![Axis(0), x1, x];
        let x = self.double_conv7.apply(&x);

        let x = self.last_conv.apply(&x);
        let x = self.sigmoid.apply(&x);

        x
    }
}

#[derive(Debug, Clone)]
struct DoubleConvBlock {
    conv1: Conv2d,
    conv2: Conv2d,
    batch_norm1: BatchNorm2d,
    batch_norm2: BatchNorm2d,
    relu: Relu,
}

impl DoubleConvBlock {
    fn new(
        in_channels: usize,
        out_channels: usize,
        conv1_weight: Vec<f32>,
        conv1_bias: Vec<f32>,
        conv2_weight: Vec<f32>,
        conv2_bias: Vec<f32>,
        bn1_weight: Vec<f32>,
        bn1_bias: Vec<f32>,
        bn1_mean: Vec<f32>,
        bn1_var: Vec<f32>,
        bn2_weight: Vec<f32>,
        bn2_bias: Vec<f32>,
        bn2_mean: Vec<f32>,
        bn2_var: Vec<f32>,
    ) -> Self {
        let conv1 = Conv2d::from_vec(in_channels, out_channels, conv1_weight, conv1_bias);
        let batch_norm1 =
            BatchNorm2d::from_vec(out_channels, bn1_weight, bn1_bias, bn1_mean, bn1_var);
        let conv2 = Conv2d::from_vec(out_channels, out_channels, conv2_weight, conv2_bias);
        let batch_norm2 =
            BatchNorm2d::from_vec(out_channels, bn2_weight, bn2_bias, bn2_mean, bn2_var);
        let relu = Relu;

        Self {
            conv1,
            conv2,
            batch_norm1,
            batch_norm2,
            relu,
        }
    }

    fn new_dict(in_channels: usize, out_channels: usize, id: usize, dict: &UNetWeightDict) -> Self {
        Self::new(
            in_channels,
            out_channels,
            dict.get(&format!("TCB{}.conv1.weight", id)),
            dict.get(&format!("TCB{}.conv1.bias", id)),
            dict.get(&format!("TCB{}.conv2.weight", id)),
            dict.get(&format!("TCB{}.conv2.bias", id)),
            dict.get(&format!("TCB{}.bn1.weight", id)),
            dict.get(&format!("TCB{}.bn1.bias", id)),
            dict.get(&format!("TCB{}.bn1.running_mean", id)),
            dict.get(&format!("TCB{}.bn1.running_var", id)),
            dict.get(&format!("TCB{}.bn2.weight", id)),
            dict.get(&format!("TCB{}.bn2.bias", id)),
            dict.get(&format!("TCB{}.bn2.running_mean", id)),
            dict.get(&format!("TCB{}.bn2.running_var", id)),
        )
    }
}

impl NNModule for DoubleConvBlock {
    fn apply(&self, x: &Array3<f32>) -> Array3<f32> {
        let x = self.conv1.apply(x);
        let x = self.batch_norm1.apply(&x);
        let x = self.relu.apply(&x);
        let x = self.conv2.apply(&x);
        let x = self.batch_norm2.apply(&x);
        let x = self.relu.apply(&x);
        x
    }
}

#[derive(Debug, Clone)]
struct UpConvBlock {
    upsample: BilinearX2,
    conv: Conv2d,
    batch_norm1: BatchNorm2d,
    batch_norm2: BatchNorm2d,
    relu: Relu,
}

impl UpConvBlock {
    fn new(
        in_channels: usize,
        out_channels: usize,
        conv_weight: Vec<f32>,
        conv_bias: Vec<f32>,
        bn1_weight: Vec<f32>,
        bn1_bias: Vec<f32>,
        bn1_mean: Vec<f32>,
        bn1_var: Vec<f32>,
        bn2_weight: Vec<f32>,
        bn2_bias: Vec<f32>,
        bn2_mean: Vec<f32>,
        bn2_var: Vec<f32>,
    ) -> Self {
        let upsample = BilinearX2;
        let batch_norm1 =
            BatchNorm2d::from_vec(in_channels, bn1_weight, bn1_bias, bn1_mean, bn1_var);
        let conv = Conv2d::from_vec(in_channels, out_channels, conv_weight, conv_bias);
        let batch_norm2 =
            BatchNorm2d::from_vec(out_channels, bn2_weight, bn2_bias, bn2_mean, bn2_var);
        let relu = Relu;

        Self {
            upsample,
            conv,
            batch_norm1,
            batch_norm2,
            relu,
        }
    }

    fn new_dict(in_channels: usize, out_channels: usize, id: usize, dict: &UNetWeightDict) -> Self {
        Self::new(
            in_channels,
            out_channels,
            dict.get(&format!("UC{}.conv.weight", id)),
            dict.get(&format!("UC{}.conv.bias", id)),
            dict.get(&format!("UC{}.bn1.weight", id)),
            dict.get(&format!("UC{}.bn1.bias", id)),
            dict.get(&format!("UC{}.bn1.running_mean", id)),
            dict.get(&format!("UC{}.bn1.running_var", id)),
            dict.get(&format!("UC{}.bn2.weight", id)),
            dict.get(&format!("UC{}.bn2.bias", id)),
            dict.get(&format!("UC{}.bn2.running_mean", id)),
            dict.get(&format!("UC{}.bn2.running_var", id)),
        )
    }
}

impl NNModule for UpConvBlock {
    fn apply(&self, x: &Array3<f32>) -> Array3<f32> {
        let x = self.upsample.apply(x);
        let x = self.batch_norm1.apply(&x);
        let x = self.conv.apply(&x);
        let x = self.batch_norm2.apply(&x);
        let x = self.relu.apply(&x);
        x
    }
}

#[derive(Debug, Clone)]
struct Conv2d {
    in_channels: usize,
    out_channels: usize,
    /// [C_out, C_in, K, K]
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

    fn from_vec(c_in: usize, c_out: usize, weight: Vec<f32>, bias: Vec<f32>) -> Self {
        const KERNEL_SIZE: usize = 3;
        let w = Array4::from_shape_vec([c_out, c_in, KERNEL_SIZE, KERNEL_SIZE], weight).unwrap();
        let b = Array1::from_shape_vec([c_out], bias).unwrap();
        Self::new(c_in, c_out, w, b)
    }
}

impl NNModule for Conv2d {
    fn apply(&self, x: &Array3<f32>) -> Array3<f32> {
        assert_eq!(x.len_of(Axis(0)), self.in_channels);
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

#[derive(Debug, Clone, Copy)]
struct Relu;

impl NNModule for Relu {
    fn apply(&self, x: &Array3<f32>) -> Array3<f32> {
        let y = x.map(|x| x.max(0.0));
        y
    }
}

#[derive(Debug, Clone, Copy)]
struct Sigmoid;

impl NNModule for Sigmoid {
    fn apply(&self, x: &Array3<f32>) -> Array3<f32> {
        let y = x.map(|x| 1.0 / (1.0 + (-x).exp()));
        y
    }
}

#[derive(Debug, Clone)]
struct BatchNorm2d {
    weights: Array1<f32>,
    bias: Array1<f32>,
    running_mean: Array1<f32>,
    running_var: Array1<f32>,
    eps: f32,
}

impl BatchNorm2d {
    fn new(
        weights: Array1<f32>,
        bias: Array1<f32>,
        running_mean: Array1<f32>,
        running_var: Array1<f32>,
        eps: f32,
    ) -> Self {
        Self {
            weights,
            bias,
            running_mean,
            running_var,
            eps,
        }
    }

    fn from_vec(
        channel: usize,
        weights: Vec<f32>,
        bias: Vec<f32>,
        mean: Vec<f32>,
        var: Vec<f32>,
    ) -> Self {
        const EPS: f32 = 1e-5;
        let w = Array1::from_shape_vec([channel], weights).unwrap();
        let b = Array1::from_shape_vec([channel], bias).unwrap();
        let m = Array1::from_shape_vec([channel], mean).unwrap();
        let v = Array1::from_shape_vec([channel], var).unwrap();
        Self::new(w, b, m, v, EPS)
    }
}

impl NNModule for BatchNorm2d {
    fn apply(&self, x: &Array3<f32>) -> Array3<f32> {
        let x_shape = x.shape();
        let mut y = Array3::zeros((x_shape[0], x_shape[1], x_shape[2]));

        for (x, mut y, weight, bias, mean, var) in izip!(
            x.outer_iter(),
            y.outer_iter_mut(),
            self.weights.iter(),
            self.bias.iter(),
            self.running_mean.iter(),
            self.running_var.iter()
        ) {
            let denominator_inv = 1.0 / (*var + self.eps).sqrt();
            let x = x.map(|v| (*v - mean) * denominator_inv * weight + bias);
            y.assign(&x);
        }

        y
    }
}

/// bilinear????????????????????????????????????2??????????????????
#[derive(Debug, Clone)]
struct BilinearX2;

impl NNModule for BilinearX2 {
    fn apply(&self, x: &Array3<f32>) -> Array3<f32> {
        const EPS: f32 = 1e-5;
        let x_shape = x.shape();
        let mut y = Array3::zeros((x_shape[0], x_shape[1] * 2, x_shape[2] * 2));

        for (x, mut y) in izip!(x.outer_iter(), y.outer_iter_mut()) {
            let y_shape = y.shape();
            let y_shape_1 = y_shape[0];
            let y_shape_2 = y_shape[1];

            for row in 0..y_shape_1 {
                let pos_y = row as f32 * (x_shape[1] - 1) as f32 / (y_shape_1 - 1) as f32;
                let floor_y = (pos_y + EPS).floor() as usize;
                let ceil_y = (pos_y - EPS).ceil() as usize;
                let dy = pos_y - floor_y as f32;

                for col in 0..y_shape_2 {
                    let pos_x = col as f32 * (x_shape[2] - 1) as f32 / (y_shape_2 - 1) as f32;
                    let floor_x = (pos_x + EPS).floor() as usize;
                    let ceil_x = (pos_x - EPS).ceil() as usize;
                    let dx = pos_x - floor_x as f32;

                    let x00 = x[[floor_y, floor_x]];
                    let x01 = x[[floor_y, ceil_x]];
                    let x10 = x[[ceil_y, floor_x]];
                    let x11 = x[[ceil_y, ceil_x]];

                    let x0 = x00 * (1.0 - dx) + x01 * dx;
                    let x1 = x10 * (1.0 - dx) + x11 * dx;
                    y[[row, col]] = x0 * (1.0 - dy) + x1 * dy;
                }
            }
        }

        y
    }
}

/// 2x2???kernel???stride=2???MaxPooling?????????
#[derive(Debug, Clone, Copy)]
struct MaxPoolX2;

impl NNModule for MaxPoolX2 {
    fn apply(&self, x: &Array3<f32>) -> Array3<f32> {
        let x_shape = x.shape();
        assert!(x_shape[1] % 2 == 0);
        assert!(x_shape[2] % 2 == 0);
        let mut y = Array3::zeros((x_shape[0], x_shape[1] / 2, x_shape[2] / 2));

        for (x, mut y) in izip!(x.outer_iter(), y.outer_iter_mut()) {
            let y_shape = y.shape();
            let y_shape_1 = y_shape[0];
            let y_shape_2 = y_shape[1];

            for row in 0..y_shape_1 {
                for col in 0..y_shape_2 {
                    let x00 = x[[row * 2, col * 2]];
                    let x01 = x[[row * 2, col * 2 + 1]];
                    let x10 = x[[row * 2 + 1, col * 2]];
                    let x11 = x[[row * 2 + 1, col * 2 + 1]];

                    let max = x00.max(x01).max(x10).max(x11);
                    y[[row, col]] = max;
                }
            }
        }

        y
    }
}

#[cfg(test)]
mod test {
    use ndarray::{array, Array3, Array4};

    use super::{BatchNorm2d, BilinearX2, Conv2d, MaxPoolX2, NNModule, Relu, Sigmoid};

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

    #[test]
    fn relu() {
        let x = Array3::from_shape_vec(
            [2, 4, 4],
            vec![
                -0.2509, 0.9014, 0.4640, 0.1973, -0.6880, -0.6880, -0.8838, 0.7324, 0.2022, 0.4161,
                -0.9588, 0.9398, 0.6649, -0.5753, -0.6364, -0.6332, -0.3915, 0.0495, -0.1361,
                -0.4175, 0.2237, -0.7210, -0.4157, -0.2673, -0.0879, 0.5704, -0.6007, 0.0285,
                0.1848, -0.9071, 0.2151, -0.6590,
            ],
        )
        .unwrap();

        let expected_y = Array3::from_shape_vec(
            [2, 4, 4],
            vec![
                0.0000, 0.9014, 0.4640, 0.1973, 0.0000, 0.0000, 0.0000, 0.7324, 0.2022, 0.4161,
                0.0000, 0.9398, 0.6649, 0.0000, 0.0000, 0.0000, 0.0000, 0.0495, 0.0000, 0.0000,
                0.2237, 0.0000, 0.0000, 0.0000, 0.0000, 0.5704, 0.0000, 0.0285, 0.1848, 0.0000,
                0.2151, 0.0000,
            ],
        )
        .unwrap();

        let relu = Relu;
        let y = relu.apply(&x);

        for (expected, actual) in expected_y.iter().zip(y.iter()) {
            assert!((expected - actual).abs() < 1e-3);
        }
    }

    #[test]
    fn sigmoid() {
        let x = Array3::from_shape_vec(
            [2, 4, 4],
            vec![
                -0.2509, 0.9014, 0.4640, 0.1973, -0.6880, -0.6880, -0.8838, 0.7324, 0.2022, 0.4161,
                -0.9588, 0.9398, 0.6649, -0.5753, -0.6364, -0.6332, -0.3915, 0.0495, -0.1361,
                -0.4175, 0.2237, -0.7210, -0.4157, -0.2673, -0.0879, 0.5704, -0.6007, 0.0285,
                0.1848, -0.9071, 0.2151, -0.6590,
            ],
        )
        .unwrap();

        let expected_y = Array3::from_shape_vec(
            [2, 4, 4],
            vec![
                0.4376, 0.7112, 0.6140, 0.5492, 0.3345, 0.3345, 0.2924, 0.6753, 0.5504, 0.6026,
                0.2771, 0.7191, 0.6604, 0.3600, 0.3461, 0.3468, 0.4034, 0.5124, 0.4660, 0.3971,
                0.5557, 0.3272, 0.3975, 0.4336, 0.4780, 0.6388, 0.3542, 0.5071, 0.5461, 0.2876,
                0.5536, 0.3410,
            ],
        )
        .unwrap();

        let sigmoid = Sigmoid;
        let y = sigmoid.apply(&x);

        for (expected, actual) in expected_y.iter().zip(y.iter()) {
            assert!((expected - actual).abs() < 1e-3);
        }
    }

    #[test]
    fn batch_norm2d() {
        let x = Array3::from_shape_vec(
            [4, 2, 2],
            vec![
                0.3745, 0.9507, 0.7320, 0.5987, 0.1560, 0.1560, 0.0581, 0.8662, 0.6011, 0.7081,
                0.0206, 0.9699, 0.8324, 0.2123, 0.1818, 0.1834,
            ],
        )
        .unwrap();

        let expected_y = Array3::from_shape_vec(
            [4, 2, 2],
            vec![
                -1.0138, -0.8100, -0.8874, -0.9346, -0.1695, -0.1695, -0.1827, -0.0738, 2.3414,
                2.3676, 2.1992, 2.4317, -0.4455, -0.5952, -0.6026, -0.6022,
            ],
        )
        .unwrap();
        let weight = array![0.3367, 0.1288, 0.2345, 0.2303];
        let bias = array![-1.1229, -0.1863, 2.2082, -0.6380];
        let mean = array![0.0664, 0.0309, 0.0575, 0.0353];
        let var = array![0.9058, 0.9140, 0.9161, 0.9103];
        let eps = 1e-5;

        let batch_norm = BatchNorm2d::new(weight, bias, mean, var, eps);
        let y = batch_norm.apply(&x);

        for (expected, actual) in expected_y.iter().zip(y.iter()) {
            assert!((expected - actual).abs() < 1e-3);
        }
    }

    #[test]
    fn bilinear_x2() {
        let x = Array3::from_shape_vec(
            [1, 3, 3],
            vec![
                0.3745, 0.9507, 0.7320, 0.5987, 0.1560, 0.1560, 0.0581, 0.8662, 0.6011,
            ],
        )
        .unwrap();

        let expected_y = Array3::from_shape_vec(
            [1, 6, 6],
            vec![
                0.3745, 0.6050, 0.8355, 0.9070, 0.8195, 0.7320, 0.4642, 0.5316, 0.5991, 0.6066,
                0.5541, 0.5016, 0.5538, 0.4583, 0.3627, 0.3062, 0.2887, 0.2712, 0.4905, 0.4135,
                0.3365, 0.2874, 0.2662, 0.2450, 0.2743, 0.3974, 0.5206, 0.5503, 0.4867, 0.4231,
                0.0581, 0.3813, 0.7046, 0.8132, 0.7071, 0.6011,
            ],
        )
        .unwrap();

        let bilinear = BilinearX2;
        let y = bilinear.apply(&x);

        for (expected, actual) in expected_y.iter().zip(y.iter()) {
            assert!((expected - actual).abs() < 1e-3);
        }
    }

    #[test]
    fn max_pool_x2() {
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
            [2, 2, 2],
            vec![
                0.9507, 0.8662, 0.8324, 0.9699, 0.6119, 0.4319, 0.7852, 0.6075,
            ],
        )
        .unwrap();

        let max_pool = MaxPoolX2;
        let y = max_pool.apply(&x);

        for (expected, actual) in expected_y.iter().zip(y.iter()) {
            println!("{} {}", expected, actual);
            assert!((expected - actual).abs() < 1e-3);
        }
    }
}
