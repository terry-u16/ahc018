use nalgebra::{DMatrix, DVector};

use crate::ChangeMinMax;

pub struct GaussianPredictor1d {
    x_list: Vec<f64>,
    y_list: Vec<f64>,
}

impl GaussianPredictor1d {
    pub fn new() -> Self {
        Self {
            x_list: vec![],
            y_list: vec![],
        }
    }

    pub fn add_data(&mut self, x: f64, y: f64) {
        self.x_list.push(x);
        self.y_list.push(y);
    }

    fn gaussian_process_regression(&self, x_pred: &DVector<f64>) -> (Vec<f64>, Vec<f64>) {
        let x_train = DVector::from_vec(self.x_list.clone());
        let y_train = DVector::from_vec(self.y_list.clone());
        let train_len = self.x_list.len();
        let pred_len = x_pred.len();

        let (t1, t2, t3) = Self::grid_search_theta(&x_train, &y_train);
        let kernel = Self::kernel_mat(&x_train, t1, t2, t3);
        let kernel_inv = kernel.clone().try_inverse().unwrap();

        let kernel_y = &kernel_inv * &y_train;
        let mut mean = vec![];
        let mut variance = vec![];

        for j in 0..pred_len {
            let mut kernel_vec = vec![];

            for i in 0..train_len {
                let k = Self::kernel(x_train[i], x_pred[j], t1, t2, t3, i, j + train_len);
                kernel_vec.push(k);
            }

            let kernel = DVector::from_vec(kernel_vec);
            let k = Self::kernel(
                x_pred[j],
                x_pred[j],
                t1,
                t2,
                t3,
                j + train_len,
                j + train_len,
            );

            mean.push(kernel.dot(&kernel_y));
            variance.push(k - (kernel_y.transpose() * (&kernel_inv * &kernel_y))[(0, 0)]);
        }

        (mean, variance)
    }

    fn kernel_mat(x: &DVector<f64>, t1: f64, t2: f64, t3: f64) -> DMatrix<f64> {
        let n = x.len();
        let mut kernel = DMatrix::zeros(n, n);

        for i in 0..n {
            for j in 0..n {
                kernel[(i, j)] = Self::kernel(x[i], x[j], t1, t2, t3, i, j)
            }
        }

        kernel
    }

    fn kernel(x0: f64, x1: f64, t1: f64, t2: f64, t3: f64, i: usize, j: usize) -> f64 {
        // 入力は1次元なのでx0, x1はスカラー量
        let diff = x0 - x1;
        let norm = diff * diff;
        let mut kernel = t1 * (-norm / t2).exp();
        if i == j {
            kernel += t3;
        }
        kernel
    }

    /// kernelの対数尤度を求める
    fn kernel_likelihood(kernel: &DMatrix<f64>, y: &DVector<f64>) -> f64 {
        let kernel_inv = kernel.clone();
        let kernel_inv = kernel_inv.try_inverse().unwrap();
        -kernel.determinant().max(1e-100).ln() - (y.transpose() * (kernel_inv * y))[(0, 0)]
    }

    fn grid_search_theta(x: &DVector<f64>, y: &DVector<f64>) -> (f64, f64, f64) {
        let mut best_t1 = std::f64::NAN;
        let mut best_t2 = std::f64::NAN;
        let mut best_t3 = std::f64::NAN;
        let mut best_liklihood = std::f64::MIN;

        for t1_pow in 4..8 {
            let t1 = (2.0f64).powi(t1_pow);

            for t2_pow in 3..6 {
                // 分散なので標準偏差の2乗
                let t2 = (2.0f64).powi(t2_pow);
                let t2 = t2 * t2;

                for t3_pow in -2..4 {
                    let t3 = (2.0f64).powi(t3_pow);
                    let kernel = Self::kernel_mat(x, t1, t2, t3);
                    let likelihood = Self::kernel_likelihood(&kernel, y);

                    if best_liklihood.change_max(likelihood) {
                        best_t1 = t1;
                        best_t2 = t2;
                        best_t3 = t3;
                    }
                }
            }
        }

        (best_t1, best_t2, best_t3)
    }
}
