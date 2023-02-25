use crate::ChangeMinMax;
use itertools::Itertools;
use nalgebra::{DMatrix, DVector};

pub struct GaussianPredictor {
    x_list: Vec<DVector<f64>>,
    y_list: Vec<f64>,
}

impl GaussianPredictor {
    pub fn new() -> Self {
        Self {
            x_list: vec![],
            y_list: vec![],
        }
    }

    pub fn add_data(&mut self, x: DVector<f64>, y: f64) {
        self.x_list.push(x);
        self.y_list.push(y);
    }

    pub fn gaussian_process_regression(&self, x_pred: &DMatrix<f64>) -> (Vec<f64>, Vec<f64>) {
        let x_train = DMatrix::from_row_slice(
            self.x_list.len(),
            self.x_list[0].len(),
            &self.x_list.iter().flatten().copied().collect_vec(),
        );
        let mut y_train = DVector::from_vec(self.y_list.clone());

        // 平均を引く
        let y_average = y_train.mean();
        y_train.add_scalar_mut(-y_average);

        let train_len = self.x_list.len();
        let pred_len = x_pred.len();

        let (t1, t2, t3) = Self::grid_search_theta(&x_train, &y_train);
        let kernel = Self::kernel_mat(&x_train, t1, t2, t3);
        let kernel_lu = kernel.lu();

        let kernel_y = kernel_lu.solve(&y_train).unwrap();
        let mut mean = vec![];
        let mut variance = vec![];

        for j in 0..pred_len {
            let mut kernel_vec = vec![];

            for i in 0..train_len {
                let xi = x_train.row(i).transpose();
                let xj = x_pred.row(j).transpose();
                let k = Self::kernel(&xi, &xj, t1, t2, t3, i, j + train_len);
                kernel_vec.push(k);
            }

            let kernel = DVector::from_vec(kernel_vec);
            let xj = x_pred.row(j).transpose();
            let k = Self::kernel(&xj, &xj, t1, t2, t3, j + train_len, j + train_len);

            mean.push(kernel.dot(&kernel_y));
            variance.push(k - (kernel_y.transpose() * kernel_lu.solve(&kernel_y).unwrap())[(0, 0)]);
        }

        // 平均を引いていたのでその分足す
        for y in mean.iter_mut() {
            *y += y_average;
        }

        (mean, variance)
    }

    fn kernel_mat(x: &DMatrix<f64>, t1: f64, t2: f64, t3: f64) -> DMatrix<f64> {
        let n = x.len();
        let mut kernel = DMatrix::zeros(n, n);

        for i in 0..n {
            for j in 0..n {
                let x0 = x.row(i).transpose();
                let x1 = x.row(j).transpose();
                kernel[(i, j)] = Self::kernel(&x0, &x1, t1, t2, t3, i, j)
            }
        }

        kernel
    }

    fn kernel(
        x0: &DVector<f64>,
        x1: &DVector<f64>,
        t1: f64,
        t2: f64,
        t3: f64,
        i: usize,
        j: usize,
    ) -> f64 {
        assert!(x0.shape().0 == 1);
        assert!(x1.shape().0 == 1);
        let diff = x0 - x1;
        let norm = diff.component_mul(&diff)[(0, 0)];
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

    fn grid_search_theta(x: &DMatrix<f64>, y: &DVector<f64>) -> (f64, f64, f64) {
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

                for t3_pow in 0..4 {
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
