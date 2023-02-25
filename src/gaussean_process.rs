use crate::ChangeMinMax;
use itertools::Itertools;
use nalgebra::{DMatrix, DVector};

#[derive(Debug, Clone)]
pub struct GaussianPredictor {
    x_list: Vec<DVector<f64>>,
    y_list: Vec<f64>,
    params: GaussianParam,
}

impl GaussianPredictor {
    pub fn new() -> Self {
        Self {
            x_list: vec![],
            y_list: vec![],
            params: GaussianParam::new(1.0, 1.0, 1.0),
        }
    }

    pub fn add_data(&mut self, x: DVector<f64>, y: f64) {
        self.x_list.push(x);
        self.y_list.push(y);
    }

    pub fn clear(&mut self) {
        self.x_list.clear();
        self.y_list.clear();
    }

    pub fn gaussian_process_regression(&self, x_pred: &DMatrix<f64>) -> (Vec<f64>, Vec<f64>) {
        let (x_train, y_train, y_average) = self.preprocess();

        let train_len = self.x_list.len();
        let pred_len = x_pred.shape().0;

        let kernel = self.kernel_mat(&x_train);
        let kernel_lu = kernel.lu();

        let kernel_y = kernel_lu.solve(&y_train).unwrap();
        let mut mean = vec![];
        let mut variance = vec![];

        for j in 0..pred_len {
            let mut kernel_vec = vec![];

            for i in 0..train_len {
                let xi = x_train.row(i).transpose();
                let xj = x_pred.row(j).transpose();
                let k = self.kernel(&xi, &xj, i, j + train_len);
                kernel_vec.push(k);
            }

            let kernel = DVector::from_vec(kernel_vec);
            let xj = x_pred.row(j).transpose();
            let k = self.kernel(&xj, &xj, j + train_len, j + train_len);

            mean.push(kernel.dot(&kernel_y));
            variance.push(k - (kernel_y.transpose() * kernel_lu.solve(&kernel_y).unwrap())[(0, 0)]);
        }

        Self::postprocess(mean, variance, y_average)
    }

    fn preprocess(&self) -> (DMatrix<f64>, DVector<f64>, f64) {
        let x_train = DMatrix::from_row_slice(
            self.x_list.len(),
            self.x_list[0].len(),
            &self.x_list.iter().flatten().copied().collect_vec(),
        );
        let mut y_train = DVector::from_vec(self.y_list.clone());

        // 平均を引く
        let y_average = y_train.mean();
        y_train.add_scalar_mut(-y_average);
        (x_train, y_train, y_average)
    }

    fn postprocess(mut mean: Vec<f64>, variance: Vec<f64>, average: f64) -> (Vec<f64>, Vec<f64>) {
        // 平均を足す
        for y in mean.iter_mut() {
            *y += average;
        }
        (mean, variance)
    }

    fn kernel_mat(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        let n = x.shape().0;
        let mut kernel = DMatrix::zeros(n, n);

        for i in 0..n {
            for j in 0..n {
                let x0 = x.row(i).transpose();
                let x1 = x.row(j).transpose();
                kernel[(i, j)] = self.kernel(&x0, &x1, i, j);
            }
        }

        kernel
    }

    fn kernel(&self, x0: &DVector<f64>, x1: &DVector<f64>, i: usize, j: usize) -> f64 {
        assert!(x0.shape().1 == 1);
        assert!(x1.shape().1 == 1);
        let diff = x0 - x1;
        let norm = diff.component_mul(&diff)[(0, 0)];
        let mut kernel = self.params.theta1 * (-norm / self.params.theta2).exp();
        if i == j {
            kernel += self.params.theta3;
        }
        kernel
    }

    /// kernelの対数尤度を求める
    fn kernel_likelihood(kernel: &DMatrix<f64>, y: &DVector<f64>) -> f64 {
        let kernel_inv = kernel.clone();
        let kernel_inv = kernel_inv.try_inverse().unwrap();
        -kernel.determinant().max(1e-100).ln() - (y.transpose() * (kernel_inv * y))[(0, 0)]
    }

    pub fn grid_search_theta(&mut self, t1_cands: &[f64], t2_cands: &[f64], t3_cands: &[f64]) {
        let (x, y, _) = self.preprocess();
        let mut best_t1 = self.params.theta1;
        let mut best_t2 = self.params.theta2;
        let mut best_t3 = self.params.theta3;
        let kernel = self.kernel_mat(&x);
        let mut best_liklihood = Self::kernel_likelihood(&kernel, &y);

        for &t1 in t1_cands {
            for &t2 in t2_cands {
                for &t3 in t3_cands {
                    self.params = GaussianParam::new(t1, t2, t3);

                    let kernel = self.kernel_mat(&x);
                    let likelihood = Self::kernel_likelihood(&kernel, &y);

                    if best_liklihood.change_max(likelihood) {
                        best_t1 = t1;
                        best_t2 = t2;
                        best_t3 = t3;
                    }
                }
            }
        }

        self.params = GaussianParam::new(best_t1, best_t2, best_t3);
    }
}

#[derive(Debug, Clone, Copy)]
struct GaussianParam {
    theta1: f64,
    theta2: f64,
    theta3: f64,
}

impl GaussianParam {
    fn new(theta1: f64, theta2: f64, theta3: f64) -> Self {
        Self {
            theta1,
            theta2,
            theta3,
        }
    }
}
