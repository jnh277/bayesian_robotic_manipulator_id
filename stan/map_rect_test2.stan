functions {
  real partial_sum(real[,] y_slice,
                        int start, int end,
                        vector x1,
                        vector x2,
                        real theta,
                        real r) {
    real target_ = normal_lpdf(to_vector(y_slice[:, 1])| theta*x1[start:end]+x2[start:end], r);
    target_ += normal_lpdf(to_vector(y_slice[:, 2])| x1[start:end]+theta*x2[start:end], r);
    return target_;
  }
}

data {
    int <lower=0> N;
    int <lower=0> k;    // number of shards
    vector[N] x1;
    vector[N] x2;
    real y_all[N, 2];
    int<lower=0> grainsize;

}

parameters {
    real theta;
    real<lower=1e-10>r;

}
model{
    r ~ cauchy(0, 1.);
    theta ~ cauchy(0, 1.);
//    y ~ normal(theta*x1 + x2, r);

  target += reduce_sum(partial_sum, y_all, grainsize,
                       x1, x2, theta, r);
}