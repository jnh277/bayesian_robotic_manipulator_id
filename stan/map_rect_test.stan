data {
    int <lower=0> N;
    int <lower=0> k;    # number of shards
    vector[N] x1;
    vector[N] x2;
    vector[N] y;
    vector[N] y2;

}

parameters {
    real theta;
    real<lower=1e-10>r;

}
model{
    r ~ cauchy(0, 1.);
    theta ~ cauchy(0, 1.);
    y ~ normal(theta*x1 + x2, r);
    y2 ~ normal(theta*x1 + x2, r);

}