data {
    int<lower=0> N;
    vector[N] x;
    vector[N] dx;
    vector[N] u;
}
parameters {
    real a;
    real<lower=0.0> r;
    real b;
}
model {
    // noise stds priors
    r ~ cauchy(0, 1.0);

    // prior on parameter
    a ~ cauchy(0, 1.0);
    b ~ cauchy(0, 1.0);


    // measurement likelihood
    dx - (-a * x + b * u) ~ normal(0.0, r);

}