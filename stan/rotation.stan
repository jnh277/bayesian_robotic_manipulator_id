data {
    int<lower=1> N;
    matrix[3,N] axis_vectors;
    matrix[3,N] y;
}

parameters {
    unit_vector[4] q;
    real<lower=1e-8> r;
}
transformed parameters{
    matrix[3,3] R;
    matrix[3,3] Inertia;
    matrix[3,N] yhat;
    R[1,1] = q[1]*q[1] + q[2]*q[2] - q[3]*q[3] - q[4]*q[4]; R[1,2] = 2*(q[2]*q[3]-q[1]*q[4]); R[1,3] = 2*(q[1]*q[3]+q[2]*q[4]);
    R[2,1] = 2*(q[2]*q[3]+q[1]*q[4]); R[2,2] = q[1]*q[1] - q[2]*q[2] + q[3]*q[3] - q[4]*q[4]; R[2,3] = 2*(q[3]*q[4]-q[1]*q[2]);
    R[3,1] = 2*(q[2]*q[4]-q[1]*q[3]); R[3,2] = 2*(q[1]*q[2]+q[3]*q[4]); R[3,3] = q[1]*q[1] - q[2]*q[2] - q[3]*q[3] + q[4]*q[4];
    yhat = R * axis_vectors;
}

model {
    r ~ cauchy(0,1.0);
    to_vector(y) ~ normal(to_vector(yhat), r);

}
