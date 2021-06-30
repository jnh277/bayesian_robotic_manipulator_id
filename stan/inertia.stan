data {
    int<lower=0> N;
    matrix[3,N] tau;
    matrix[3,N] y;
}

parameters {
    real <lower=0.0> eig2;        // eigen values
    real <lower=0.0> eig3;
    real <lower=fmax(eig3-eig2,eig2-eig3), upper=eig2+eig3> eig1;
    unit_vector[4] q;
    real <lower=1e-8> r;
}

transformed parameters {
    vector[3] eigs;
    matrix[3,3] R;
    matrix[3,3] Inertia;
    matrix[3,N] yhat;
    R[1,1] = q[1]*q[1] + q[2]*q[2] - q[3]*q[3] - q[4]*q[4]; R[1,2] = 2*(q[2]*q[3]-q[1]*q[4]); R[1,3] = 2*(q[1]*q[3]+q[2]*q[4]);
    R[2,1] = 2*(q[2]*q[3]+q[1]*q[4]); R[2,2] = q[1]*q[1] - q[2]*q[2] + q[3]*q[3] - q[4]*q[4]; R[2,3] = 2*(q[3]*q[4]-q[1]*q[2]);
    R[3,1] = 2*(q[2]*q[4]-q[1]*q[3]); R[3,2] = 2*(q[1]*q[2]+q[3]*q[4]); R[3,3] = q[1]*q[1] - q[2]*q[2] - q[3]*q[3] + q[4]*q[4];

    eigs[1] = eig1; eigs[2] = eig2; eigs[3] = eig3;
    Inertia = diag_post_multiply(R,eigs) * R';

    yhat = mdivide_left_spd(Inertia,tau);
}

model {
    // some priors
    eig1 ~ cauchy(0, 1);
    eig2 ~ cauchy(0, 1);
    eig3 ~ cauchy(0, 1);

    to_vector(y) ~ normal(to_vector(yhat), r);

}