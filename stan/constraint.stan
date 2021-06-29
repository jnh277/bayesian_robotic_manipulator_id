data {
    int<lower=0> N;
}

parameters {
    real <lower=0.0> y2;
    real <lower=0.0> y3;
    real <lower=fmax(y3-y2,y2-y3), upper=y2+y3> y1;
    unit_vector[4] q;
}

transformed parameters {
    vector[3] y;
    matrix[3,3] R;
    matrix[3,3] Inertia;
    R[1,1] = q[1]*q[1] + q[2]*q[2] - q[3]*q[3] - q[4]*q[4]; R[1,2] = 2*(q[2]*q[3]-q[1]*q[4]); R[1,3] = 2*(q[1]*q[3]+q[2]*q[4]);
    R[2,1] = 2*(q[2]*q[3]+q[1]*q[4]); R[2,2] = q[1]*q[1] - q[2]*q[2] + q[3]*q[3] - q[4]*q[4]; R[2,3] = 2*(q[3]*q[4]-q[1]*q[2]);
    R[3,1] = 2*(q[2]*q[4]-q[1]*q[3]); R[3,2] = 2*(q[1]*q[2]+q[3]*q[4]); R[3,3] = q[1]*q[1] - q[2]*q[2] - q[3]*q[3] + q[4]*q[4];

    y[1] = y1; y[2] = y2; y[3] = y3;
    Inertia = diag_post_multiply(R,y) * R';
}

model {
    // some priors
    y1 ~ normal(0, 1);
    y2 ~ normal(0, 1);
    y3 ~ normal(0, 1);



}