data {
    int<lower=1> N;
    matrix[3,N] axis_vectors;
    matrix[3,N] y;
}

parameters {
    unit_vector[3] e1;
    vector[3] v2;
    real<lower=1e-8> r;
}
transformed parameters{
    matrix[3,3] R;
    vector[3] e2;
    matrix[3,N] yhat;
    e2 = v2 - dot_product(v2,e1)*e1;

    R[:,1] = e1;
    R[:,2] = e2;
    R[1,3] = e1[2]*e2[3] - e1[3]*e2[2];
    R[2,3] = e1[3]*e2[1] - e1[1]*e2[3];
    R[3,3] = e1[1]*e2[2] - e1[2]*e2[1];
    yhat = R * axis_vectors;
}

model {
    r ~ cauchy(0,1.0);
    to_vector(y) ~ normal(to_vector(yhat), r);

}
