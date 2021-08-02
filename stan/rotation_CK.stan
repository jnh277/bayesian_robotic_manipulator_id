data {
    int<lower=1> N;
    matrix[3,N] axis_vectors;
    matrix[3,N] y;
}

parameters {
    vector[3] theta;
    real<lower=1e-8> r;
}
transformed parameters{
    matrix[3,3] S;
    matrix[3,3] R;
    matrix[3,N] yhat;
    matrix[3,3] Id = [[1., 0., 0.],[0., 1., 0.],[0.,0.,1.]];

    S[1,1] = 0.0;           S[1,2] = -theta[3];     S[1,3] = theta[2];
    S[2,1] = theta[3];      S[2,2] = 0.0;           S[2,3] = -theta[1];
    S[3,1] = -theta[2];     S[3,2] = theta[1];      S[3,3] = 0.0;

    R = (Id - S) / (Id + S);
    yhat = R * axis_vectors;
}

model {
    r ~ cauchy(0,1.0);
    to_vector(y) ~ normal(to_vector(yhat), r);

}
