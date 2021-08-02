data {
    int<lower=1> N;
    matrix[3,N] axis_vectors;
    matrix[3,N] y;
}

parameters {
    vector<lower=-pi(),upper=pi()>[3] euler;
    real<lower=1e-8> r;
}
transformed parameters{
    matrix[3,3] R;
    matrix[3,N] yhat;
    R[1,1] = cos(euler[2])*cos(euler[3]);
    R[1,2] = cos(euler[3])*sin(euler[1])*sin(euler[2]) - cos(euler[1])*sin(euler[3]);
    R[1,3] = sin(euler[1])*sin(euler[3]) + cos(euler[1])*cos(euler[3])*sin(euler[2]);
    R[2,1] = cos(euler[2])*sin(euler[3]);
    R[2,2] = cos(euler[1])*cos(euler[3]) + sin(euler[1])*sin(euler[2])*sin(euler[3]);
    R[2,3] = cos(euler[1])*sin(euler[2])*sin(euler[3]) - cos(euler[3])*sin(euler[1]);
    R[3,1] = -sin(euler[2]);
    R[3,2] = cos(euler[2])*sin(euler[1]);
    R[3,3] = cos(euler[1])*cos(euler[2]);
    yhat = R * axis_vectors;
}

model {
    r ~ cauchy(0,1.0);
    to_vector(y) ~ normal(to_vector(yhat), r);

}
