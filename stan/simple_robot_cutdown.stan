data {
    int<lower=0> N;
    matrix[2, N] q;
    matrix[2, N] dq;
    matrix[2, N] ddq;
    matrix[2, N] tau;
    real a1;
}

parameters {
//# I_1zz,
//# r_1x, r_1y
//# m_1, m_2
//# fv_1, fv_2
//# r_2x, r_2y,
//# I_2zz
    real<lower=1e-8> I_1zz;
    real<lower=1e-8> I_2zz;
    real r_1x;
    real r_1y;
    real r_2x;
    real r_2y;
    real<lower=1e-6> m_1;
    real<lower=1e-6> m_2;
    real<lower=0.0> fv_1;
    real<lower=0.0> fv_2;
    real<lower=1e-6> r;     // measurement standard deviation
}

transformed parameters {
    real L_1xx, L_1yy, L_1zz, L_1xy, L_1xz, L_1yz;
    real L_2xx, L_2yy, L_2zz, L_2xy, L_2xz, L_2yz;
    real l_1x = r_1x * m_1;
    real l_1y = r_1y * m_1;
    real l_2x = r_2x * m_2;
    real l_2y = r_2y * m_2;

    L_1zz = I_1zz + m_1*pow(r_1x,2) + m_1*pow(r_1y,2);
    L_2zz = I_2zz + m_2*pow(r_2x,2) + m_2*pow(r_2y,2);


    row_vector[22] params = [0., 0., 0., 0., 0., L_1zz, l_1x, l_1y, 0., m_1, fv_1,
                        0., 0., 0., 0., 0., L_2zz, l_2x, l_2y, 0., m_2, fv_2];



}

model {
    matrix[2, N] tau_hat;
    row_vector[N] x0 = cos(q[2, :]);
    row_vector[N] x1 = ddq[1, :] + ddq[2, :];
    row_vector[N] x2 = -((dq[1, :] + dq[2, :]) .* (dq[1, :] + dq[2, :]));
    row_vector[N] x3 = 9.81 * sin(q[1, :]);
    row_vector[N] x4 = -a1*dq[1, :] .* dq[1,:] + x3;
    row_vector[N] x5 = sin(q[2, :]);
    row_vector[N] x6 = 9.81*cos(q[1, :]);
    row_vector[N] x7 = a1*ddq[1, :] + x6;
    row_vector[N] x8 = x0 .* x7 - x4 .* x5;
    row_vector[N] x9 = x0 .* x4 + x5 .* x7;
    row_vector[N] x10 = params[17] * x1 + params[18] * x8 - params[19] * x9;
    //
    tau_hat[1, :] = a1*(x0 .* (params[18]*x1 + params[19]*x2 + params[21]*x8) + x5 .* (params[18]*x2 - params[19]*x1 + params[21]*x9)) + ddq[1, :]*params[6] + dq[1, :]*params[11] + params[7]*x6 - params[8]*x3 + x10;
    tau_hat[2, :] = dq[2, :]*params[22] + x10;

    // priors
    r ~ cauchy(0, 1.0);
    I_1zz ~ cauchy(0, 1.0);
    I_2zz ~ cauchy(0, 1.0);
    r_1x ~ cauchy(0, 1.0);
    r_1y ~ cauchy(0, 1.0);
    r_2x ~ cauchy(0, 1.0);
    r_2y ~ cauchy(0, 1.0);
    fv_1 ~ cauchy(0, 1.0);
    fv_2 ~ cauchy(0, 1.0);

    //
    tau[1, :] ~ normal(tau_hat[1, :], r);
    tau[2, :] ~ normal(tau_hat[2, :], r);

}