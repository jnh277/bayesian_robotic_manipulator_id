data {
    int<lower=0> N;
    matrix[2, N] q;
    matrix[2, N] dq;
    matrix[2, N] ddq;
    matrix[2, N] tau;
    real a1;
}

parameters {
    cov_matrix[3] I_1;
    cov_matrix[3] I_2;
    real<lower=1e-6> m_1;
    real<lower=1e-6> m_2;
    vector[3] r_1;
    vector[3] r_2;
    real<lower=0.0> fv_1;
    real<lower=0.0> fv_2;
    real<lower=1e-6> r;     // measurement standard deviation
}

transformed parameters {
    // need l_1, l_2, L_1, L_2
    real L_1xx, L_1yy, L_1zz, L_1xy, L_1xz, L_1yz;
    real L_2xx, L_2yy, L_2zz, L_2xy, L_2xz, L_2yz;
    vector[3] l_1 = r_1 * m_1;
    vector[3] l_2 = r_2 * m_2;


    L_1xx = I_1[1,1] + m_1*pow(r_1[2],2) + m_1*pow(r_1[3],2);
    L_1yy = I_1[2,2] + m_1*pow(r_1[1],2) + m_1*pow(r_1[3],2);
    L_1zz = I_1[3,3] + m_1*pow(r_1[1],2) + m_1*pow(r_1[2],2);
    L_1xy = I_1[1,2] - m_1*r_1[1]*r_1[2];
    L_1xz = I_1[1,3] - m_1*r_1[1]*r_1[3];
    L_1yz = I_1[2,3] - m_1*r_1[2]*r_1[3];

    L_2xx = I_2[1,1] + m_2*pow(r_2[2],2) + m_2*pow(r_2[3],2);
    L_2yy = I_2[2,2] + m_2*pow(r_2[1],2) + m_2*pow(r_2[3],2);
    L_2zz = I_2[3,3] + m_2*pow(r_2[1],2) + m_2*pow(r_2[2],2);
    L_2xy = I_2[1,2] - m_2*r_2[1]*r_2[2];
    L_2xz = I_2[1,3] - m_2*r_2[1]*r_2[3];
    L_2yz = I_2[2,3] - m_2*r_2[2]*r_2[3];

    row_vector[22] params = [L_1xx, L_1xy, L_1xz, L_1yy, L_1yz, L_1zz, l_1[1], l_1[2], l_1[3], m_1, fv_1,
                        L_2xx, L_2xy, L_2xz, L_2yy, L_2yz, L_2zz, l_2[1], l_2[2], l_2[3], m_2, fv_2];



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
  tau_hat[1, :] = a1*(x0 .* (params[18]*x1 + params[19]*x2 + params[21]*x8) + x5 .* (params[18]*x2 - params[19]*x1 + params[21]*x9)) + ddq[1]*params[6] + dq[1]*params[11] + params[7]*x6 - params[8]*x3 + x10;
  tau_hat[2, :] = dq[2]*params[22] + x10;

// priors
  r ~ cauchy(0,1.0);

//
  tau[1, :] ~ normal(tau_hat[1, :], r);
  tau[2, :] ~ normal(tau_hat[2, :], r);

}