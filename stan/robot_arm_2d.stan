data {
    int<lower=0> N;
    matrix[2, N] q;
    matrix[2, N] dq;
    matrix[2, N] ddq;
    matrix[2, N] tau;
    real g;
}
//transformed data {
//    row_vector[N] sq2 = sin(q[2, :]);
//    row_vector[N] cq1 = cos(q[1, :]);
//    row_vector[N] cq1q2 = cos(q[1, :] + q[2, :]);
//}
parameters {
    real <lower=1e-8> m1;
    real <lower=1e-8> m2;
    real <lower=1e-8> l1;
    real <lower=1e-8> l2;
    real <lower=0.0> d1;
    real <lower=0.0> d2;
    real <lower=1e-8> r;
    real <lower=0.01> nu;
//    vector[2] bias;
//    vector<lower=0.0>[2] hs_bias;
//    real<lower=0.0> hs_scale;
}

transformed parameters {
    real theta1 = l2^2 * m2 + l1^2 * (m1 + m2);
    real theta2 = l1 * l2 * m2;
    real theta3 = l2^2 * m2;
    real theta4 = l2 * m2;
    real theta5 = l1 * (m1 + m2);

    row_vector[N] m11 = theta1 + 2 * theta2 * cos(q[2, :]);
    row_vector[N] m12 = theta3 + theta2 * cos(q[2, :]);
    real m22 = theta3;
    row_vector[N] mdet = m11 * m22 - m12 .* m12;

    row_vector[N] u1 = g * theta4 * cos(q[1, :] + q[2, :]) + theta5 * g * cos(q[1, :]);
    row_vector[N] u2 = theta4 * g * cos(q[1, :] + q[2, :]);
    row_vector[N] c2 = -2 * theta2 * (dq[1, :] .* dq[1, :] .* sin(q[2, :])) - 2 * theta2 * dq[1, :] .* dq[2, :] .* sin(q[2, :]);

    row_vector[N] f1 = - u1 - d1 * dq[1] + tau[1, :];
    row_vector[N] f2 = c2 - u2 - d2 * dq[2] + tau[2, :];

    row_vector[N] ddq1hat = (m22 * f1 - m12 .* f2) ./ mdet;
    row_vector[N] ddq2hat = (m11 .* f2 - m12 .* f1) ./ mdet;

}

model {


    r ~ cauchy(0, 1.0);

//    hs_scale ~ cauchy(0., 1.);
//    hs_bias ~ cauchy(0., 1.);
//    bias ~ normal(0, hs_bias * hs_scale);

    nu ~ gamma(2,0.1);
    ddq[1, :] ~ student_t(nu, ddq1hat, r);
    ddq[2, :] ~ student_t(nu, ddq2hat, r);
//    ddq[1, :] ~ normal(ddq1hat, r);
//    ddq[2, :] ~ normal(ddq2hat, r);



}