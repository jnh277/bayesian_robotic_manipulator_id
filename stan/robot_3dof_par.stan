functions {
  real partial_sum(real[,] tau_slice,
                   int start, int end,
                   row_vector params,
                   int dof,
                   real r,
                   matrix q,
                   matrix dq,
                   matrix ddq,
                   matrix sin_q,
                   matrix cos_q,
                   row_vector x0,row_vector x1,row_vector x2,row_vector x3,row_vector x4,row_vector x5,
                   row_vector x6,row_vector x7,row_vector x10,row_vector x11,
                   row_vector x12,row_vector x13,row_vector x14,row_vector x15,row_vector x16,row_vector x17,
                   row_vector x18,row_vector x19,row_vector x20,row_vector x21,row_vector x22,row_vector x23,
                   row_vector x24,row_vector x28,
                   row_vector x32,row_vector x33,row_vector x34,row_vector x35,
                   row_vector x36,row_vector x37
                   ) {
   matrix[dof, end-start+1] tau_hat;

    row_vector[end-start+1] x27 = params[24]*x23[start:end] + params[26]*x20[start:end] + params[27]*x10[start:end] - params[29]*x11[start:end] + params[31]*x17[start:end] + x19[start:end].*(params[25]*x18[start:end] + params[27]*x22[start:end] + params[28]*x24[start:end]) + x24[start:end].*(params[23]*x18[start:end] + params[24]*x22[start:end] + params[25]*x24[start:end]);
	row_vector[end-start+1] x30 = params[23]*x23[start:end] + params[24]*x20[start:end] + params[25]*x10[start:end] + params[30]*x11[start:end] - params[31]*x28[start:end] + x22[start:end].*(params[25]*x18[start:end] + params[27]*x22[start:end] + params[28]*x24[start:end]) - x24[start:end].*(params[24]*x18[start:end] + params[26]*x22[start:end] + params[27]*x24[start:end]);
	row_vector[end-start+1] x38 = params[25]*x23[start:end] + params[27]*x20[start:end] + params[28]*x10[start:end] + params[29]*x28[start:end] - params[30]*x17[start:end] + x18[start:end].*(params[24]*x18[start:end] + params[26]*x22[start:end] + params[27]*x24[start:end]) - x22[start:end].*(params[23]*x18[start:end] + params[24]*x22[start:end] + params[25]*x24[start:end]);
	tau_hat[1,:] = ddq[1,start:end]*params[6] + dq[1,start:end]*params[11] + x0[start:end].*(ddq[2,start:end]*params[16] + dq[2,start:end].*(dq[2,start:end]*params[14] + params[12]*x5[start:end] + params[13]*x3[start:end]) + params[13]*x4[start:end] + params[15]*x7[start:end] + params[20]*x15[start:end] - 0.8*params[29]*(-x20[start:end] + x35[start:end]) - 0.8*params[30]*(x23[start:end] + x32[start:end]) - 0.8*params[31]*(x33[start:end] + x34[start:end]) - 0.8*params[32]*x11[start:end] + x12[start:end].*x30 + x14[start:end].*x27 - x5[start:end].*(dq[2,start:end]*params[17] + params[14]*x5[start:end] + params[16]*x3[start:end])) + x2[start:end].*(ddq[2,start:end]*params[14] - dq[2,start:end].*(dq[2,start:end]*params[16] + params[13]*x5[start:end] + params[15]*x3[start:end]) + params[12]*x4[start:end] + params[13]*x7[start:end] - params[20]*x1[start:end] + x14[start:end].*x30 + x21[start:end].*x27 + x3[start:end].*(dq[2,start:end]*params[17] + params[14]*x5[start:end] + params[16]*x3[start:end]));
	tau_hat[2,:] = ddq[2,start:end]*params[17] + dq[2,start:end]*params[22] + params[14]*x4[start:end] + params[16]*x7[start:end] + params[18]*x1[start:end] - params[19]*x15[start:end] + 0.8*x12[start:end].*(params[29]*(x34[start:end] + x36[start:end]) + params[30]*(-x10[start:end] + x37[start:end]) + params[31]*(x20[start:end] + x35[start:end]) + params[32]*x17[start:end]) + 0.8*x14[start:end].*(params[29]*(x10[start:end] + x37[start:end]) + params[30]*(x33[start:end] + x36[start:end]) + params[31]*(-x23[start:end] + x32[start:end]) + params[32]*x28[start:end]) - x3[start:end].*(dq[2,start:end]*params[14] + params[12]*x5[start:end] + params[13]*x3[start:end]) + x38 + x5[start:end].*(dq[2,start:end]*params[16] + params[13]*x5[start:end] + params[15]*x3[start:end]);
	tau_hat[3,:] = dq[3,start:end]*params[33] + x38;

    real target_ = normal_lpdf(to_row_vector(tau_slice[:, 1]) | tau_hat[1, :], r);
    target_ += normal_lpdf(to_row_vector(tau_slice[:, 2]) | tau_hat[2, :], r);
    target_ += normal_lpdf(to_row_vector(tau_slice[:, 3]) | tau_hat[3, :], r);
    return target_;
  }
}

data {
	int<lower=0> N;
	int<lower=0> dof;
	matrix[3, N] q;
	matrix[3, N] dq;
	matrix[3, N] ddq;
	real tau[N, 3];
	int<lower=0> grainsize;
}
transformed data {
	matrix[dof, N] sin_q = sin(q);
	matrix[dof, N] cos_q = cos(q);
	row_vector[N] x0 = cos_q[2,:];
	row_vector[N] x1 = 9.81*x0;
	row_vector[N] x2 = sin_q[2,:];
	row_vector[N] x3 = dq[1,:].*x0;
	row_vector[N] x4 = ddq[1,:].*x2 + dq[2,:].*x3;
	row_vector[N] x5 = dq[1,:].*x2;
	row_vector[N] x6 = dq[2,:].*x5;
	row_vector[N] x7 = ddq[1,:].*x0 - x6;
	row_vector[N] x10 = ddq[2,:] + ddq[3,:];
	row_vector[N] x11 = 0.8*x6 - 0.8*x7;
	row_vector[N] x12 = sin_q[3,:];
	row_vector[N] x13 = 0.8*ddq[2,:] + x1 + 0.8*x3.*x5;
	row_vector[N] x14 = cos_q[3,:];
	row_vector[N] x15 = 9.81*x2;
	row_vector[N] x16 = -0.8*((dq[2,:]).*(dq[2,:])) + x15 - 0.8*((x3).*(x3));
	row_vector[N] x17 = x12.*x13 + x14.*x16;
	row_vector[N] x18 = x12.*x3 + x14.*x5;
	row_vector[N] x19 = -x18;
	row_vector[N] x20 = dq[3,:].*x19 - x12.*x4 + x14.*x7;
	row_vector[N] x21 = -x12;
	row_vector[N] x22 = x14.*x3 + x21.*x5;
	row_vector[N] x23 = dq[3,:].*x22 + x12.*x7 + x14.*x4;
	row_vector[N] x24 = dq[2,:] + dq[3,:];
	row_vector[N] x28 = x13.*x14 + x16.*x21;
	row_vector[N] x32 = x22.*x24;
	row_vector[N] x33 = -((x18).*(x18));
	row_vector[N] x34 = -((x22).*(x22));
	row_vector[N] x35 = x18.*x24;
	row_vector[N] x36 = -((x24).*(x24));
	row_vector[N] x37 = x18.*x22;
}
parameters {
	real<lower=1e-6> r;
	real<lower=1e-6> m[dof];
	vector[3] r_com[dof];
	unit_vector[4] quat[dof];
	real<lower=0.0> fv[dof];
	real <lower=0.0> eig12;
	real <lower=0.0> eig13;
	real <lower=fmax(eig13-eig12,eig12-eig13), upper=eig12+eig13> eig11;
	real <lower=0.0> eig22;
	real <lower=0.0> eig23;
	real <lower=fmax(eig23-eig22,eig22-eig23), upper=eig22+eig23> eig21;
	real <lower=0.0> eig32;
	real <lower=0.0> eig33;
	real <lower=fmax(eig33-eig32,eig32-eig33), upper=eig32+eig33> eig31;
}
transformed parameters {
	vector[3] eigs[dof];
	matrix[3,3] R[dof];
	matrix[3,3] I[dof];
	vector[3] l[dof];
	real Lxx[dof];	real Lyy[dof];	real Lzz[dof];	real Lxy[dof];	real Lxz[dof];	real Lyz[dof];		eigs[1,1] = eig11; eigs[1,2] = eig12; eigs[1,3] = eig13;
	eigs[2,1] = eig21; eigs[2,2] = eig22; eigs[2,3] = eig23;
	eigs[3,1] = eig31; eigs[3,2] = eig32; eigs[3,3] = eig33;
	for (d in 1:dof){
		R[d,1,1] = quat[d,1]*quat[d,1] + quat[d,2]*quat[d,2] - quat[d,3]*quat[d,3] - quat[d,4]*quat[d,4]; R[d,1,2] = 2*(quat[d,2]*quat[d,3]-quat[d,1]*quat[d,4]); R[d,1,3] = 2*(quat[d,1]*quat[d,3]+quat[d,2]*quat[d,4]);
		R[d,2,1] = 2*(quat[d,2]*quat[d,3]+quat[d,1]*quat[d,4]); R[d,2,2] = quat[d,1]*quat[d,1] - quat[d,2]*quat[d,2] + quat[d,3]*quat[d,3] - quat[d,4]*quat[d,4]; R[d,2,3] = 2*(quat[d,3]*quat[d,4]-quat[d,1]*quat[d,2]);
		R[d,3,1] = 2*(quat[d,2]*quat[d,4]-quat[d,1]*quat[d,3]); R[d,3,2] = 2*(quat[d,1]*quat[d,2]+quat[d,3]*quat[d,4]); R[d,3,3] = quat[d,1]*quat[d,1] - quat[d,2]*quat[d,2] - quat[d,3]*quat[d,3] + quat[d,4]*quat[d,4];
		I[d] = diag_post_multiply(R[d],eigs[d]) * R[d]';
		l[d] = r_com[d] * m[d];
		Lxx[d] = I[d,1,1] + m[d]*pow(r_com[d,2],2) + m[d]*pow(r_com[d,3],2);
		Lyy[d] = I[d,2,2] + m[d]*pow(r_com[d,1],2) + m[d]*pow(r_com[d,3],2);
		Lzz[d] = I[d,3,3] + m[d]*pow(r_com[d,1],2) + m[d]*pow(r_com[d,2],2);
		Lxy[d] = I[d,1,2] - m[d]*r_com[d,1]*r_com[d,2];
		Lxz[d] = I[d,1,3] - m[d]*r_com[d,1]*r_com[d,3];
		Lyz[d] = I[d,2,3] - m[d]*r_com[d,2]*r_com[d,3];
	}
	row_vector[33] params = [Lxx[1], Lxy[1], Lxz[1], Lyy[1], Lyz[1], Lzz[1], l[1,1], l[1,2], l[1,3], m[1], fv[1],
				Lxx[2], Lxy[2], Lxz[2], Lyy[2], Lyz[2], Lzz[2], l[2,1], l[2,2], l[2,3], m[2], fv[2],
				Lxx[3], Lxy[3], Lxz[3], Lyy[3], Lyz[3], Lzz[3], l[3,1], l[3,2], l[3,3], m[3], fv[3]];
}
model {
	r ~ cauchy(0, 1.0);
	for (d in 1:dof){
		r_com[d] ~ cauchy(0, 1.0);
		fv[d] ~ cauchy(0, 1.0);
	}
	eig11 ~ cauchy(0, 1);
	eig12 ~ cauchy(0, 1);
	eig13 ~ cauchy(0, 1);
	eig21 ~ cauchy(0, 1);
	eig22 ~ cauchy(0, 1);
	eig23 ~ cauchy(0, 1);
	eig31 ~ cauchy(0, 1);
	eig32 ~ cauchy(0, 1);
	eig33 ~ cauchy(0, 1);

//    target += partial_sum(tau[1:100,:], 1, 100, params, dof, r, q[:,1:100], dq[:,1:100], ddq[:,1:100], sin_q[:,1:100], cos_q[:,1:100], x0[1:100], x1[1:100], x2[1:100], x3[1:100],
//                    x4[1:100], x5[1:100], x6[1:100], x7[1:100], x10[1:100], x11[1:100], x12[1:100], x13[1:100], x14[1:100], x15[1:100], x16[1:100], x17[1:100], x18[1:100], x19[1:100], x20[1:100],
//                    x21[1:100], x22[1:100], x23[1:100], x24[1:100], x28[1:100], x32[1:100], x33[1:100], x34[1:100], x35[1:100], x36[1:100],
//                    x36[1:100]);
  target += reduce_sum(partial_sum, tau, grainsize, params, dof, r, q, dq, ddq, sin_q, cos_q, x0, x1, x2, x3,
                    x4, x5, x6, x7, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20,
                    x21, x22, x23, x24, x28, x32, x33, x34, x35, x36,
                    x36);

}
