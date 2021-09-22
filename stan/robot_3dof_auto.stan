data { 
	int<lower=0> N;
	int<lower=0> dof;
	matrix[dof, N] q;
	matrix[dof, N] dq;
	matrix[dof, N] ddq;
	matrix[dof, N] tau;
}
transformed data { 
	matrix[dof, N] sin_q = sin(q);
	matrix[dof, N] cos_q = cos(q);
		row_vector[N] x5 = 0.8*(dq[2,:].*(dq[1,:].*(sin_q[2,:]))) - 0.8*(ddq[1,:].*(cos_q[2,:]) - (dq[2,:].*(dq[1,:].*(sin_q[2,:]))));
	row_vector[N] x16 = dq[3,:].*((dq[1,:].*(sin_q[2,:])).*(-(sin_q[3,:])) + (cos_q[3,:]).*(dq[1,:].*(cos_q[2,:]))) + (ddq[1,:].*(sin_q[2,:]) + dq[2,:].*(dq[1,:].*(cos_q[2,:]))).*(cos_q[3,:]) + (ddq[1,:].*(cos_q[2,:]) - (dq[2,:].*(dq[1,:].*(sin_q[2,:])))).*(sin_q[3,:]);
	row_vector[N] x18 = dq[3,:].*(-((dq[1,:].*(sin_q[2,:])).*(cos_q[3,:]) + (dq[1,:].*(cos_q[2,:])).*(sin_q[3,:]))) - (ddq[1,:].*(sin_q[2,:]) + dq[2,:].*(dq[1,:].*(cos_q[2,:]))).*(sin_q[3,:]) + (ddq[1,:].*(cos_q[2,:]) - (dq[2,:].*(dq[1,:].*(sin_q[2,:])))).*(cos_q[3,:]);
	row_vector[N] x23 = -0.8*((dq[2,:]).*(dq[2,:])) + (9.81*(sin_q[2,:])) - 0.8*(((dq[1,:].*(cos_q[2,:]))).*((dq[1,:].*(cos_q[2,:]))));
	row_vector[N] x24 = (0.8*ddq[2,:] + 0.8*(dq[1,:].*(sin_q[2,:])).*(dq[1,:].*(cos_q[2,:])) + (9.81*(cos_q[2,:]))).*(cos_q[3,:]) + x23.*(-(sin_q[3,:]));
	row_vector[N] x26 = ((dq[1,:].*(sin_q[2,:])).*(-(sin_q[3,:])) + (cos_q[3,:]).*(dq[1,:].*(cos_q[2,:]))).*(dq[2,:] + dq[3,:]);
	row_vector[N] x27 = (dq[2,:] + dq[3,:]).*((dq[1,:].*(sin_q[2,:])).*(cos_q[3,:]) + (dq[1,:].*(cos_q[2,:])).*(sin_q[3,:]));
	row_vector[N] x28 = -((((dq[1,:].*(sin_q[2,:])).*(cos_q[3,:]) + (dq[1,:].*(cos_q[2,:])).*(sin_q[3,:]))).*(((dq[1,:].*(sin_q[2,:])).*(cos_q[3,:]) + (dq[1,:].*(cos_q[2,:])).*(sin_q[3,:]))));
	row_vector[N] x29 = -((((dq[1,:].*(sin_q[2,:])).*(-(sin_q[3,:])) + (cos_q[3,:]).*(dq[1,:].*(cos_q[2,:])))).*(((dq[1,:].*(sin_q[2,:])).*(-(sin_q[3,:])) + (cos_q[3,:]).*(dq[1,:].*(cos_q[2,:])))));
	row_vector[N] x33 = (0.8*ddq[2,:] + 0.8*(dq[1,:].*(sin_q[2,:])).*(dq[1,:].*(cos_q[2,:])) + (9.81*(cos_q[2,:]))).*(sin_q[3,:]) + x23.*(cos_q[3,:]);
	row_vector[N] x37 = ((dq[1,:].*(sin_q[2,:])).*(-(sin_q[3,:])) + (cos_q[3,:]).*(dq[1,:].*(cos_q[2,:]))).*((dq[1,:].*(sin_q[2,:])).*(cos_q[3,:]) + (dq[1,:].*(cos_q[2,:])).*(sin_q[3,:]));
	row_vector[N] x38 = -(((dq[2,:] + dq[3,:])).*((dq[2,:] + dq[3,:])));
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
	matrix[dof, N] tau_hat;
		row_vector[N] x13 = params[25]*((dq[1,:].*(sin_q[2,:])).*(cos_q[3,:]) + (dq[1,:].*(cos_q[2,:])).*(sin_q[3,:])) + params[27]*((dq[1,:].*(sin_q[2,:])).*(-(sin_q[3,:])) + (cos_q[3,:]).*(dq[1,:].*(cos_q[2,:]))) + params[28]*(dq[2,:] + dq[3,:]);
	row_vector[N] x14 = params[24]*((dq[1,:].*(sin_q[2,:])).*(cos_q[3,:]) + (dq[1,:].*(cos_q[2,:])).*(sin_q[3,:])) + params[26]*((dq[1,:].*(sin_q[2,:])).*(-(sin_q[3,:])) + (cos_q[3,:]).*(dq[1,:].*(cos_q[2,:]))) + params[27]*(dq[2,:] + dq[3,:]);
	row_vector[N] x25 = params[23]*x16 + params[24]*x18 + params[25]*(ddq[2,:] + ddq[3,:]) + params[30]*x5 - params[31]*x24 + ((dq[1,:].*(sin_q[2,:])).*(-(sin_q[3,:])) + (cos_q[3,:]).*(dq[1,:].*(cos_q[2,:]))).*x13 - (dq[2,:] + dq[3,:]).*x14;
	row_vector[N] x32 = params[23]*((dq[1,:].*(sin_q[2,:])).*(cos_q[3,:]) + (dq[1,:].*(cos_q[2,:])).*(sin_q[3,:])) + params[24]*((dq[1,:].*(sin_q[2,:])).*(-(sin_q[3,:])) + (cos_q[3,:]).*(dq[1,:].*(cos_q[2,:]))) + params[25]*(dq[2,:] + dq[3,:]);
	row_vector[N] x34 = params[24]*x16 + params[26]*x18 + params[27]*(ddq[2,:] + ddq[3,:]) - params[29]*x5 + params[31]*x33 + (dq[2,:] + dq[3,:]).*x32 + x13.*(-((dq[1,:].*(sin_q[2,:])).*(cos_q[3,:]) + (dq[1,:].*(cos_q[2,:])).*(sin_q[3,:])));
	row_vector[N] x36 = params[25]*x16 + params[27]*x18 + params[28]*(ddq[2,:] + ddq[3,:]) + params[29]*x24 - params[30]*x33 - ((dq[1,:].*(sin_q[2,:])).*(-(sin_q[3,:])) + (cos_q[3,:]).*(dq[1,:].*(cos_q[2,:]))).*x32 + ((dq[1,:].*(sin_q[2,:])).*(cos_q[3,:]) + (dq[1,:].*(cos_q[2,:])).*(sin_q[3,:])).*x14;
	tau_hat[1,:] = ddq[1,:]*params[6] + dq[1,:]*params[11] + (sin_q[2,:]).*(ddq[2,:]*params[14] - dq[2,:].*(dq[2,:]*params[16] + params[13]*(dq[1,:].*(sin_q[2,:])) + params[15]*(dq[1,:].*(cos_q[2,:]))) + params[12]*(ddq[1,:].*(sin_q[2,:]) + dq[2,:].*(dq[1,:].*(cos_q[2,:]))) + params[13]*(ddq[1,:].*(cos_q[2,:]) - (dq[2,:].*(dq[1,:].*(sin_q[2,:])))) - params[20]*(9.81*(cos_q[2,:])) + x25.*(cos_q[3,:]) + (dq[2,:]*params[17] + params[14]*(dq[1,:].*(sin_q[2,:])) + params[16]*(dq[1,:].*(cos_q[2,:]))).*(dq[1,:].*(cos_q[2,:])) + x34.*(-(sin_q[3,:]))) + (cos_q[2,:]).*(ddq[2,:]*params[16] + dq[2,:].*(dq[2,:]*params[14] + params[12]*(dq[1,:].*(sin_q[2,:])) + params[13]*(dq[1,:].*(cos_q[2,:]))) + params[13]*(ddq[1,:].*(sin_q[2,:]) + dq[2,:].*(dq[1,:].*(cos_q[2,:]))) + params[15]*(ddq[1,:].*(cos_q[2,:]) - (dq[2,:].*(dq[1,:].*(sin_q[2,:])))) + params[20]*(9.81*(sin_q[2,:])) - 0.8*params[29]*(-x18 + x27) - 0.8*params[30]*(x16 + x26) - 0.8*params[31]*(x28 + x29) - 0.8*params[32]*x5 - (dq[1,:].*(sin_q[2,:])).*(dq[2,:]*params[17] + params[14]*(dq[1,:].*(sin_q[2,:])) + params[16]*(dq[1,:].*(cos_q[2,:]))) + x25.*(sin_q[3,:]) + x34.*(cos_q[3,:]));
	tau_hat[2,:] = ddq[2,:]*params[17] + dq[2,:]*params[22] + params[14]*(ddq[1,:].*(sin_q[2,:]) + dq[2,:].*(dq[1,:].*(cos_q[2,:]))) + params[16]*(ddq[1,:].*(cos_q[2,:]) - (dq[2,:].*(dq[1,:].*(sin_q[2,:])))) + params[18]*(9.81*(cos_q[2,:])) - params[19]*(9.81*(sin_q[2,:])) + (dq[1,:].*(sin_q[2,:])).*(dq[2,:]*params[16] + params[13]*(dq[1,:].*(sin_q[2,:])) + params[15]*(dq[1,:].*(cos_q[2,:]))) - (dq[2,:]*params[14] + params[12]*(dq[1,:].*(sin_q[2,:])) + params[13]*(dq[1,:].*(cos_q[2,:]))).*(dq[1,:].*(cos_q[2,:])) + x36 + 0.8*(cos_q[3,:]).*(params[29]*((ddq[2,:] + ddq[3,:]) + x37) + params[30]*(x28 + x38) + params[31]*(-x16 + x26) + params[32]*x24) + 0.8*(sin_q[3,:]).*(params[29]*(x29 + x38) + params[30]*(-(ddq[2,:] + ddq[3,:]) + x37) + params[31]*(x18 + x27) + params[32]*x33);
	tau_hat[3,:] = dq[3,:]*params[33] + x36;
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
	tau[1, :] ~ normal(tau_hat[1, :], r);
	tau[2, :] ~ normal(tau_hat[2, :], r);
	tau[3, :] ~ normal(tau_hat[3, :], r);
}
