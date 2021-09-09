data { 
	int<lower=0> N;
	int<lower=0> dof;
	matrix[3, N] q;
	matrix[3, N] dq;
	matrix[3, N] ddq;
	matrix[3, N] tau;
}
transformed data { 
	matrix[dof, N] sin_q = sin(q);
	matrix[dof, N] cos_q = cos(q);
	row_vector[N] x0 = cos_q[2,:];
	row_vector[N] x1 = dq[1,:].*x0;
	row_vector[N] x2 = sin_q[2,:];
	row_vector[N] x3 = dq[1,:].*x2;
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
		row_vector[N] x16 = params[24]*(x1.*(sin_q[3,:]) + (cos_q[3,:]).*x3) + params[26]*(x1.*(cos_q[3,:]) + (-(sin_q[3,:])).*x3) + params[27]*(dq[2,:] + dq[3,:]);
	row_vector[N] x18 = dq[3,:].*(-(x1.*(sin_q[3,:]) + (cos_q[3,:]).*x3)) - (sin_q[3,:]).*(ddq[1,:].*x2 + dq[2,:].*x1) + (cos_q[3,:]).*(ddq[1,:].*x0 - (dq[2,:].*x3));
	row_vector[N] x21 = params[25]*(x1.*(sin_q[3,:]) + (cos_q[3,:]).*x3) + params[27]*(x1.*(cos_q[3,:]) + (-(sin_q[3,:])).*x3) + params[28]*(dq[2,:] + dq[3,:]);
	row_vector[N] x22 = dq[3,:].*(x1.*(cos_q[3,:]) + (-(sin_q[3,:])).*x3) + (sin_q[3,:]).*(ddq[1,:].*x0 - (dq[2,:].*x3)) + (cos_q[3,:]).*(ddq[1,:].*x2 + dq[2,:].*x1);
	row_vector[N] x26 = (cos_q[3,:]).*(0.8*ddq[2,:] + 0.8*x1.*x3 + (9.81*x0)) + (-(sin_q[3,:])).*(-0.8*((dq[2,:]).*(dq[2,:])) - 0.8*((x1).*(x1)) + (9.81*x2));
	row_vector[N] x27 = params[23]*x22 + params[24]*x18 + params[25]*(ddq[2,:] + ddq[3,:]) + params[30]*(0.8*(dq[2,:].*x3) - 0.8*(ddq[1,:].*x0 - (dq[2,:].*x3))) - params[31]*x26 + (x1.*(cos_q[3,:]) + (-(sin_q[3,:])).*x3).*x21 - (dq[2,:] + dq[3,:]).*x16;
	row_vector[N] x32 = (sin_q[3,:]).*(0.8*ddq[2,:] + 0.8*x1.*x3 + (9.81*x0)) + (cos_q[3,:]).*(-0.8*((dq[2,:]).*(dq[2,:])) - 0.8*((x1).*(x1)) + (9.81*x2));
	row_vector[N] x33 = params[23]*(x1.*(sin_q[3,:]) + (cos_q[3,:]).*x3) + params[24]*(x1.*(cos_q[3,:]) + (-(sin_q[3,:])).*x3) + params[25]*(dq[2,:] + dq[3,:]);
	row_vector[N] x34 = params[24]*x22 + params[26]*x18 + params[27]*(ddq[2,:] + ddq[3,:]) - params[29]*(0.8*(dq[2,:].*x3) - 0.8*(ddq[1,:].*x0 - (dq[2,:].*x3))) + params[31]*x32 + (dq[2,:] + dq[3,:]).*x33 + (-(x1.*(sin_q[3,:]) + (cos_q[3,:]).*x3)).*x21;
	row_vector[N] x38 = params[25]*x22 + params[27]*x18 + params[28]*(ddq[2,:] + ddq[3,:]) + params[29]*x26 - params[30]*x32 + (x1.*(sin_q[3,:]) + (cos_q[3,:]).*x3).*x16 - (x1.*(cos_q[3,:]) + (-(sin_q[3,:])).*x3).*x33;
	tau_hat[1,:] = ddq[1,:]*params[6] + dq[1,:]*params[11] + x0.*(ddq[2,:]*params[16] + dq[2,:].*(dq[2,:]*params[14] + params[12]*x3 + params[13]*x1) + params[13]*(ddq[1,:].*x2 + dq[2,:].*x1) + params[15]*(ddq[1,:].*x0 - (dq[2,:].*x3)) + params[20]*(9.81*x2) - 0.8*params[29]*(-x18 + ((x1.*(sin_q[3,:]) + (cos_q[3,:]).*x3).*(dq[2,:] + dq[3,:]))) - 0.8*params[30]*(x22 + ((x1.*(cos_q[3,:]) + (-(sin_q[3,:])).*x3).*(dq[2,:] + dq[3,:]))) - 0.8*params[31]*((-(((x1.*(sin_q[3,:]) + (cos_q[3,:]).*x3)).*((x1.*(sin_q[3,:]) + (cos_q[3,:]).*x3)))) + (-(((x1.*(cos_q[3,:]) + (-(sin_q[3,:])).*x3)).*((x1.*(cos_q[3,:]) + (-(sin_q[3,:])).*x3))))) - 0.8*params[32]*(0.8*(dq[2,:].*x3) - 0.8*(ddq[1,:].*x0 - (dq[2,:].*x3))) + (sin_q[3,:]).*x27 + (cos_q[3,:]).*x34 - x3.*(dq[2,:]*params[17] + params[14]*x3 + params[16]*x1)) + x2.*(ddq[2,:]*params[14] - dq[2,:].*(dq[2,:]*params[16] + params[13]*x3 + params[15]*x1) + params[12]*(ddq[1,:].*x2 + dq[2,:].*x1) + params[13]*(ddq[1,:].*x0 - (dq[2,:].*x3)) - params[20]*(9.81*x0) + x1.*(dq[2,:]*params[17] + params[14]*x3 + params[16]*x1) + (cos_q[3,:]).*x27 + (-(sin_q[3,:])).*x34);
	tau_hat[2,:] = ddq[2,:]*params[17] + dq[2,:]*params[22] + params[14]*(ddq[1,:].*x2 + dq[2,:].*x1) + params[16]*(ddq[1,:].*x0 - (dq[2,:].*x3)) + params[18]*(9.81*x0) - params[19]*(9.81*x2) - x1.*(dq[2,:]*params[14] + params[12]*x3 + params[13]*x1) + 0.8*(sin_q[3,:]).*(params[29]*((-(((x1.*(cos_q[3,:]) + (-(sin_q[3,:])).*x3)).*((x1.*(cos_q[3,:]) + (-(sin_q[3,:])).*x3)))) + (-(((dq[2,:] + dq[3,:])).*((dq[2,:] + dq[3,:]))))) + params[30]*(-(ddq[2,:] + ddq[3,:]) + ((x1.*(sin_q[3,:]) + (cos_q[3,:]).*x3).*(x1.*(cos_q[3,:]) + (-(sin_q[3,:])).*x3))) + params[31]*(x18 + ((x1.*(sin_q[3,:]) + (cos_q[3,:]).*x3).*(dq[2,:] + dq[3,:]))) + params[32]*x32) + 0.8*(cos_q[3,:]).*(params[29]*((ddq[2,:] + ddq[3,:]) + ((x1.*(sin_q[3,:]) + (cos_q[3,:]).*x3).*(x1.*(cos_q[3,:]) + (-(sin_q[3,:])).*x3))) + params[30]*((-(((x1.*(sin_q[3,:]) + (cos_q[3,:]).*x3)).*((x1.*(sin_q[3,:]) + (cos_q[3,:]).*x3)))) + (-(((dq[2,:] + dq[3,:])).*((dq[2,:] + dq[3,:]))))) + params[31]*(-x22 + ((x1.*(cos_q[3,:]) + (-(sin_q[3,:])).*x3).*(dq[2,:] + dq[3,:]))) + params[32]*x26) + x3.*(dq[2,:]*params[16] + params[13]*x3 + params[15]*x1) + x38;
	tau_hat[3,:] = dq[3,:]*params[33] + x38;
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
