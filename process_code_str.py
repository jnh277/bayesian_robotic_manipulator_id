import re

def c_to_stan(code_str, dof, num_params):
    code_str = code_str.replace("  ", "\t")     # replace double spaces with tabs
    for d in range(dof-1,-1,-1):
        code_str = code_str.replace("q["+str(d)+"]", "q["+str(d+1)+",:]")
        code_str = code_str.replace("dq[" + str(d) + "]", "ddq[" + str(d + 1) + ",:]")
        code_str = code_str.replace("dq[" + str(d) + "]", "ddq[" + str(d + 1) + ",:]")
        code_str = code_str.replace("tau_out[" + str(d) + "]", "tau_hat[" + str(d + 1) + ",:]")

    idx = code_str.index("double x0")
    # code_str = code_str[:idx] + "matrix["+str(dof)+",N] tau_hat; \n  " + code_str[idx:]
    code_str = "\tmatrix[" + str(dof) + ",N] tau_hat; \n  " + code_str[idx:]
    code_str = code_str.replace("double", "row_vector[N]")

    for i in range(num_params-1,-1,-1):
        code_str = code_str.replace("parms["+str(i)+"]", "params["+str(i+1)+"]")

    sub_strs = re.findall('x\d*\*x\d*|x\d*\*\(|x\d*\)\*',code_str)
    for sub_str in sub_strs:
        new_str = sub_str.replace('*','.*')
        code_str = code_str.replace(sub_str, new_str)

    sub_strs = re.findall('x\d*\*q\[\d*,:\]|q\[\d*,:\]\*x\d|q\[\d*,:\]\*\(|q\[\d*,:\]\*q\[\d*,:\]|q\[\d*,:\]\*\)',code_str)
    for sub_str in sub_strs:
        new_str = sub_str.replace('*','.*')
        code_str = code_str.replace(sub_str, new_str)

    sub_strs = re.findall('x\d*\*d*q\[\d*,:\]|d*q\[\d*,:\]\*x\d|d*q\[\d*,:\]\*\(|d*q\[\d*,:\]\)\*|d*q\[\d*,:\]\*q\[\d*,:\]|d*q\[\d*,:\]\*d*q\[\d*,:\]|q\[\d*,:\]\*d*q\[\d*,:\]',code_str)
    for sub_str in sub_strs:
        new_str = sub_str.replace('*','.*')
        code_str = code_str.replace(sub_str, new_str)

    idx = code_str.index("return;")
    return code_str[:idx]


def create_param_block(dof):
    code_str = "parameters { \n"
    code_str += "\treal<lower=1e-6> r;\n"
    for d in range(1,dof+1):
        code_str += "\treal<lower=1e-6> m_"+str(d)+";\n"
        code_str += "\tvector[3] r_"+str(d)+";\n"
        code_str += "\treal<lower=0.0> fv_"+str(d)+";\n"
        code_str += "\treal <lower=0.0> eig"+str(d)+"2;\n"
        code_str += "\treal <lower=0.0> eig"+str(d)+"3;\n"
        code_str += "\treal <lower=fmax(eig"+str(d)+"3-eig"+str(d)+"2,eig"+str(d)+"2-eig"+str(d)+"3), upper=eig"+str(d)+"2+eig"+str(d)+"3> eig" + str(d) + "1;\n"
        code_str += "\tunit_vector[4] quat"+str(d)+";\n"
    code_str += "}\n"
    return code_str

def create_trans_params_block(dof, num_params):
    code_str = "transformed parameters {\n"
    for d in range(1,dof+1):
        code_str += "\tvector[3] eigs"+str(d)+";\n"
        code_str += "\tmatrix[3,3] R"+str(d)+";\n"
        code_str += "\tmatrix[3,3] I_"+str(d)+";\n"
        code_str += "\tR"+str(d)+"[1,1] = quat"+str(d)+"[1]*quat"+str(d)+"[1] + quat"+str(d)+"[2]*quat"+str(d)+"[2] - quat"+str(d)+"[3]*quat"+str(d)+"[3] - quat"+str(d)+"[4]*quat"+str(d)+"[4]; R"+str(d)+"[1,2] = 2*(quat"+str(d)+"[2]*quat"+str(d)+"[3]-quat"+str(d)+"[1]*quat"+str(d)+"[4]); R"+str(d)+"[1,3] = 2*(quat"+str(d)+"[1]*quat"+str(d)+"[3]+quat"+str(d)+"[2]*quat"+str(d)+"[4]);\n"
        code_str += "\tR"+str(d)+"[2,1] = 2*(quat"+str(d)+"[2]*quat"+str(d)+"[3]+quat"+str(d)+"[1]*quat"+str(d)+"[4]); R"+str(d)+"[2,2] = quat"+str(d)+"[1]*quat"+str(d)+"[1] - quat"+str(d)+"[2]*quat"+str(d)+"[2] + quat"+str(d)+"[3]*quat"+str(d)+"[3] - quat"+str(d)+"[4]*quat"+str(d)+"[4]; R"+str(d)+"[2,3] = 2*(quat"+str(d)+"[3]*quat"+str(d)+"[4]-quat"+str(d)+"[1]*quat"+str(d)+"[2]);\n"
        code_str += "\tR"+str(d)+"[3,1] = 2*(quat"+str(d)+"[2]*quat"+str(d)+"[4]-quat"+str(d)+"[1]*quat"+str(d)+"[3]); R"+str(d)+"[3,2] = 2*(quat"+str(d)+"[1]*quat"+str(d)+"[2]+quat"+str(d)+"[3]*quat"+str(d)+"[4]); R"+str(d)+"[3,3] = quat"+str(d)+"[1]*quat"+str(d)+"[1] - quat"+str(d)+"[2]*quat"+str(d)+"[2] - quat"+str(d)+"[3]*quat"+str(d)+"[3] + quat"+str(d)+"[4]*quat"+str(d)+"[4];\n"
        code_str += "\teigs"+str(d)+"[1] = eig"+str(d)+"1; eigs"+str(d)+"[2] = eig"+str(d)+"2; eigs"+str(d)+"[3] = eig"+str(d)+"3;\n"
        code_str += "\tI_"+str(d)+" = diag_post_multiply(R"+str(d)+",eigs"+str(d)+") * R"+str(d)+"';\n"
        code_str += "\treal L_"+str(d)+"xx, L_"+str(d)+"yy, L_"+str(d)+"zz, L_"+str(d)+"xy, L_"+str(d)+"xz, L_"+str(d)+"yz;\n"
        code_str += "\tvector[3] l_"+str(d)+" = r_"+str(d)+" * m_"+str(d)+";\n"
        code_str += "\tL_"+str(d)+"xx = I_"+str(d)+"[1,1] + m_"+str(d)+"*pow(r_"+str(d)+"[2],2) + m_"+str(d)+"*pow(r_"+str(d)+"[3],2);\n"
        code_str += "\tL_"+str(d)+"yy = I_"+str(d)+"[2,2] + m_"+str(d)+"*pow(r_"+str(d)+"[1],2) + m_"+str(d)+"*pow(r_"+str(d)+"[3],2);\n"
        code_str += "\tL_"+str(d)+"zz = I_"+str(d)+"[3,3] + m_"+str(d)+"*pow(r_"+str(d)+"[1],2) + m_"+str(d)+"*pow(r_"+str(d)+"[2],2);\n"
        code_str += "\tL_"+str(d)+"xy = I_"+str(d)+"[1,2] - m_"+str(d)+"*r_"+str(d)+"[1]*r_"+str(d)+"[2];\n"
        code_str += "\tL_"+str(d)+"xz = I_"+str(d)+"[1,3] - m_"+str(d)+"*r_"+str(d)+"[1]*r_"+str(d)+"[3];\n"
        code_str += "\tL_"+str(d)+"yz = I_"+str(d)+"[2,3] - m_"+str(d)+"*r_"+str(d)+"[2]*r_"+str(d)+"[3];\n"
    code_str += "\trow_vector["+str(num_params)+"] params = ["
    for d in range(1,dof+1):
        code_str += "L_"+str(d)+"xx, L_"+str(d)+"xy, L_"+str(d)+"xz, L_"+str(d)+"yy, L_"+str(d)+"yz, L_"+str(d)+"zz, l_"+str(d)+"[1], l_"+str(d)+"[2], l_"+str(d)+"[3], m_"+str(d)+", fv_"+str(d)
        if d != dof:
            code_str += ", \n\t\t\t\t"
    code_str += '];\n'

    code_str +="}\n"
    return code_str

def create_model_block(dof, num_params, c_code_str):
    code_str = "model {\n"
    code_str += c_to_stan(c_code_str, dof, num_params)
    code_str += "\tr ~ cauchy(0, 1.0);\n"
    for d in range(1,dof+1):
        code_str += "\teig"+str(d)+"1 ~ cauchy(0, 1);\n"
        code_str += "\teig"+str(d)+"2 ~ cauchy(0, 1);\n"
        code_str += "\teig"+str(d)+"3 ~ cauchy(0, 1);\n"
        code_str += "\tr_"+str(d)+" ~ cauchy(0, 1.0);\n"
        code_str += "\tfv_"+str(d)+" ~ cauchy(0, 1.0);\n"

    for d in range(1, dof+1):
        code_str += "\ttau["+str(d)+", :] ~ normal(tau_hat["+str(d)+", :], r);\n"
    code_str += "}"

    return code_str

dof = 3
print(create_trans_params_block(dof, 22))

print(create_param_block(dof))