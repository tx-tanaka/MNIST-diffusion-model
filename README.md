# MNIST-diffusion-model
Demo


Let \(X_0\sim\rho_0\) be an arbitrary random vector in \(\mathbb R^n\), and define
\[
X_t=a_tX_0+\sigma_tZ,
\qquad Z\sim\mathcal N(0,I_n),
\]
where \(a_t=e^{-t/2}\), \(\sigma_t=\sqrt{1-e^{-t}}\), and \(Z\) is independent of \(X_0\).
(a) Derive the conditional score \(\nabla_x\log p_t(x\mid x_0)\).
(b) Prove that
\[
\nabla_x\log\rho_t(x)
\mathbb E[\nabla_x\log p_t(X_t\mid X_0)\mid X_t=x].
\]
(c) Show that the minimizer of
\[
\mathbb E\left[|f(X_t)-Z|^2\right]
\]
over all square-integrable measurable functions \(f\) is
\[
f^*(x)=-\sigma_t\nabla_x\log\rho_t(x).
\]
(d) Using the result in (c), explain how to estimate the score function from independent samples \(X_0^{(1)},\ldots,X_0^{(N)}\) without knowing \(\rho_0\).
(e) Explain how the estimated score can be used to simulate the reverse-time diffusion process in equation (4.118) of the lecture notes.