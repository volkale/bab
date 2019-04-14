data {
 int<lower=0> N;             // number of samples per group
 real y1[N];                 // outcome in group 1
 real y2[N];                 // outcome in group 2
 real muM;                   // mean of y
 real<lower=0> muP;          // default choice: 100 * sd_y
 real<lower=0> sigmaLow;     // default choice: sd_y / 1000
 real<lower=0> sigmaHigh;    // default choice: sd_y * 1000
}

parameters {
 real<lower=0> nuMinusOne;
 real mu[2];
 real<lower=0> sigma[2];
}

transformed parameters {
  real nu;
  nu = nuMinusOne + 1;
}

model {
 nuMinusOne ~ exponential(1./29.);

 y1 ~ student_t(nu, mu[1], sigma[1]);
 y2 ~ student_t(nu, mu[2], sigma[2]);

 mu ~ normal(muM, muP);
 sigma ~ uniform(sigmaLow, sigmaHigh);
}

generated quantities {
   real y1_pred[N];
   real y2_pred[N];

   for (j in 1:N)
     y1_pred[j] = student_t_rng(nu, mu[1], sigma[1]);

   for (j in 1:N)
     y2_pred[j] = student_t_rng(nu, mu[2], sigma[2]);
}
