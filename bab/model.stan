data {
    int<lower=1> N1;                       // number of rows in data from group 1
    int<lower=1> N2;                       // number of rows in data from group 2
    real y1[N1];                           // outcome in group 1
    real y2[N2];                           // outcome in group 2
    int<lower=1> w1[N1];                   // frequency of outcome in group 1
    int<lower=1> w2[N2];                   // frequency of outcome in group 2
    real muM;                              // mean of y
    real<lower=0> muP;                     // default choice: 100 * sd_y
    real<lower=0> sigmaLow;                // default choice: sd_y / 1000
    real<lower=0> sigmaHigh;               // default choice: sd_y * 1000
    int<lower=0, upper=1> run_estimation;  // flag whether to sample for posterior or from prior
}

transformed data {
    int<lower=1> n1 = sum(w1);                       // number of samples from group 1
    int<lower=1> n2 = sum(w2);                       // number of samples from group 2
}

parameters {
    real<lower=0> nuMinusOne;
    real mu[2];
    real<lower=0> sigma[2];
}

transformed parameters {
    real nu;
    real log_nu; // include log-transformed parameter for plotting purposes
    nu = nuMinusOne + 1;
    log_nu = log(nu);
}

model {
    nuMinusOne ~ exponential(1./29.);

    if (run_estimation == 1) {
        for (i in 1:N1) target += w1[i] * student_t_lpdf(y1[i] | nu, mu[1], sigma[1]);
        for (i in 1:N2) target += w2[i] * student_t_lpdf(y2[i] | nu, mu[2], sigma[2]);
    }

    mu ~ normal(muM, muP);
    sigma ~ uniform(sigmaLow, sigmaHigh);
}

generated quantities {
    real y1_pred[n1];
    real y2_pred[n2];
    real log_lik[N1 + N2];

    for (j in 1:n1) y1_pred[j] = student_t_rng(nu, mu[1], sigma[1]);
    for (j in 1:N1) log_lik[j] = w1[j] * student_t_lpdf(y1[j] | nu, mu[1], sigma[1]);

    for (j in 1:n2) y2_pred[j] = student_t_rng(nu, mu[2], sigma[2]);
    for (j in 1:N2) log_lik[N1 + j] = w2[j] * student_t_lpdf(y2[j] | nu, mu[2], sigma[2]);
}
