# Load necessary libraries
library(rstan)
library(circular)
library(MASS)
library(ggplot2)

# Set seed for reproducibility
set.seed(123)

# Define the number of observations
n <- 100

# Generate predictor variables
x1 <- rnorm(n, mean = 0, sd = 1)
x2 <- rnorm(n, mean = 0, sd = 1)
X <- cbind(1, x1, x2)

# True parameter values
beta <- c(0.5, 1, -0.5)
kappa_true <- 2

# Generate mean direction using the linear predictor and arctan link function
mu <- atan(X %*% beta)

# Generate circular response variable from the von Mises distribution
theta <- rvonmises(n, circular(0), kappa_true) + mu
theta <- theta %% (2 * pi)

# Convert theta to a numeric vector
theta <- as.numeric(theta)

# Plot the generated data
png("simulated_circular_data.png")
plot.circular(circular(theta), stack = TRUE, bins = 100, main = "Simulated Circular Data")
dev.off()

# Stan model code for Bayesian circular regression using von Mises distribution
stan_code <- "
data {
  int<lower=0> n;
  vector[n] x1;
  vector[n] x2;
  vector[n] theta;
}
parameters {
  real beta0;
  real beta1;
  real beta2;
  real<lower=0> kappa;
}
model {
  vector[n] mu;
  beta0 ~ normal(0, 10);
  beta1 ~ normal(0, 10);
  beta2 ~ normal(0, 10);
  kappa ~ gamma(2, 0.1);
  for (i in 1:n) {
    mu[i] = atan(beta0 + beta1 * x1[i] + beta2 * x2[i]);
  }
  theta ~ von_mises(mu, kappa);
}
"

# Prepare data for Stan
data_list <- list(
  n = n,
  x1 = x1,
  x2 = x2,
  theta = theta
)

# Compile the Stan model
stan_model <- stan_model(model_code = stan_code)

# Fit the model using MCMC
fit <- sampling(stan_model, data = data_list, iter = 2000, chains = 4, seed = 123)

# Print summary of the fitted model
print(fit, pars = c("beta0", "beta1", "beta2", "kappa"))

# Extract posterior samples
posterior_samples <- extract(fit)

# Plot posterior distributions
png("posterior_distributions.png")
par(mfrow = c(2, 2))
hist(posterior_samples$beta0, main = expression(beta[0]), xlab = "Value", probability = TRUE, col = "lightblue", breaks = 30)
abline(v = beta[1], col = "red", lwd = 2)
hist(posterior_samples$beta1, main = expression(beta[1]), xlab = "Value", probability = TRUE, col = "lightblue", breaks = 30)
abline(v = beta[2], col = "red", lwd = 2)
hist(posterior_samples$beta2, main = expression(beta[2]), xlab = "Value", probability = TRUE, col = "lightblue", breaks = 30)
abline(v = beta[3], col = "red", lwd = 2)
hist(posterior_samples$kappa, main = expression(kappa), xlab = "Value", probability = TRUE, col = "lightblue", breaks = 30)
abline(v = kappa_true, col = "red", lwd = 2)
dev.off()

# Plot true vs. estimated mean directions
mu_hat <- colMeans(posterior_samples$beta0) + colMeans(posterior_samples$beta1) * x1 + colMeans(posterior_samples$beta2) * x2
mu_hat <- atan(mu_hat)

png("true_vs_estimated_mean_directions.png")
ggplot(data.frame(True = mu, Estimated = mu_hat), aes(x = True, y = Estimated)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(title = "True vs. Estimated Mean Directions", x = "True Mean Direction", y = "Estimated Mean Direction") +
  theme_minimal()
dev.off()
