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

# Calculate mean estimates of the parameters
beta0_hat <- mean(posterior_samples$beta0)
beta1_hat <- mean(posterior_samples$beta1)
beta2_hat <- mean(posterior_samples$beta2)

# Calculate estimated mean directions
mu_hat <- beta0_hat + beta1_hat * x1 + beta2_hat * x2
mu_hat <- atan(mu_hat)

# Plot true vs. estimated mean directions
png("true_vs_estimated_mean_directions.png")
ggplot(data.frame(True = mu, Estimated = mu_hat), aes(x = True, y = Estimated)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(title = "True vs. Estimated Mean Directions", x = "True Mean Direction", y = "Estimated Mean Direction") +
  theme_minimal()
dev.off()

######################


# Additional model diagnostic plots
library(bayesplot)

# Trace plots for MCMC chains
png("trace_plots.png")
traceplot(fit, pars = c("beta0", "beta1", "beta2", "kappa"))
dev.off()

# Density plots of the posterior distributions
png("density_plots.png")
mcmc_areas(as.array(fit), pars = c("beta0", "beta1", "beta2", "kappa"),
           prob = 0.8) + ggtitle("Posterior Distributions with 80% Intervals")
dev.off()

# Autocorrelation plots to check for mixing
png("autocorrelation_plots.png")
mcmc_acf(as.array(fit), pars = c("beta0", "beta1", "beta2", "kappa")) +
  ggtitle("Autocorrelation Plots")
dev.off()

# Pair plots for parameter relationships
png("pair_plots.png")
mcmc_pairs(as.array(fit), pars = c("beta0", "beta1", "beta2", "kappa")) +
  ggtitle("Pairs Plot of Parameters")
dev.off()

# Gelman-Rubin diagnostic
gelman_diag <- gelman.diag(fit)
print(gelman_diag)

# Summary of Gelman-Rubin diagnostic
cat("Gelman-Rubin diagnostic:\n")
print(gelman_diag)

# Effective sample sizes
ess <- effectiveSize(fit)
print(ess)

# Summary of effective sample sizes
cat("Effective sample sizes:\n")
print(ess)

# Summary of the fitted model
summary_fit <- summary(fit)
cat("Summary of the fitted model:\n")
print(summary_fit)
#########################################

# Density plots of the posterior distributions with truth values as red dotted lines
true_values <- c(beta[1], beta[2], beta[3], kappa_true)

png("density_plots_with_truth.png")
mcmc_areas(as.array(fit), pars = c("beta0", "beta1", "beta2", "kappa"),
           prob = 0.8) +
  ggtitle("Posterior Distributions with 80% Intervals") +
  geom_vline(aes(xintercept = true_values[1]), linetype = "dotted", color = "red") +
  geom_vline(aes(xintercept = true_values[2]), linetype = "dotted", color = "red") +
  geom_vline(aes(xintercept = true_values[3]), linetype = "dotted", color = "red") +
  geom_vline(aes(xintercept = true_values[4]), linetype = "dotted", color = "red")
dev.off()

# Density plots of the posterior distributions with true values
png("density_plots.png")
mcmc_areas(
  as.array(fit),
  pars = c("beta0", "beta1", "beta2", "kappa"),
  prob = 0.8
) +
  ggtitle("Posterior Distributions with 80% Intervals") +
  geom_vline(aes(xintercept = 0.5), color = "red", linetype = "dotted") +  # True value for beta0
  geom_vline(aes(xintercept = 1), color = "blue", linetype = "dotted") +   # True value for beta1
  geom_vline(aes(xintercept = -0.5), color = "green", linetype = "dotted") + # True value for beta2
  geom_vline(aes(xintercept = 2), color = "purple", linetype = "dotted")   # True value for kappa
dev.off()
########################

# Density plots of the posterior distributions with true values
png("density_plots.png")
mcmc_areas(
  as.array(fit),
  pars = c("beta0", "beta1", "beta2", "kappa"),
  prob = 0.8
) +
  ggtitle("Posterior Distributions with 80% Intervals") +
  geom_vline(aes(xintercept = 0.5), color = "red", linetype = "dotted", size = 1) +  # True value for beta0
  geom_vline(aes(xintercept = 1), color = "blue", linetype = "dotted", size = 1) +   # True value for beta1
  geom_vline(aes(xintercept = -0.5), color = "green", linetype = "dotted", size = 1) + # True value for beta2
  geom_vline(aes(xintercept = 2), color = "purple", linetype = "dotted", size = 1) +   # True value for kappa
  annotate("text", x = 0.5, y = Inf, label = "True beta0", color = "red", vjust = 2) +
  annotate("text", x = 1, y = Inf, label = "True beta1", color = "blue", vjust = 2) +
  annotate("text", x = -0.5, y = Inf, label = "True beta2", color = "green", vjust = 2) +
  annotate("text", x = 2, y = Inf, label = "True kappa", color = "purple", vjust = 2)
dev.off()


######################

# Density plots of the posterior distributions with true values, 95% credible intervals, and synchronized colors
library(ggplot2)
library(bayesplot)

# Extract the posterior samples
posterior_samples <- extract(fit)

# Convert to data frame for ggplot2
posterior_df <- as.data.frame(posterior_samples)

# Create a combined plot with density and truth lines
png("density_plots.png", width = 1200, height = 800)
p <- mcmc_areas(
  as.array(fit),
  pars = c("beta0", "beta1", "beta2", "kappa"),
  prob = 0.95,  # 95% credible intervals
  prob_outer = 0.95
) +
  ggtitle("Posterior Distributions with 95% Intervals and Truth Lines") +
  xlab("Parameter Value") +
  ylab("Density") +
  theme_minimal()

# Adding truth lines and labels
p <- p +
  geom_vline(aes(xintercept = 0.5), color = "red", linetype = "dotted", size = 1) +
  geom_text(aes(x = 0.5, y = 0.1, label = "True beta0 = 0.5"), color = "red", vjust = -1) +
  geom_vline(aes(xintercept = 1), color = "blue", linetype = "dotted", size = 1) +
  geom_text(aes(x = 1, y = 0.1, label = "True beta1 = 1"), color = "blue", vjust = -1) +
  geom_vline(aes(xintercept = -0.5), color = "green", linetype = "dotted", size = 1) +
  geom_text(aes(x = -0.5, y = 0.1, label = "True beta2 = -0.5"), color = "green", vjust = -1) +
  geom_vline(aes(xintercept = 2), color = "purple", linetype = "dotted", size = 1) +
  geom_text(aes(x = 2, y = 0.1, label = "True kappa = 2"), color = "purple", vjust = -1) +
  theme(legend.position = "bottom")

print(p)
dev.off()


# Density plots of the posterior distributions with true values and legends
png("density_plots_with_truth.png")
mcmc_areas(
  as.array(fit),
  pars = c("beta0", "beta1", "beta2", "kappa"),
  prob = 0.95
) +
  ggtitle("Posterior Distributions with 95% Intervals and True Values") +
  geom_vline(aes(xintercept = 0.5, color = "Beta0 True"), linetype = "dotted", lwd = 1) +  # True value for beta0
  geom_vline(aes(xintercept = 1, color = "Beta1 True"), linetype = "dotted", lwd = 1) +   # True value for beta1
  geom_vline(aes(xintercept = -0.5, color = "Beta2 True"), linetype = "dotted", lwd = 1) + # True value for beta2
  geom_vline(aes(xintercept = 2, color = "Kappa True"), linetype = "dotted", lwd = 1) +   # True value for kappa
  scale_color_manual(values = c("Beta0 True" = "red", "Beta1 True" = "blue", "Beta2 True" = "green", "Kappa True" = "purple")) +
  guides(color = guide_legend(title = "Parameter True Values")) +
  theme_minimal()
dev.off()

######
# Load necessary libraries
library(ggplot2)
library(bayesplot)

# Extract the posterior samples
posterior_samples <- extract(fit)

# Convert to data frame for ggplot2
posterior_df <- as.data.frame(posterior_samples)

# Create a combined plot with density and truth lines
png("density_plots.png", width = 1200, height = 800)
p <- mcmc_areas(
  as.array(fit),
  pars = c("beta0", "beta1", "beta2", "kappa"),
  prob = 0.95,  # 95% credible intervals
  prob_outer = 0.95
) +
  ggtitle("Posterior Distributions with 95% Intervals and Truth Lines") +
  xlab("Parameter Value") +
  ylab("Density") +
  theme_minimal()

# Adding truth lines and labels
p <- p +
  geom_vline(aes(xintercept = 0.5), color = "red", linetype = "dotted", size = 1) +
  geom_text(aes(x = 0.5, y = 0.15, label = "True beta0 = 0.5"), color = "red", vjust = -1) +
  geom_vline(aes(xintercept = 1), color = "blue", linetype = "dotted", size = 1) +
  geom_text(aes(x = 1, y = 0.15, label = "True beta1 = 1"), color = "blue", vjust = -1) +
  geom_vline(aes(xintercept = -0.5), color = "green", linetype = "dotted", size = 1) +
  geom_text(aes(x = -0.5, y = 0.15, label = "True beta2 = -0.5"), color = "green", vjust = -1) +
  geom_vline(aes(xintercept = 2), color = "purple", linetype = "dotted", size = 1) +
  geom_text(aes(x = 2, y = 0.15, label = "True kappa = 2"), color = "purple", vjust = -1) +
  theme(legend.position = "bottom")

print(p)
dev.off()
