# ---
# jupyter:
#   jupytext:
#     formats: ipynb,R:light
#     text_representation:
#       extension: .R
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: R
#     language: R
#     name: ir
# ---

# # Pepper ANOVAs and post-hoc tests
#
# Author: Kate Doubleday
#
# Last updated: August 22, 2024

# Based on assumptions explored and decisions made in stats_exploration

library('readxl')
library('tidyverse')
library('dplyr')
library("writexl")
library("magrittr")
library("ggplot2")
library(lubridate)
library(rstatix)
library(car)
library(heplots)
library(broom)
library(emmeans)
library(ggpubr)
library(grid)

source("custom contrasts.R")
source("plot_functions.R")

# # Data prep

# Results and factor definitions
data_dir <- file.path(dirname(getwd()), "Processed data")
results_dir <- file.path(dirname(getwd()), "Results_graphs")
anova_dir <- file.path(dirname(getwd()), "ANOVA_Results")
ci_dir <- file.path(results_dir, "2016-2018 mean and 95per CI")
stats_dir <- file.path(ci_dir, "With stats")
nutrient_dir <- file.path(results_dir, "Nutrients")
weather_dir <- file.path(results_dir, "Weather")
misc_dir <- file.path(results_dir, "Data exploration and assumptions checks")
gap_order <- c("2", "3", "4", "5", "Control")
yrs <- c("2016", "2017", "2018")
fill_col <- "#0079C2" # NREL blue
nrel_cols <- c("#0079C2", "#F7A11A", "#5D9732", "#5E6A71", "#933C06")
location_order <- c("Under PV Array", "Control")

data_dir <- file.path(dirname(getwd()), "Processed data")

# Load in table titled df
load(file.path(data_dir, "pepper_2016-2018.Rda"))

# For plotting
dodge <- 0.4
pd <- position_dodge(dodge) # move them .4 to the left and right

t_dist <- FALSE

# # Peppers per plant ANOVA
# Same effects and assumption satisfaction are found with untransformed and square transformed data; similarly with and without outliers, so outliers are retained.

# Select dependent variable column
dep_var <- "peppers_per_plant"

dep_var <- rlang::ensym(dep_var)

res_aov <- anova_test(get({{dep_var}}) ~ gap_ft  * year,
  data = df,
    type = 3, 
    observed = c("year"), # This changes the generalized eta effect size
    #detailed = TRUE
)
residuals <- attributes(res_aov)$args$model$residuals
get_anova_table(res_aov, correction = "GG")

write.table(get_anova_table(res_aov, correction = "GG"), 
            file = file.path(anova_dir, "peppers_per_plant.csv"), sep=",", row.names=FALSE)

# ## Interaction plots

# +
# Preliminary interaction plot on transformed data
# This whole thing is equivalent to emmip(model, gap_ft ~ year)
means <- 
  df %>% 
  group_by(year, gap_ft) %>% # <- remember to group by *both* factors
  summarise(Means = mean({{dep_var}}))

ggplot(means, 
       aes(x = year, y = Means, colour = gap_ft, group = gap_ft)) +
  geom_point(size = 4) + geom_line() + 
    ylim(c(0, NA))
# -

# Not going to do a main effects because means are not ordinal (almost, for 2016-2017!)

# # Specific comparisons

# ## Control vs. pooled agrivoltaic comparisons
# Given that both treament and year are significant, compare: 
# - The average of means from the agPV groups to the mean of the control group, for each year
# - Year-to-year comparisons (including 2016-to-2018) for the control and for the pooled agPV groups

model <- lm(get({{dep_var}}) ~ gap_ft  * year, data = df)
# Due to balanced design (all cells have six sample sizes, these are actually the marginal means, not estimated marginal means
emm <- emmeans(model, c("gap_ft", "year"))

comp_df <- summary(contrast(emm, "all_pooled_comp", years = 3, combine = TRUE, adjust = "holm")) %>% add_significance(
  p.col = "p.value",
 cutpoints = c(0, 0.001, 0.01, 0.05, 1),
  symbols = c("***", "**", "*", "ns")
) 

comp_df

# +
if (t_dist) {
    ypos <- c(NA, NA, 12.5, 10, NA, 11.5, 10.5, 11.0, 12.0)
    } else {
    ypos <- 0.7 + c(NA, NA, 12.5, 10, NA, 11.5, 10.5, 11.0, 12.0)
    } 

comp_df %<>% mutate(y.position = ypos, 
                  xmin = c(NA, NA, 3 - dodge/4, 1 - dodge/4, NA, 2 - dodge/4, 1 + dodge/4, 1 + dodge/4, 2 + dodge/4),
                  xmax = c(NA, NA, 3 + dodge/4, 2 - dodge/4, NA, 3 - dodge/4, 2 + dodge/4, 3 + dodge/4, 3 + dodge/4))
# -

summary_stats <- get_treatment_year_error_bar_data(df, t_dist)

fig <- ggplot(df, aes(x = year, y = get(dep_var))) +
geom_point(data = summary_stats, inherit.aes = FALSE, 
           mapping = aes(x = year, y = means, color = treatment, fill = treatment),
           position = pd, size = 2) + 
geom_errorbar(data = summary_stats, inherit.aes = FALSE, 
              mapping = aes(x = year, ymin=means-error, ymax=means+error, 
                            color = treatment), 
              width=.1, position=pd, linewidth = 0.4) +
geom_jitter(aes(fill = treatment), position=position_jitterdodge(dodge.width = dodge, jitter.width = 0.1),
       size = 0.5, pch = 21, color = "black", stroke = 0.3) +
geom_bracket(aes(xmin = xmin, xmax = xmax, label = p.value.signif), data = comp_df[1:3, ], na.rm = TRUE,
             tip.length = 0.02, vjust = 0.4, inherit.aes = FALSE) +
geom_bracket(aes(xmin = xmin, xmax = xmax, label = p.value.signif, ), data = comp_df[4:6, ], na.rm = TRUE,
             tip.length = 0, vjust = 0.6, inherit.aes = FALSE, color = nrel_cols[1]) +
geom_bracket(aes(xmin = xmin, xmax = xmax, label = p.value.signif, ), data = comp_df[7:9, ], na.rm = TRUE,
             tip.length = 0, vjust = 0.6, inherit.aes = FALSE, color = nrel_cols[2]) +
scale_color_manual(values=nrel_cols) +
scale_fill_manual(values=nrel_cols, guide = "none") +
labs(x="Year", y = "Average Peppers Per Plant", color = "Treatment") +
scale_y_continuous(limits = c(0, NA), expand = expansion(mult = c(0, .05)),
                  breaks = seq(0, 12, by = 2)) +
theme(panel.grid.major.x = element_blank(),
      panel.grid.minor.y = element_blank(),
      legend.position="bottom", legend.margin=margin(t=-10),
     legend.title = element_text(size=10),
     legend.text = element_text(size=8))

fig

ggsave(file=file.path(stats_dir, "peppers_per_plant__ctrl_v_treatment.jpg"), width=4, height=3)

# + [markdown] jp-MarkdownHeadingCollapsed=true
# ## Side-note: Faceting by year
# -

temp <- comp_df[4:9,] %>% mutate(treatment = factor(c(rep("Under PV Array", times = 3), rep("Control", times = 3))),
                        group1 = factor(rep(c(2016, 2016, 2017), times = 2)),
                        group2 = factor(rep(c(2017, 2018, 2018), times = 2)),
                        .y. = "peppers_per_plant") %>%
                        rename(p.adj.signif = p.value.signif)

fig <- ggplot(df, aes(x = year, y = get(dep_var))) +
    facet_grid(~ treatment) + 
geom_point(data = summary_stats, inherit.aes = FALSE, 
           mapping = aes(x = year, y = means, color = year, fill = year),
          size = 2) + 
geom_errorbar(data = summary_stats, inherit.aes = FALSE, 
              mapping = aes(x = year, ymin=means-error, ymax=means+error, 
                            color = year), 
              width=.1, linewidth = 0.4) +
geom_jitter(aes(fill = treatment), position=position_jitterdodge(jitter.width = 0.1),
       size = 0.5, pch = 21, color = "black", stroke = 0.3) +
stat_pvalue_manual(temp, hide.ns = TRUE, label = "p.adj.signif", y.position = c( 10.5, 11, 10.5, 11, 11.5 ),
                  tip.length = 0) +
scale_color_manual(values=nrel_cols) +
scale_fill_manual(values=nrel_cols, guide = "none") +
labs(x="Year", y = "Average Peppers Per Plant", color = "Location", 
    title = "95% Confidence Interval of the Mean") +
ylim(0, NA) +
theme(panel.grid.major.x = element_blank(),
     legend.position="none")

fig

# + [markdown] jp-MarkdownHeadingCollapsed=true
# ## Side-note: Subset of pairwise comparisons
# Want to verify that there aren't insignificant differences between large inter-panel spacings (e.g., 5 ft.) and the control that we should be reporting (since those could be "brought down" by lower production by small inter-panel spacings, if there was a linear-ish relationship)
# - Findings: No such relationship found; these are consistent with the "pooled" findings above, and we should just report those
# -

pairwise_to_control.emmc <- function(levels, years, reverse = FALSE) {
    M <- data.frame(matrix(0, length(levels), (length(levels)/years-1)*years))
    row.names(M) <- levels
    # Within-year pooled AgPV - control comparison
    for (i in seq_len(years)){
        for (j in 1:(length(levels)/years-1)) {
            # Treatment
            M[j + (i - 1) * length(levels)/years, j + (i - 1) * (length(levels)/years-1)] <- -1
            # Control
            M[i * length(levels)/years, j + (i - 1) * (length(levels)/years-1)] <- 1
            names(M)[j + (i - 1) * (length(levels)/years-1)] <- 
                paste(levels[i * length(levels)/years], levels[j + (i - 1) * length(levels)/years], sep = " - ")
         }
    }
    M
}

contrast(emm, "pairwise_to_control", years = 3, combine = TRUE, adjust = "holm")

# ## Inter-panel gap trend analysis

# "the estimated linear contrast is not the slope of a line fitted to the data. It is simply a contrast having coefficients that increase linearly. It does test the linear trend, however."

poly_cont <- contrast(emm, "poly_excl_cont", exclude = "Control", by = "year", adjust = "holm")
poly_cont

if (t_dist) {
    fig <- plot_inter_panel_gap_trends__t(df, pd, "Average Peppers Per Plant")
    } else {
        fig <- plot_inter_panel_gap_trends__emmeans(df, emm, pd, "Average Peppers Per Plant")
    }

ann_text <- data.frame(gap_ft = factor(3), peppers_per_plant = 9.5,
                       year = factor(2017,levels = c(2016, 2017, 2018)))
fig + geom_text(data = ann_text, label = expression(paste("Linear ", italic("p"), " = 0.020")),
               size = 2.5)

ggsave(file=file.path(stats_dir, "peppers_per_plant__poly_contrasts.jpg"), width=4, height=3)

# ### Interaction contrasts
# A contrast of contrasts, calculating polynomial contrasts by gap_ft and comparing pairwise by year
# - Given that the polynomial contrasts themselves were not or barely significant (2017 only for swiss chard), this does not appear to be a useful add-on
# - But including for completeness 

contrast(emm, interaction = c(gap_ft = "poly_excl_cont", year = "pairwise"), exclude = "Control")

# ## Area comparisons

res_aov <- anova_test(get({{dep_var}}) ~ area * year,
  data = df,
    type = 3, 
    observed = c("year"), # This changes the generalized eta effect size
    #detailed = TRUE
)
residuals <- attributes(res_aov)$args$model$residuals
get_anova_table(res_aov, correction = "GG")

# # Fresh weight per pepper ANOVA
# Nothing significant except for year

# Select dependent variable column
dep_var <- "fw_per_pepper"

dep_var <- rlang::ensym(dep_var)

res_aov <- anova_test(get({{dep_var}}) ~ gap_ft  * year,
  data = df,
    type = 3, 
    observed = c("year"), # This changes the generalized eta effect size
    #detailed = TRUE
)
residuals <- attributes(res_aov)$args$model$residuals
get_anova_table(res_aov, correction = "GG")

write.table(get_anova_table(res_aov, correction = "GG"), 
            file = file.path(anova_dir, "fw_per_pepper.csv"), sep=",", row.names=FALSE)

# ## Pair-wise tests on year

# Use original 3-way model for error term
model <- lm(get({{dep_var}}) ~ gap_ft  * year, data = df)

fm <- as.formula(paste(dep_var, " ~ year"))

pairwise_df <- df %>% 
  emmeans_test(fm, 
                p.adjust.method = "holm",
              model = model) 

pairwise_df

pairwise_df %<>% add_xy_position(x = "year")
# Move second line down to save some space due to skipping ns line
pairwise_df$y.position[3] <- 265

if (t_dist) {
    summary_stats <- summarize(df, n = sum(!is.na(n_plants)),
      means = mean(get(dep_var), na.rm = TRUE),
      sd = sd(get(dep_var), na.rm = TRUE),
      error = qt(0.975,df=n-1)*sd/sqrt(n),
      .by = c(year))
} else {
    plot_model <- lm(get({{dep_var}}) ~ year, data = df)
     summary_stats <- summary(emmeans(plot_model, c("year"))) %>%
        rename(means = emmean) %>%
    mutate(error = upper.CL - lower.CL)
}

# +
pd <- position_dodge(0.3) # move them .03 to the left and right

fig <- ggplot(df, aes(x = year, y = get(dep_var))) +
geom_point(data = summary_stats, inherit.aes = FALSE, 
           mapping = aes(x = year, y = means), color = "black", fill = "black",
           position = pd, size = 2) + 
geom_errorbar(data = summary_stats, inherit.aes = FALSE, 
              mapping = aes(x = year, ymin=means-error, ymax=means+error),
                            color = "black", 
              width=.1, position=pd, linewidth = 0.4) +
geom_jitter(aes(fill = gap_ft, shape = gap_ft, color = gap_ft), position=position_jitter(width = 0.1),
       size = 0.7) +
stat_pvalue_manual(pairwise_df, hide.ns = TRUE, bracket.nudge.y = c(2, 2), label = "p.adj.signif") +
scale_fill_manual(values=nrel_cols) +
scale_color_manual(values=nrel_cols) +
scale_shape_manual(values = c(16, 17, 15, 18, 3)) + 
labs(x="Year", y = "Average Fresh Weight per Pepper (g)", fill = "Spacing (ft)", shape = "Spacing (ft)",
    color = "Spacing (ft)") +
scale_y_continuous(limits = c(0, NA), expand = expansion(mult = c(0, .05))) +
theme(panel.grid.major.x = element_blank(),
      # legend.position="bottom", legend.margin=margin(t=-10),
     legend.title = element_text(size=10),
     legend.text = element_text(size=8))
# -

fig

ggsave(file=file.path(stats_dir, "fw_per_pepper.jpg"), width=4, height=3)

# ## Area comparisons

res_aov <- anova_test(get({{dep_var}}) ~ area * year,
  data = df,
    type = 3, 
    observed = c("year"), # This changes the generalized eta effect size
    #detailed = TRUE
)
residuals <- attributes(res_aov)$args$model$residuals
get_anova_table(res_aov, correction = "GG")


