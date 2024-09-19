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

# # Broccoli ANOVAs and post-hoc tests
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

source("custom contrasts.R")
source("plot_functions.R")

data_dir <- file.path(dirname(getwd()), "Processed data")

# Load in table titled df
load(file.path(data_dir, "broccoli_2017-2018.Rda"))

# For plotting
dodge <- 0.4
pd <- position_dodge(dodge) # move them .4 to the left and right

# Previously, had been using a t-distribution to estimate the confidence intervals due to information that it's better for small sample sizes. However, that's not consistent with the rest of the stats (ANOVA and post-hoc tests), which assume both normality and homogeneity of variance. In order to best represent what the stats are doing, want to switch from the t-distribution to a normal distribution, calculated through emmeans because it handily accounts for differing sample sizes. That does require re-defining the model based on the sub-categories desired by the post-hoc test, but I think that is likely either entirely correct or close enough for visualization purposes. 

t_dist <- FALSE

# # Head fresh weight per plant ANOVA
# Results are the same with and without the square root transform. Therefore, conducting analysis on the original data.

# +
# df %<>% mutate(head_fw_per_plant_sqrt = sqrt(head_fw_per_plant))
# -

# Select dependent variable column
dep_var <- "head_fw_per_plant"

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
            file = file.path(anova_dir, "broccoli_head_fw.csv"), sep=",", row.names=FALSE)

#  no follow up test on year as needed. With only two levels, 2018 is significantly higher.

# ## Interaction plots

# +
# Preliminary interaction plot on transformed data
# This whole thing is equivalent to emmip(model, gap_ft ~ year)
means <- 
  df %>% 
  group_by(year, gap_ft) %>% # <- remember to group by *both* factors
  summarise(Means = mean({{dep_var}}, na.rm=TRUE))

ggplot(means, 
       aes(x = year, y = Means, colour = gap_ft, group = gap_ft)) +
  geom_point(size = 4) + geom_line() + 
    ylim(c(0, NA))
# -

# This is almost ordinal. Looking at four And three, there is some crossing, but the uncertainty must be such that this is not significant. Will move forward on pairwise tests of gap for each of the two years.

# Year-to-year comparisons are not necessary. Need within year pooled agrivoltaic and control comparisons, and trend analysis only.

# # Specific comparisons

# ## Control vs. pooled agrivoltaic comparisons
# Given that both treament and year are significant, compare: 
# - The average of means from the agPV groups to the mean of the control group, for each year

model <- lm(get({{dep_var}}) ~ gap_ft  * year, data = df)
#  estimated marginal means
emm <- emmeans(model, c("gap_ft", "year"))

comp_df <- summary(contrast(emm, "within_year_pooled_comp", years = 2, combine = TRUE, adjust = "holm")) %>% add_significance(
  p.col = "p.value",
 cutpoints = c(0, 0.001, 0.01, 0.05, 1),
  symbols = c("***", "**", "*", "ns")
) 

comp_df

# +
if (t_dist) {
        ypos <- c(250, 580)
    } else {
        ypos <- c(325, 580)
    }

comp_df %<>% mutate(y.position = ypos, 
                  xmin = c(1 - dodge/4, 2 - dodge/4),
                  xmax = c(1 + dodge/4, 2 + dodge/4))
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
labs(x="Year", y = "Head Fresh Weight (g)", color = "Treatment") +
scale_y_continuous(limits = c(0, NA), expand = expansion(mult = c(0, .05))) +
theme(panel.grid.major.x = element_blank(),
      panel.grid.minor.y = element_blank(),
      legend.position="bottom", legend.margin=margin(t=-10),
     legend.title = element_text(size=10),
     legend.text = element_text(size=8))

fig

ggsave(file=file.path(stats_dir, "broccoli_head_fresh_weight__ctrl_v_treatment.jpg"), width=4, height=3)

# ## Inter-panel gap trend analysis

# "the estimated linear contrast is not the slope of a line fitted to the data. It is simply a contrast having coefficients that increase linearly. It does test the linear trend, however."
#
# -  no significant  trends found

poly_cont <- contrast(emm, "poly_excl_cont", exclude = "Control", by = "year", adjust = "holm")
poly_cont

emm

if (t_dist) {
    fig <- plot_inter_panel_gap_trends__t(df, pd, "Head Fresh Weight (g)")
    } else {
    fig <- plot_inter_panel_gap_trends__emmeans(df, emm, pd, "Head Fresh Weight (g)")
    }

fig

ggsave(file=file.path(stats_dir, "broccoli_head_weight__poly_contrasts.jpg"), width=4, height=3)

# ## Area comparisons

res_aov <- anova_test(get({{dep_var}}) ~ area * year,
  data = df,
    type = 3, 
    observed = c("year"), # This changes the generalized eta effect size
    #detailed = TRUE
)
residuals <- attributes(res_aov)$args$model$residuals
get_anova_table(res_aov, correction = "GG")

# # Fresh weight of  stems ANOVA

# Select dependent variable column
dep_var <- "stem_fw_per_plant"

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
            file = file.path(anova_dir, "broccoli_stem_fw.csv"), sep=",", row.names=FALSE)

# ## Interaction plots

# +
# Preliminary interaction plot on transformed data
# This whole thing is equivalent to emmip(model, gap_ft ~ year)
means <- 
  df %>% 
  group_by(year, gap_ft) %>% # <- remember to group by *both* factors
  summarise(Means = mean({{dep_var}}, na.rm=TRUE))

ggplot(means, 
       aes(x = year, y = Means, colour = gap_ft, group = gap_ft)) +
  geom_point(size = 4) + geom_line() + 
    ylim(c(0, NA))
# -

# ## Control vs. pooled agrivoltaic comparisons
# Given that both interaction and year are significant, compare: 
# - The average of means from the agPV groups to the mean of the control group, for each year
# - Year-to-year comparisons for the control and for the pooled agPV groups

model <- lm(get({{dep_var}}) ~ gap_ft  * year, data = df)
#  estimated marginal means
emm <- emmeans(model, c("gap_ft", "year"))

comp_df <- summary(contrast(emm, "all_pooled_comp", years = 2, combine = TRUE, adjust = "holm")) %>% add_significance(
  p.col = "p.value",
 cutpoints = c(0, 0.001, 0.01, 0.05, 1),
  symbols = c("***", "**", "*", "ns")
) 

comp_df %<>% mutate(y.position = c(NA, 950, 870, 910), 
                  xmin = c(NA, 2 - dodge/4, 1 - dodge/4, 1 + dodge/4),
                  xmax = c(NA, 2 + dodge/4, 2 - dodge/4, 2 + dodge/4))

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
geom_bracket(aes(xmin = xmin, xmax = xmax, label = p.value.signif), data = comp_df[1:2, ], na.rm = TRUE,
             tip.length = 0.02, vjust = 0.4, inherit.aes = FALSE) +
geom_bracket(aes(xmin = xmin, xmax = xmax, label = p.value.signif, ), data = comp_df[3, ], na.rm = TRUE,
             tip.length = 0, vjust = 0.6, inherit.aes = FALSE, color = nrel_cols[1]) +
geom_bracket(aes(xmin = xmin, xmax = xmax, label = p.value.signif, ), data = comp_df[4, ], na.rm = TRUE,
             tip.length = 0, vjust = 0.6, inherit.aes = FALSE, color = nrel_cols[2]) +
scale_color_manual(values=nrel_cols) +
scale_fill_manual(values=nrel_cols, guide = "none") +
labs(x="Year", y = "Stem Fresh Weight (g)", color = "Treatment") +
scale_y_continuous(expand = expansion(mult = c(0, .05))) +
coord_cartesian(ylim = c(0, NA)) + 
theme(panel.grid.major.x = element_blank(),
      panel.grid.minor.y = element_blank(),
      legend.position="bottom", legend.margin=margin(t=-10),
     legend.title = element_text(size=10),
     legend.text = element_text(size=8))

fig

ggsave(file=file.path(stats_dir, "broccoli_stem_fresh_weight__ctrl_v_treatment.jpg"), width=4, height=3)

# ## Inter-panel gap trend analysis

# "the estimated linear contrast is not the slope of a line fitted to the data. It is simply a contrast having coefficients that increase linearly. It does test the linear trend, however."
#
# -  No significant trends

poly_cont <- contrast(emm, "poly_excl_cont", exclude = "Control", by = "year", adjust = "holm")
poly_cont

if (t_dist) {
    fig <- plot_inter_panel_gap_trends__t(df, pd, "Stem Fresh Weight (g)")
    } else {
    fig <- plot_inter_panel_gap_trends__emmeans(df, emm, pd, "Stem Fresh Weight (g)")
    }

fig

ggsave(file=file.path(stats_dir, "broccoli_stem_weight__poly_contrasts.jpg"), width=4, height=3)

# ## Area comparisons

res_aov <- anova_test(get({{dep_var}}) ~ area * year,
  data = df,
    type = 3, 
    observed = c("year"), # This changes the generalized eta effect size
    #detailed = TRUE
)
residuals <- attributes(res_aov)$args$model$residuals
get_anova_table(res_aov, correction = "GG")


