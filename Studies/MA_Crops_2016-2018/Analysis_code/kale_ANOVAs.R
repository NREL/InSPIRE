# ---
# jupyter:
#   jupytext:
#     formats: ipynb,R:light
#     text_representation:
#       extension: .R
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: R
#     language: R
#     name: ir
# ---

# # Kale ANOVAs and post-hoc tests
#
# Author: Kate Doubleday
#
# Last updated: May 30, 2024

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
load(file.path(data_dir, "kale_2016-2018.Rda"))

# For plotting
dodge <- 0.4
pd <- position_dodge(dodge) # move them .4 to the left and right

t_dist <- FALSE

# # Fresh weight per leaf ANOVA

# Select dependent variable column
dep_var <- "fw_per_leaf"

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
            file = file.path(anova_dir, "kale_fw_per_leaf.csv"), sep=",", row.names=FALSE)

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

# # Specific comparisons

# ## Control vs. pooled agrivoltaic comparisons
# Given that treatment and its interaction with year are significant, compare: 
# - The average of means from the agPV groups to the mean of the control group, for each year

model <- lm(get({{dep_var}}) ~ gap_ft  * year, data = df)
#  estimated marginal means
emm <- emmeans(model, c("gap_ft", "year"))

comp_df <- summary(contrast(emm, "all_pooled_comp", years = 3, combine = TRUE, adjust = "holm")) %>% 
add_significance(
  p.col = "p.value",
 cutpoints = c(0, 0.001, 0.01, 0.05, 1),
  symbols = c("***", "**", "*", "ns")
) 

comp_df

# +
if (t_dist) {
        ypos <- c(NA, 37, 37, NA, NA, NA, 34.5, 35.5, NA)
    } else {
        ypos <- c(NA, 38, 38, NA, NA, NA, 34.5, 36, NA) + 3
    }

comp_df %<>% mutate(y.position = ypos, 
                  xmin = c(NA, 2 - dodge/4, 3 - dodge/4, NA, NA, NA, 1 + dodge/4, 1 + dodge/4, NA),
                  xmax = c(NA, 2 + dodge/4, 3 + dodge/4, NA, NA, NA, 2 + dodge/4, 3 + dodge/4, NA))
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
labs(x="Year", y = "Fresh Weight per Leaf (g)", color = "Treatment") +
scale_y_continuous(limits = c(0, NA), expand = expansion(mult = c(0, .05))) +
theme(panel.grid.major.x = element_blank(),
      panel.grid.minor.y = element_blank(),
      legend.position="bottom", legend.margin=margin(t=-10),
     legend.title = element_text(size=10),
     legend.text = element_text(size=8))

fig

ggsave(file=file.path(stats_dir, "kale_fresh_weight_per_leaf__ctrl_v_treatment.jpg"), width=4, height=3)

# ## Inter-panel gap trend analysis

# "the estimated linear contrast is not the slope of a line fitted to the data. It is simply a contrast having coefficients that increase linearly. It does test the linear trend, however."
#
# -  no significant  trends found

poly_cont <- contrast(emm, "poly_excl_cont", exclude = "Control", by = "year", adjust = "holm")
poly_cont

if (t_dist) {
        fig <- plot_inter_panel_gap_trends__t(df, pd, "Fresh Weight per Leaf (g)")
    } else {
        fig <- plot_inter_panel_gap_trends__emmeans(df, emm, pd, "Fresh Weight per Leaf (g)")
    }

fig

ggsave(file=file.path(stats_dir, "kale_fw_per_leaf__poly_contrasts.jpg"), width=4, height=3)

# + [markdown] jp-MarkdownHeadingCollapsed=true
# ## Area comparisons

# + jupyter={"source_hidden": true}
res_aov <- anova_test(get({{dep_var}}) ~ area * year,
  data = df,
    type = 3, 
    observed = c("year"), # This changes the generalized eta effect size
    #detailed = TRUE
)
residuals <- attributes(res_aov)$args$model$residuals
get_anova_table(res_aov, correction = "GG")
# -

# # Leaves per plant ANOVA

# Select dependent variable column
dep_var <- "leaves_per_plant"

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
            file = file.path(anova_dir, "kale_leaves_per_plant.csv"), sep=",", row.names=FALSE)

# No interactions; look separately at year and gap

# ## Pair-wise tests on year

# This is now redundant with the control vs. pooled agrivoltaic comparisons added below; after doing both, I don't think this is as intuitive as doing them with the breakdown

# Use original 3-way model for error term
model <- lm(get({{dep_var}}) ~ gap_ft  * year, data = df)

fm <- as.formula(paste(dep_var, " ~ year"))

pairwise_df <- df %>% 
  emmeans_test(fm, 
                p.adjust.method = "holm",
              model = model) 

pairwise_df

pairwise_df %<>% add_xy_position(x = "year")

pairwise_df

# Add a little more space between bars
pairwise_df$y.position <- c(41.5, 44.5, 47.5)

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
stat_pvalue_manual(pairwise_df, hide.ns = TRUE, bracket.nudge.y = -2, label = "p.adj.signif") +
scale_fill_manual(values=nrel_cols) +
scale_color_manual(values=nrel_cols) +
scale_shape_manual(values = c(16, 17, 15, 18, 3)) + 
labs(x="Year", y = "Leaves per Plant", fill = "Spacing (ft)", shape = "Spacing (ft)",
    color = "Spacing (ft)") +
scale_y_continuous(limits = c(0, NA), expand = expansion(mult = c(0, .05))) +
theme(panel.grid.major.x = element_blank(),
      # legend.position="bottom", legend.margin=margin(t=-10),
     legend.title = element_text(size=10),
     legend.text = element_text(size=8))

fig

ggsave(file=file.path(stats_dir, "kale_leaves_per_plant__by_year.jpg"), width=4, height=3)

# ## Control vs. pooled agrivoltaic comparisons
# Given that both year and gap are significant, compare: 
# - The average of means from the agPV groups to the mean of the control group, for each year
# (Looked briefly at just looking at the group-to-group difference across all 3 years, which might be most technically correct based on no significant interaction, but I don't think it actually addresses the question of interest, which is year-by-year)
# - This is a weaker test, but it does enable us to generalize about 2016 and 2018: Was there a difference in treatments consistent across crops? We can only comment if we test it.
# - Not testing same treatments across years; this is just focusing on the locational differences, with the caveat of subsetting by year 

model <- lm(get({{dep_var}}) ~ gap_ft * year, data = df)
emm <- emmeans(model, c("gap_ft", "year"))

# +
# This is just a number comparing two groups, no graphing is needed.

# comp_df <- summary(contrast(emm, "all_years_pooled_comp", years = 3, combine = TRUE, adjust = "holm")) %>% add_significance(
#   p.col = "p.value",
#  cutpoints = c(0, 0.001, 0.01, 0.05, 1),
#   symbols = c("***", "**", "*", "ns")
# ) 
# -

comp_df <- summary(contrast(emm, "within_year_pooled_comp", years = 3, combine = TRUE, adjust = "holm")) %>% add_significance(
  p.col = "p.value",
 cutpoints = c(0, 0.001, 0.01, 0.05, 1),
  symbols = c("***", "**", "*", "ns")
) 

comp_df

summary_stats <- get_treatment_year_error_bar_data(df, t_dist)

# +
if (t_dist) {
        ypos <- c(NA, 37, 43)
    } else {
        ypos <- c(NA, 37, 43)
    }

comp_df %<>% mutate(y.position = ypos, 
                  xmin = c(NA, 2 - dodge/4, 3 - dodge/4),
                  xmax = c(NA, 2 + dodge/4, 3 + dodge/4))
# -

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
scale_color_manual(values=nrel_cols) +
scale_fill_manual(values=nrel_cols, guide = "none") +
labs(x="Year", y = "Leaves per Plant", color = "Treatment") +
scale_y_continuous(limits = c(0, NA), expand = expansion(mult = c(0, .05))) +
theme(panel.grid.major.x = element_blank(),
      panel.grid.minor.y = element_blank(),
      legend.position="bottom", legend.margin=margin(t=-10),
     legend.title = element_text(size=10),
     legend.text = element_text(size=8))

fig

ggsave(file=file.path(stats_dir, "kale_leaves_per_plant__ctrl_v_treatment.jpg"), width=4, height=3)

# ## Inter-panel gap trend analysis

# "the estimated linear contrast is not the slope of a line fitted to the data. It is simply a contrast having coefficients that increase linearly. It does test the linear trend, however."
# - Verrrry marginal linear trend found. Combining all data across years due to lack of interaction

emm <- emmeans(model, c("gap_ft"))
poly_cont <- contrast(emm, "poly_excl_cont", exclude = "Control", adjust = "holm")
poly_cont

# This one is already in a normal distribution from emm

# +
stats_df <- summary(emm) # , adjust = "holm"
stats_df %<>% filter(gap_ft != "Control")

fig <- ggplot(filter(df, treatment != "Control"), 
                     aes(x = gap_ft, y = get(dep_var))) +
    geom_point(data = stats_df, inherit.aes = FALSE, 
               mapping = aes(x = gap_ft, y = emmean), color = "black", fill = "black",
               position = pd, size = 2) + 
    geom_errorbar(data = stats_df, inherit.aes = FALSE, 
                  mapping = aes(x = gap_ft, ymin=lower.CL, ymax=upper.CL, color = gap_ft), 
                  color = "black", width=.1, position=pd, linewidth = 0.4) +
    geom_jitter(aes(fill = year, color = year, shape = year), position=position_jitter(width = 0.1),
           size = 0.5, stroke = 0.3) +
    scale_color_manual(values=nrel_cols) +
    scale_fill_manual(values=nrel_cols) +
    scale_shape_manual(values = c(16, 17, 15)) + 
    labs(x="Inter-panel spacing (ft)", y = "Leaves per Plant", fill = "Year", shape = "Year", color = "Year") +
    scale_y_continuous(expand = expansion(mult = c(0, .05))) +
    coord_cartesian(ylim = c(0, NA)) + 
    theme( panel.grid.minor.y = element_blank())

ann_text <- data.frame(gap_ft = factor(3), leaves_per_plant = 35)
fig + geom_text(data = ann_text, label = expression(paste("Linear ", italic("p"), " = 0.045")),
               size = 2.5)
# -

ggsave(file=file.path(stats_dir, "kale_leaves_per_plant__poly_contrasts.jpg"), width=4, height=3)

# ## Area comparisons

res_aov <- anova_test(get({{dep_var}}) ~ area * year,
  data = df,
    type = 3, 
    observed = c("year"), # This changes the generalized eta effect size
    #detailed = TRUE
)
residuals <- attributes(res_aov)$args$model$residuals
get_anova_table(res_aov, correction = "GG")


