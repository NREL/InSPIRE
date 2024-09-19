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

library('readxl')
library('tidyverse')
library('dplyr')
library("writexl")
library("magrittr")
library("ggplot2")
library("ggpubr")
library("cowplot")

# Results and factor definitions
data_dir <- file.path(dirname(getwd()), "Processed data")
results_dir <- file.path(dirname(getwd()), "Results_graphs")
ci_dir <- file.path(results_dir, "2016-2018 mean and 95per CI")
box_dir <- file.path(results_dir, "2016-2018 box plot")
nutrient_dir <- file.path(results_dir, "Nutrients")
weather_dir <- file.path(results_dir, "Weather")
misc_dir <- file.path(results_dir, "Data exploration and assumptions checks")
gap_order <- c("2", "3", "4", "5", "Control")
yrs <- c("2016", "2017", "2018")
fill_col <- "#0079C2" # NREL blue
nrel_cols <- c("#0079C2", "#F7A11A", "#5D9732", "#5E6A71", "#933C06")
location_order <- c("Under PV Array", "Control")

groupings <- c("gap_ft", "area")
grouping_names <- c("Inter-Panel Spacing (ft)", "Plot Location Relative to Panel Hub")
grouping_names_short <- c("Spacing (ft)", "Orientation")
grouping_dirs <- c("By inter-panel spacing", "By area")

# Directory handling
if (!dir.exists(results_dir)){
  dir.create(results_dir)
}
for (yr in yrs) {
    if (!dir.exists(file.path(results_dir, yr))){
      dir.create(file.path(results_dir, yr))
    }
}
if (!dir.exists(ci_dir)){
      dir.create(ci_dir)
}
if (!dir.exists(box_dir)){
      dir.create(box_dir)
}
for (g in c("By inter-panel spacing", "By area", "By Year")) {
    if (!dir.exists(file.path(ci_dir, g))){
      dir.create(file.path(ci_dir, g))
    }
    if (!dir.exists(file.path(box_dir, g))){
      dir.create(file.path(box_dir, g))
    }
}

# # Define plots

# Plot of mean and 95% confidence interval of the mean
# Calculate 95% confidence interval assuming t distribution (better than normal distribution for small sample sizes)
three_year_ci_plot <- function(df, crop_name, 
                    grouping_dir, grouping, grouping_label, 
                    metric, metric_label, fill = "year", fill_label = "Year",
                              title_detail = "", save = TRUE) {
    
    # Remove control if looking by east/center/west location
    if (grouping == "area" | fill == "area") {
        df %<>% filter(gap_ft != "Control")
    }
    
    summary_stats <- summarize(df, n = sum(!is.na(n_plants)), 
          avg_metric = mean(get(metric), na.rm = TRUE),
          sd = sd(get(metric), na.rm = TRUE),
          se = sd / sqrt(n),
          error = qt(0.975,df=n-1)*sd/sqrt(n),
          .by = c(fill, grouping))
    
    pd <- position_dodge(0.5) # move them .05 to the left and right
    
    fig <- ggplot(df, aes(x = get(grouping), y = get(metric), fill = get(fill))) +
    geom_point(data = summary_stats, inherit.aes = FALSE, 
               mapping = aes(x = get(grouping), y=avg_metric, color = get(fill)),
               position = pd, size = 2, na.rm = TRUE) + 
    geom_errorbar(data = summary_stats, inherit.aes = FALSE, 
                  mapping = aes(x = get(grouping), ymin=avg_metric-error, ymax=avg_metric+error, 
                                color = get(fill)), 
                  width=.1, position=pd, linewidth = 0.4, na.rm = TRUE) +
    geom_jitter(position=position_jitterdodge(dodge.width = 0.5, jitter.width = 0.1),
           size = 0.5, pch = 21, color = "black", stroke = 0.3, na.rm = TRUE) +
    scale_color_manual(values=nrel_cols) +
    scale_fill_manual(values=nrel_cols, guide = "none") +
    labs(x=grouping_label, y = metric_label, color = fill_label, fill = fill_label,
        title = "95% Confidence Interval of the Mean") +
    scale_y_continuous(limits = c(0, NA), expand = expansion(mult = c(0, .05)))
    
    if (grouping == "gap_ft") {
        fig <- fig + scale_x_discrete(limits = gap_order)
    }


    if (save) {
        ggsave(file=file.path(ci_dir, grouping_dir, paste(paste(crop_name, metric_label, title_detail), ".jpg", sep = "")), 
            width=4, height=3)
        } else {
        return(fig)
    }
}

# Plot of mean and 95% confidence interval of the mean
# Calculate 95% confidence interval assuming t distribution (better than normal distribution for small sample sizes)
control_vs_array_ci_plot <- function(df, crop_name, 
                    grouping_dir,
                    metric, metric_label) {
    df %<>% mutate(area = factor(ifelse(gap_ft == "Control", "Control", "Under PV Array"),
                                levels = location_order))
    summary_stats <- summarize(df, n = sum(!is.na(n_plants)), 
              avg_metric = mean(get(metric), na.rm = TRUE),
              sd = sd(get(metric), na.rm = TRUE),
              se = sd / sqrt(n),
              error = qt(0.975,df=n-1)*sd/sqrt(n),
              .by = c(year, area))
    
    pd <- position_dodge(0.3) # move them .03 to the left and right
    
    fig <- ggplot(df, aes(x = year, y = get(metric), fill = area)) +
    geom_point(data = summary_stats, inherit.aes = FALSE, 
               mapping = aes(x = year, y=avg_metric, color = area),
               position = pd, size = 2, na.rm = TRUE) + 
    geom_errorbar(data = summary_stats, inherit.aes = FALSE, 
                  mapping = aes(x = year, ymin=avg_metric-error, ymax=avg_metric+error, 
                                color = area), 
                  width=.1, position=pd, linewidth = 0.4, na.rm = TRUE) +
    geom_jitter(position=position_jitterdodge(dodge.width = 0.3, jitter.width = 0.1),
           size = 0.5, pch = 21, color = "black", stroke = 0.3, na.rm = TRUE) +
    scale_color_manual(values=nrel_cols) +
    scale_fill_manual(values=nrel_cols, guide = "none") +
    labs(x="Year", y = metric_label, color = "Location", 
        title = "95% Confidence Interval of the Mean") +
    scale_y_continuous(limits = c(0, NA), expand = expansion(mult = c(0, .05)))

    ggsave(file=file.path(ci_dir, grouping_dir, paste(crop_name, metric_label, ".jpg")), width=4, height=3)
}

# Plot samples over box plots
three_year_box_plot <- function(df, crop_name, 
                    grouping_dir, grouping, grouping_label, 
                    metric, metric_label, fill = "year", fill_label = "Year",
                              title_detail = "", save = TRUE) {
    
    # Remove control if looking by east/center/west location
    if (grouping == "area" | fill == "area") {
        df %<>% filter(gap_ft != "Control")
    }
    
    summary_stats <- summarize(df, n = sum(!is.na(n_plants)), 
          avg_metric = mean(get(metric), na.rm = TRUE),
          sd = sd(get(metric), na.rm = TRUE),
          se = sd / sqrt(n),
          error = qt(0.975,df=n-1)*sd/sqrt(n),
          .by = c(fill, grouping))
    
    pd <- position_dodge(0.5) # move them .05 to the left and right
    
    fig <- ggplot(df, aes(x = get(grouping), y = get(metric), fill = get(fill))) +
    facet_grid(~ year) + 
    geom_boxplot(linewidth = 0.4, na.rm = TRUE) + 
        # geom_boxplot(width=.3, position=pd, linewidth = 0.4, na.rm = TRUE) + 
    # geom_jitter(position=position_jitterdodge(dodge.width = 0.5, jitter.width = 0.1),
    #        size = 0.5, pch = 21, color = "black", stroke = 0.3, na.rm = TRUE) +
    geom_jitter(position=position_jitter(width = 0.1),
           size = 0.5, pch = 21, color = "black", stroke = 0.3, na.rm = TRUE) +
    scale_color_manual(values=nrel_cols, guide = "none") +
    scale_fill_manual(values=nrel_cols, guide = "none") +
    labs(x=grouping_label, y = metric_label) +
    scale_y_continuous(limits = c(0, NA), expand = expansion(mult = c(0, .05)))
    
    if (grouping == "gap_ft") {
        fig <- fig + scale_x_discrete(limits = gap_order)
    }

    if (save) {
        ggsave(file=file.path(box_dir, grouping_dir, paste(paste(crop_name, metric_label, title_detail), ".jpg", sep = "")), 
            width=4, height=3)
        } else {
        return(fig)
    }
}

control_vs_array_box_plot <- function(df, crop_name, 
                    grouping_dir,
                    metric, metric_label) {
    df %<>% mutate(area = factor(ifelse(gap_ft == "Control", "Control", "Under PV Array"),
                                levels = location_order))
    summary_stats <- summarize(df, n = sum(!is.na(n_plants)), 
              avg_metric = mean(get(metric), na.rm = TRUE),
              sd = sd(get(metric), na.rm = TRUE),
              se = sd / sqrt(n),
              error = qt(0.975,df=n-1)*sd/sqrt(n),
              .by = c(year, area))
    
    pd <- position_dodge(0.5) # move them .03 to the left and right
    
    fig <- ggplot(df, aes(x = year, y = get(metric), fill = area)) +
    geom_boxplot(width=.3, position=pd, linewidth = 0.4, na.rm = TRUE) + 
    geom_jitter(position=position_jitterdodge(dodge.width = 0.5, jitter.width = 0.1),
           size = 0.5, pch = 21, color = "black", stroke = 0.3, na.rm = TRUE) +
    scale_color_manual(values=nrel_cols) +
    scale_fill_manual(values=nrel_cols) + 
    labs(x="Year", y = metric_label, color = "Location", fill = "Location") +
    scale_y_continuous(limits = c(0, NA), expand = expansion(mult = c(0, .05)))

    ggsave(file=file.path(box_dir, grouping_dir, paste(crop_name, metric_label, ".jpg")), width=4, height=3)
}

nutrient_ci_plot <- function(df, crop_name, 
                    grouping_dir, grouping, grouping_label, 
                    metric, metric_label) {
    summary_stats <- summarize(df, n = sum(!is.na(get(metric))), 
          avg_metric = mean(get(metric), na.rm = TRUE),
          sd = sd(get(metric), na.rm = TRUE),
          se = sd / sqrt(n),
          error = qt(0.975,df=n-1)*sd/sqrt(n),
          .by = c(grouping))
    
    fig <- ggplot(df, aes(x = get(grouping), y = get(metric))) +
    geom_point(data = summary_stats, inherit.aes = FALSE, 
               mapping = aes(x = get(grouping), y=avg_metric), 
               color = nrel_cols[1], size = 2) + 
    geom_errorbar(data = summary_stats, inherit.aes = FALSE, 
                  mapping = aes(x = get(grouping), ymin=avg_metric-error, ymax=avg_metric+error), 
                  color = nrel_cols[1], width=.1, linewidth = 0.4) +
    geom_jitter(width = 0.1,
           size = 1, pch = 21, color = "black", stroke = 0.3, fill = nrel_cols[1]) +
    labs(x=grouping_label, y = metric_label,
        title = "95% Confidence Interval of the Mean")
    
    if (grouping == "gap_ft") {
        fig <- fig + scale_x_discrete(limits = gap_order)
    }
    
    ggsave(file=file.path(nutrient_dir, grouping_dir, paste(crop_name, metric_label, ".jpg")), width=4, height=3)
}

# # Plot the 3-year data + confidence interval plots

leafy_metric_names <- c("Average Leaves per Plant", 
                        "Fresh Weight per Plant",
                        "Dry Weight per Plant",
                       "Fresh Weight per Leaf",
                        "Dry Weight per Leaf")
metrics <- c("leaves_per_plant", "fw_per_plant", "dw_per_plant", "fw_per_leaf", "dw_per_leaf")

# ### Kale

# Load in table titled df
load(file.path(data_dir, "kale_2016-2018.Rda"))

for (i in seq_along(leafy_metric_names)) {
    for (ii in seq_along(groupings)) {
        three_year_ci_plot(df, "kale", grouping_dirs[ii], groupings[ii], grouping_names[ii],
                metrics[i], leafy_metric_names[i])
        # Reversed grouping/year plot
    three_year_ci_plot(df, "kale", grouping_dirs[ii], "year", "Year",
                metrics[i], leafy_metric_names[i], fill = groupings[ii], fill_label = grouping_names_short[ii],
                      title_detail = "by year")
    }
    control_vs_array_ci_plot(df, "kale", "By year", metrics[i], leafy_metric_names[i])
}

# ## Swiss Chard

# Load in table titled df
load(file.path(data_dir, "swiss_chard_2016-2018.Rda"))

for (i in seq_along(leafy_metric_names)) {
    for (ii in seq_along(groupings)) {
        three_year_ci_plot(df, "swiss chard", grouping_dirs[ii], groupings[ii], grouping_names[ii],
                metrics[i], leafy_metric_names[i])
        # Reversed grouping/year plot
    three_year_ci_plot(df, "swiss chard", grouping_dirs[ii], "year", "Year",
                metrics[i], leafy_metric_names[i], fill = groupings[ii], fill_label = grouping_names_short[ii],
                      title_detail = "by year")
    }
    control_vs_array_ci_plot(df, "swiss chard", "By year", metrics[i], leafy_metric_names[i])
}

ggplot(df, aes(x = plant_covariate, y = dw_per_plant, color = gap_ft)) +
    facet_grid(~ year) + 
    geom_smooth(method = "lm",
              formula = y ~ x,
               linewidth = 0.5) + 
    geom_point(size = 0.5) + 
    labs(x="Mean plants per plug", y = "Mean dry weight per plant")
ggsave(file=file.path(misc_dir, paste("swiss_chard_plant_covariate_by_cell.jpg")), width=4, height=3)

# ## Pepper

pepper_metric_names <- c("Average Peppers per Plant", 
                        "Pepper Fresh Weight per Plant",
                        "Pepper Dry Weight per Plant",
                        "Fresh Weight per Pepper",
                        "Dry Weight per Pepper")
metrics <- c("peppers_per_plant", "fw_per_plant", "dw_per_plant", "fw_per_pepper", "dw_per_pepper")

# Load in table titled df
load(file.path(data_dir, "pepper_2016-2018.Rda"))

# Add in "per pepper" metrics

for (i in seq_along(pepper_metric_names)) {
    for (ii in seq_along(groupings)) {
        three_year_ci_plot(df, "pepper", grouping_dirs[ii], groupings[ii], grouping_names[ii],
                metrics[i], pepper_metric_names[i])
    }
    # Reversed area/year plot
    three_year_ci_plot(df, "pepper", "By area", "year", "Year",
                metrics[i], pepper_metric_names[i], fill = groupings[2], fill_label = "Location",
                      title_detail = "by year")
    control_vs_array_ci_plot(df, "pepper", "By year", metrics[i], pepper_metric_names[i])
}

# +
fig1 <- ggplot(df, aes(x = peppers_per_plant, y = fw_per_plant, color = year, shape = gap_ft)) +
    geom_point() + 
  # facet_grid(~ year) + 
    scale_shape_manual(values = c(16, 17, 15, 18, 3)) + 
    scale_color_manual(values=nrel_cols) +
    labs(x= "Peppers per plant", y = "Fresh weight per plant (g)", shape = "Spacing (ft)", color = "Year") +
    scale_y_continuous(limits = c(0, NA), expand = expansion(mult = c(0, .05)))
# ggsave(file=file.path(misc_dir, paste("fw_vs_peppers_per_plant.jpg")), width=4, height=3)

fig2 <- ggplot(df, aes(x = peppers_per_plant, y = fw_per_pepper, color = year, shape = gap_ft)) +
    geom_point() + 
   # facet_grid(~ year) + 
    scale_shape_manual(values = c(16, 17, 15, 18, 3)) + 
    scale_color_manual(values=nrel_cols) +
    labs(x= "Peppers per plant", y = "Fresh weight per pepper (g)", shape = "Spacing (ft)", color = "Year") +
    scale_y_continuous(limits = c(0, NA), expand = expansion(mult = c(0, .05)))

legend_r <- get_legend(fig1 + theme(legend.position="right"))
figs <- plot_grid(fig1 + theme(legend.position="none"),
                  fig2 + theme(legend.position="none"),
                  nrow = 1, 
                 labels=c("(a)","(b)"),
                 label_x = 0, label_y = 0, # Put plot labels in lower left-hand corner
                 hjust = -0.5, vjust = -0.5)
figs <- plot_grid(figs,
                  legend_r,
                 ncol = 2, rel_widths = c(1, .2))
# -

ggsave(file=file.path(misc_dir, paste("per_pepper_weight_comparisons.jpg")), width=8, height=3)

# # Broccoli plots

broc_metric_names <- c("Head Fresh Weight per Plant", 
                        "Head Dry Weight per Plant",
                       "Stem Fresh Weight per Plant", 
                        "Stem Dry Weight per Plant")
metrics <- c("head_fw_per_plant", "head_dw_per_plant", "stem_fw_per_plant", "stem_dw_per_plant")

# Load in table titled df
load(file.path(data_dir, "broccoli_2017-2018.Rda"))

for (i in seq_along(broc_metric_names)) {
    for (ii in seq_along(groupings)) {
        three_year_ci_plot(df, "broccoli", grouping_dirs[ii], groupings[ii], grouping_names[ii],
                metrics[i], broc_metric_names[i])
        three_year_box_plot(df, "broccoli", grouping_dirs[ii], groupings[ii], grouping_names[ii],
                metrics[i], broc_metric_names[i], fill = groupings[ii])
        # Reversed grouping/year plot
    three_year_ci_plot(df, "broccoli", grouping_dirs[ii], "year", "Year",
                metrics[i], broc_metric_names[i], fill = groupings[ii], fill_label = grouping_names_short[ii],
                      title_detail = "by year")
    }
    control_vs_array_ci_plot(df, "broc", "By year", metrics[i], broc_metric_names[i])
    control_vs_array_box_plot(df, "broc", "By year", metrics[i], broc_metric_names[i])
}

# # Cross-vegetable area comparisons

# Actually don't want to facet wrap because all the metrics are different; want to mix different plots

# grouping by area
ii <- 2

load(file.path(data_dir, "swiss_chard_2016-2018.Rda"))
fig3 <- three_year_ci_plot(df, "pepper", grouping_dirs[ii], "year", "Year",
                 "fw_per_plant", "Fresh Weight per Plant (g)", fill = groupings[ii], fill_label = grouping_names_short[ii],
                      title_detail = "by year", save = FALSE)
fig3 <- fig3 + labs(title = "Swiss Chard")

load(file.path(data_dir, "kale_2016-2018.Rda"))
fig4 <- three_year_ci_plot(df, "pepper", grouping_dirs[ii], "year", "Year",
                 "fw_per_plant", "Fresh Weight per Plant (g)", fill = groupings[ii], fill_label = grouping_names_short[ii],
                      title_detail = "by year", save = FALSE)
fig4 <- fig4 + labs(title = "Kale")

# Load in table titled df
load(file.path(data_dir, "broccoli_2017-2018.Rda"))
levels(df$year) <- c(levels(df$year), "2016") # add blank level
df$year <- factor(df$year, levels = c("2016", "2017", "2018"))
fig1 <- three_year_ci_plot(df, "broccoli", grouping_dirs[ii], "year", "Year",
                 "head_fw_per_plant", "Head Fresh Weight per Plant (g)", fill = groupings[ii], fill_label = grouping_names_short[ii],
                      title_detail = "by year", save = FALSE) 
fig1 <- fig1 + labs(title = "Broccoli") + 
    scale_x_discrete(drop=FALSE)

load(file.path(data_dir, "pepper_2016-2018.Rda"))
fig2 <- three_year_ci_plot(df, "pepper", grouping_dirs[ii], "year", "Year",
                 "fw_per_plant", "Fresh Weight per Plant (g)", fill = groupings[ii], fill_label = grouping_names_short[ii],
                      title_detail = "by year", save = FALSE)
fig2 <- fig2 + labs(title = "Pepper")

options(repr.plot.width = 8, repr.plot.height = 6)
legend_r <- get_legend(fig1 + theme(legend.position="right"))
figs <- plot_grid(fig3 + theme(legend.position="none"),
                  fig4 + theme(legend.position="none"),
                  legend_r,
                  fig2 + theme(legend.position="none"),
                  fig1 + theme(legend.position="none"),
                  nrow = 2,
                 rel_widths = c(1, 1, .3), 
                 labels=c("(a)","(b)", "", "(c)", "(d)"),
                 # label_x = 0, label_y = 0, # Put plot labels in lower left-hand corner
                 hjust = -1, vjust = 1.4) 

figs

ggsave(file=file.path(ci_dir, grouping_dirs[ii], "cross_crop_fresh_weight_by_area.jpg"), 
            width=8, height=6)

# # Nutrients

nutrient_metric_names <- c("Nitrogen", 
                        "Phosphorus",
                        "Potassium")
nutrient_metrics <- c("N", "P", "K")

# ## Kale

# Load in table titled df
load(file.path(data_dir, "kale_2018_NPK.Rda"))

for (i in seq_along(nutrient_metric_names)) {
    for (ii in seq_along(groupings)) {
        nutrient_ci_plot(df, "kale", grouping_dirs[ii], groupings[ii], grouping_names[ii],
                nutrient_metrics[i], nutrient_metric_names[i])
    }
}

# ## Swiss chard

# Load in table titled df
load(file.path(data_dir, "swiss_chard_2018_NPK.Rda"))

for (i in seq_along(nutrient_metric_names)) {
    for (ii in seq_along(groupings)) {
        nutrient_ci_plot(df, "swiss chard", grouping_dirs[ii], groupings[ii], grouping_names[ii],
                nutrient_metrics[i], nutrient_metric_names[i])
    }
}

# ## Broccoli

# Load in table titled df
load(file.path(data_dir, "broccoli_2018_NPK.Rda"))

for (i in seq_along(nutrient_metric_names)) {
    for (ii in seq_along(groupings)) {
        nutrient_ci_plot(df, "broccoli", grouping_dirs[ii], groupings[ii], grouping_names[ii],
                nutrient_metrics[i], nutrient_metric_names[i])
    }
}

# ## Pepper

# Load in table titled df
load(file.path(data_dir, "pepper_2018_NPK.Rda"))

for (i in seq_along(nutrient_metric_names)) {
    for (ii in seq_along(groupings)) {
        nutrient_ci_plot(df, "pepper", grouping_dirs[ii], groupings[ii], grouping_names[ii],
                nutrient_metrics[i], nutrient_metric_names[i])
    }
}

# # Weather

# Load in table titled df
load(file.path(data_dir, "monthly_weather.Rda"))
df %<>% pivot_longer(cols = c("total_prec_inch", "avg_t_min", "avg_t_max"), 
                   names_to = "measure") %>% 
    filter(month >= 5 & month <= 9) %>%
    mutate(measure = factor(measure, levels = c("total_prec_inch", "avg_t_max", "avg_t_min")))

fig1 <- ggplot(filter(df, measure == "total_prec_inch"), aes(x = month, y = value), ) +
    geom_col(fill = nrel_cols[1]) + 
    scale_x_continuous(limits = c(4.5, 9.5), breaks = seq(5.0, 9.0, 1.0),
                    labels=c("May", 
                              "June",
                              "July",
                              "Aug",
                              "Sept")) + 
    facet_grid(~ year) + 
    labs(x=element_blank(), y = ("Monthly Precipitation (in)")) + 
    ylim(0, NA) + 
    # theme_bw() + 
    theme(axis.text.x = element_text(angle = 45,  hjust=1), panel.grid.minor.x = element_blank()) 

fig2 <- ggplot(filter(df, measure != "total_prec_inch"), aes(x = month, y = value, color = measure)) +
    geom_line(linewidth = 0.75) +
    geom_point() + 
    scale_y_continuous(expression("Temperature ("*degree*F*")"), breaks = seq(30,90,10)) + 
    scale_x_continuous(limits = c(4.5, 9.5), breaks = seq(5.0, 9.0, 1.0),
                    labels=c("May", 
                              "June",
                              "July",
                              "Aug",
                              "Sept")) + 
    facet_grid(~ year) + 
    scale_color_manual(values=nrel_cols[c(5,2)], labels=c("Average daily maximum", "Average daily minimum"), name = "") +
    labs(x="Month in Growing Season") + 
    # theme_bw() + 
    theme(axis.text.x = element_text(angle = 45,  hjust=1), 
          panel.grid.minor.x = element_blank(),
          legend.position="bottom", legend.box = "horizontal",
          legend.box.margin=margin(-10,-10,-10,-10)
         ) 

# now add Available insolation from irradiance modeling
df <- read_csv(file.path(results_dir, "Irradiance modeling", "Monthly_Available_insolation_kWhm2day.csv"),
                        col_names = c("month", "2016", "2017", "2018"), skip = 1)

df %<>% pivot_longer(2:4, values_to = "kWhm2day", names_to = "year")

fig3 <- ggplot(df, aes(x = month, y = kWhm2day), ) +
    geom_col(fill = nrel_cols[2]) + 
    scale_x_continuous(limits = c(4.5, 9.5), breaks = seq(5.0, 9.0, 1.0),
                    labels=c("May", 
                              "June",
                              "July",
                              "Aug",
                              "Sept")) + 
    facet_grid(~ year) + 
    labs(x=element_blank(), y = bquote('Average Daily Insolation ' (kWh/m^2))) + 
    ylim(0, NA) + 
    # theme_bw() + 
    theme(axis.text.x = element_text(angle = 45,  hjust=1), panel.grid.minor.x = element_blank()) 

# cowplot works better than ggarrange to get the gridding & alignment nice
figs <- plot_grid(fig1, fig2, fig3, nrow = 2, ncol = 2, align = "v", rel_heights = c(1, 1.1))

figs

ggsave(file=file.path(weather_dir, "monthly_weather.jpg"), width=8, height=5)

# ## Available insolation from irradiance modeling

fig

ggsave(file=file.path(weather_dir, "monthly_avg_insolation.jpg"), width=4, height=3)


