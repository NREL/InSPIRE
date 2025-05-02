# ---
# jupyter:
#   jupytext:
#     formats: ipynb,R:light
#     text_representation:
#       extension: .R
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: R
#     language: R
#     name: ir
# ---

# Base plot using emmeans to plot estimated marginal mean, 95% CI (assuming normal distribution), and input data
# Each year is own facet
plot_inter_panel_gap_trends__emmeans <- function(df, emm, pd, dep_var, y_axis_title, breaks = NA, ymax = NA){

    stats_df <- summary(emm) # , adjust = "holm"
    stats_df %<>% filter(gap_ft != "Control")
    
    fig <- ggplot(filter(df, treatment != "Control"), 
                         aes(x = gap_ft, y = get(dep_var), fill = gap_ft)) +
        facet_grid(~year) + 
        geom_point(data = stats_df, inherit.aes = FALSE, 
                   mapping = aes(x = gap_ft, y = emmean, color = gap_ft, fill = gap_ft),
                   position = pd, size = 2) + 
        geom_errorbar(data = stats_df, inherit.aes = FALSE, 
                      mapping = aes(x = gap_ft, ymin=lower.CL, ymax=upper.CL, color = gap_ft), 
                      width=.1, position=pd, linewidth = 0.4) +
        geom_jitter(position=position_jitter(width = 0.1),
               size = 0.5, pch = 21, color = "black", stroke = 0.3) +
        scale_color_manual(values=alt_cols, labels = paste(levels(df$gap_ft), "ft")) +
        scale_fill_manual(values=alt_cols, guide = "none") +
        labs(x="Inter-panel spacing (ft)", y = y_axis_title, color = "") +
        theme( legend.position="right",
         panel.grid.minor.y = element_blank())

    if (!all(is.na(breaks))) {
        fig <- fig + scale_y_continuous(expand = expansion(mult = c(0, .05)), breaks = breaks) +
            coord_cartesian(ylim = c(0, ymax)) 
    } else {
        fig <- fig + scale_y_continuous(expand = expansion(mult = c(0, .05))) + 
            coord_cartesian(ylim = c(0, ymax))
    }
    return(fig)
}

# Base plot plotting mean, 95% CI (assuming T distribution), and input data
# Each year is own facet
plot_inter_panel_gap_trends__t <- function(df, pd, dep_var, y_axis_title){

    summary_stats <- df %>% filter(treatment != "Control") %>%
    summarize(n = sum(!is.na(n_plants)),
      means = mean(get(dep_var), na.rm = TRUE),
      sd = sd(get(dep_var), na.rm = TRUE),
      error = qt(0.975,df=n-1)*sd/sqrt(n),
      .by = c(year, gap_ft))
    
    
    fig <- ggplot(filter(df, treatment != "Control"), 
                         aes(x = gap_ft, y = get(dep_var), fill = gap_ft)) +
        facet_grid(~year) + 
        geom_point(data = summary_stats, inherit.aes = FALSE, 
                   mapping = aes(x = gap_ft, y = means, color = gap_ft, fill = gap_ft),
                   position = pd, size = 2) + 
        geom_errorbar(data = summary_stats, inherit.aes = FALSE, 
                      mapping = aes(x = gap_ft, ymin=means-error, ymax=means+error, color = gap_ft), 
                      width=.1, position=pd, linewidth = 0.4) +
        geom_jitter(position=position_jitter(width = 0.1),
               size = 0.5, pch = 21, color = "black", stroke = 0.3) +
        scale_color_manual(values=nrel_cols) +
        scale_fill_manual(values=nrel_cols, guide = "none") +
        labs(x="Inter-panel spacing (ft)", y = y_axis_title) +
        scale_y_continuous(expand = expansion(mult = c(0, .05))) +
        coord_cartesian(ylim = c(0, NA)) + 
        theme( legend.position="none",
             panel.grid.minor.y = element_blank())

    return(fig)
}

# Set up data for error bars around the group mean
# t_dist: Bool indicating whether to use a t-distribution (calculated manually) or normal distribution (calculated with emmeans)
get_treatment_year_error_bar_data <- function(df, t_dist, dep_var) {
    if (t_dist) {
        summary_stats <- summarize(df, n = sum(!is.na(n_plants)),
          means = mean(get(dep_var), na.rm = TRUE),
          sd = sd(get(dep_var), na.rm = TRUE),
          error = qt(0.975,df=n-1)*sd/sqrt(n),
          .by = c(year, treatment))
    } else {
        plot_model <- lm(get({{dep_var}}) ~ treatment  * year, data = df)
         summary_stats <- summary(emmeans(plot_model, c("treatment", "year"))) %>%
            rename(means = emmean) %>%
        mutate(error = upper.CL - lower.CL)
    }
}
