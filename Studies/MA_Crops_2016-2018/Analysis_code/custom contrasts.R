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

# ## Contrast functions

# Contrasts to compare pooled AgPV - control comparison per year, only
within_year_pooled_comp.emmc <- function(levels, years, reverse = FALSE) {
    M <- data.frame(matrix(0, length(levels), years))
    row.names(M) <- levels
    # Within-year pooled AgPV - control comparison
    for (i in seq_len(years)){
        M[(1:(length(levels)/years)) + (i - 1) * length(levels)/years, i] <- 
            c(-1 * rep(1/(length(levels)/years-1), times = length(levels)/years - 1), 1)
        names(M)[i] <- paste(levels[i * length(levels)/years], 'pooled agPV', sep = " - ")
    }
    M
}

# Contrasts to compare pooled AgPV - control comparison across all years
all_years_pooled_comp.emmc <- function(levels, years, reverse = FALSE) {
    M <- data.frame(matrix(0, length(levels), 1))
    row.names(M) <- levels
    # All years pooled AgPV - control comparison
    M[(1:(length(levels))), 1] <- 
        rep(c(-1 * rep(1/(length(levels)/years-1), times = length(levels)/years - 1), 1),
            times = years)
    names(M)[1] <- paste('control', 'pooled agPV', sep = " - ")
    M
}

# Contrasts to compare pooled AgPV - control comparison per year,
# as well as each control or pooled agPV group from year-to-year
all_pooled_comp.emmc <- function(levels, years, reverse = FALSE) {
    M <- data.frame(matrix(0, length(levels), years^2))
    row.names(M) <- levels
    # Within-year pooled AgPV - control comparison
    for (i in seq_len(years)){
        M[(1:(length(levels)/years)) + (i - 1) * length(levels)/years, i] <- 
            c(-1 * rep(1/(length(levels)/years-1), times = length(levels)/years - 1), 1)
        names(M)[i] <- paste(levels[i * length(levels)/years], 'pooled agPV', sep = " - ")
    }

    for (i in 1 + seq_len(years-1)) {
        for (j in seq_len(i-1)) {
            # Between year pooled agPV comparisons
            M[(1:(length(levels)/years - 1)) + (i - 1) * length(levels)/years, i + j - 2 + years] <- 
                rep(1/(length(levels)/years-1), times = length(levels)/years - 1)
            M[(1:(length(levels)/years - 1)) + (j - 1) * length(levels)/years, i + j - 2 + years] <- 
                -1 * rep(1/(length(levels)/years-1), times = length(levels)/years - 1)
            names(M)[i + j - 2 + years] <- 
                paste(paste('pooled agPV ', unlist(strsplit(levels[[i * length(levels)/years]], split = " "))[2]),
                      unlist(strsplit(levels[[j * length(levels)/years]], split = " "))[2], sep = " - ")
            
            # Between year control comparisons
            M[i * length(levels)/years, i + j - 2 + years + years*(years-1)/2] <- 1
            M[j * length(levels)/years, i + j - 2 + years + years*(years-1)/2] <- -1
            names(M)[i + j - 2 + years + years*(years-1)/2] <- 
                paste(paste('control', unlist(strsplit(levels[[i * length(levels)/years]], split = " "))[2]),
                      unlist(strsplit(levels[[j * length(levels)/years]], split = " "))[2], sep = " - ")
        }
    }
    M
}

# This is a custom variant on emmeans' poly contrasts to remove the control plot from trend analysis
poly_excl_cont.emmc = function(levs, max.degree = min(6, k-1), exclude = integer(0), include, ...) {
    exclude = .get.excl(levs, exclude, include)
    nm = c("linear", "quadratic", "cubic", "quartic", paste("degree",5:20))
    k = length(levs) - length(exclude)
    M = as.data.frame(poly(seq_len(k), min(20,max.degree)))
    for (j in seq_len(ncol(M))) {
        con = M[, j]
        pos = which(con > .01)
        con = con / min(con[pos])
        z = max(abs(con - round(con)))
        while (z > .05) {
            con = con / z
            z = max(abs(con - round(con)))
        }
        M[ ,j] = round(con)
    }
    M[exclude,] = 0
    row.names(M) = levs
    names(M) = nm[seq_len(ncol(M))]
    attr(M, "desc") = "polynomial contrasts"
    attr(M, "adjust") = "none"
    M
}
