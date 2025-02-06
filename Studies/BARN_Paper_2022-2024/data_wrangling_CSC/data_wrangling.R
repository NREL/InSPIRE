library(tidyverse)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
tomatopepper_calendar <- read_csv("2022-2024/2022/tomatopepper_calendar.csv")
tomatopepper_jd <- read_csv("2022-2024/2022/tomatopepper_jd.csv")


# Set the directory to the current script location


# 2022 --------------------------------------------------------------------

# Pivot longer to gather all date columns under "Date"
long_df <- tomatopepper_calendar %>%
    pivot_longer(
    cols = `6/10/2022`:`10/21/2022`, 
    names_to = "Date", 
    values_to = "Value"
  ) %>%
  mutate(
    Date = as.Date(Date, format = "%m/%d/%Y"),
    `Observation Type` = case_when(
      `Observation Type` == "Insect Damage? (0-3)" ~ "Insect damage",
      `Observation Type` == "Animal Damage? (0-3)" ~ "Animal damage",
      `Observation Type` == "Water Stress? (0-2)" ~ "Water stress",
      `Observation Type` == "Insect damage? (0-3)" ~ "Insect damage",
      `Observation Type` == "Animal damage? (0-3)" ~ "Animal damage",
      `Observation Type` == "Water stress? (0-2)" ~ "Water stress",
      TRUE ~ `Observation Type`
    )
  ) %>%
  distinct() %>%
  mutate(
    Age = as.numeric(Date - as.Date("2022-06-10"))
  ) %>%
  select(Crop, Variety, Bed, `Plant #`, Date, Age, `Observation Type`, Value)

write_csv(long_df, "formatted/tplong_2022.csv")

# Pivot wider to spread "Observation Type" to multiple columns
wide_df <- long_df %>%
  pivot_wider(
    names_from = `Observation Type`,
    values_from = Value,
    # id_cols = c(Row_ID,Crop, Variety, Bed, `Plant #`, Date)
    id_cols = c(Crop, Variety, Bed, `Plant #`, Date, Age)
  ) %>%
  mutate(
    `Insect damage?` = replace_na(`Insect damage?`, 0),
    `Animal damage?` = replace_na(`Animal damage?`, 0),
    `Water stress?` = replace_na(`Water stress?`, 0)
  )

# Detect duplicates by grouping and counting
duplicates <- long_df %>%
  group_by(Crop, Variety, Bed, `Plant #`, Date, `Observation Type`, Value) %>%
  filter(n() > 1) %>%
  summarize(Count = n(), .groups = "drop")
# View the resulting dataframe
tp_wide_df <- wide_df

write_csv(tp_wide_df, "tomatopepper_2022.csv")

leafy_calendar <- read_csv("2022-2024/2022/leafy_calendar.csv")

# Pivot longer to gather all date columns under "Date"
long_leafy <- leafy_calendar %>%
  pivot_longer(
    cols = `6/10/2022`:`10/21/2022`, 
    names_to = "Date", 
    values_to = "Value"
  ) %>%
  mutate(
    Date = as.Date(Date, format = "%m/%d/%Y"),
    `Observation Type` = case_when(
      `Observation Type` == "Insect Damage? (0-3)" ~ "Insect damage",
      `Observation Type` == "Animal Damage? (0-3)" ~ "Animal damage",
      `Observation Type` == "Water Stress? (0-2)" ~ "Water stress",
      TRUE ~ `Observation Type`
    )
  ) %>%
  distinct() %>%
  mutate(
    Age = as.numeric(Date - as.Date("2022-06-10"))
  ) %>%
  select(Crop, Variety, Bed, `Plant #`, Date, Age, `Observation Type`, Value)

write_csv(long_df, "formatted/leafylong_2022.csv")
# Pivot wider to spread "Observation Type" to multiple columns
wide_leafy <- long_leafy %>%
  pivot_wider(
    names_from = `Observation Type`,
    values_from = Value,
    id_cols = c(Crop, Variety, Bed, `Plant #`, Date, Age)
  ) %>%
  mutate(
    `Insect damage?` = replace_na(`Insect damage?`, 0),
    `Animal damage?` = replace_na(`Animal damage?`, 0),
    `Water stress?` = replace_na(`Water stress?`, 0)
  )

# Detect duplicates by grouping and counting
duplicates <- long_leafy %>%
  group_by(Crop, Variety, Bed, `Plant #`, Date, `Observation Type`, Value) %>%
  filter(n() > 1) %>%
  summarize(Count = n(), .groups = "drop")

leafy_wide_df <- wide_leafy

write_csv(leafy_wide_df, "leafy_2022.csv")


# 2023 --------------------------------------------------------------------
all2023 <- read.csv("2022-2024/2023 BARN Raw Data.csv")

