# Haiti Minigrid Light and Shading
# Format and analyze crop shading results for Haiti Microgrids from SAM output
# Here, we took the values of insolation and calculated the fraction of light
# hitting the ground to come up with a perfect total sunlight for different
# minigrid scenarios in Haiti and different areas within the arrays.

# Load libraries
library(shiny)
library(rsconnect)
library(dplyr)
library(tidyr)
library(ggplot2)
library(plotly)

# working directory
#setwd("~/Documents/GitHub/InSPIRE/ShinyApps/Haiti-Shade")

# Set SAM file folder
sam_folder <- "SAM_irradiance"

# Get all 8760 files
sam_shading_files_all <- list.files(path = sam_folder)
sam_shading_files <- grep("100kW_crop", sam_shading_files_all, value = TRUE)

# Create list of file names without the .csv suffix
sam_options <- sub("-100kW_crop_shading.csv", "", sam_shading_files)

ui <- fluidPage(
  titlePanel(
    "USAID Haiti Agrivoltaic Microgrids - Irradiance Analysis",
    windowTitle = "USAID Haiti Agrivoltaic Microgrids - Irradiance Analysis" # Set browser window title
  ),
  fluidRow(
    column(width = 4,
           selectInput(inputId = "config",
                       label = "Configuration:",
                       choices = sam_options,
                       width = "100%")
    ),
    column(width = 8,
           h5("The dropdown on the left allows you to select the location and configuration of the minigrid system. On each of the graphs below, North is up. The transparent white box on the bottom with a black line at the top represents the solar panel. Cells on the graph underneath this box represent ground beneath the solar panel in the configuration. Click, hold and drag anywhere on the graph to zoom in; hover to see specific values; press the home icon to reset the graph space.")
    ),
    column(width = 12,
           tags$hr(style = "border-top: 1px solid #ccc;")
    ),
    column(width = 11,
           plotlyOutput(outputId = "haitiPlotDay", width = "100%"),
    ),
    column(width = 1,
           h4("Download"),
           downloadButton("downloadDaily", "CSV"),
           downloadButton("downloadDailyPlot", "Plot"),
           textInput("widthDay", "Width:", value = 12, width = "50px"),
           textInput("heightDay", "Height:", value = 4, width = "50px")
    ),
    column(width = 12,
           tags$hr(style = "border-top: 1px solid #ccc;")
    ),
    column(width = 5,
           plotlyOutput(outputId = "haitiPlotMonth", width = "100%"),
    ),
    column(width = 1,
           h4("Download"),
           downloadButton("downloadMonthly", "CSV"),
           downloadButton("downloadMonthlyPlot", "Plot"),
           textInput("widthMonth", "Width:", value = 6, width = "50px"),
           textInput("heightMonth", "Height:", value = 6, width = "50px"),
           checkboxInput("monthFlip", "Flip", value = FALSE)
    ),
    column(width = 5,
           plotlyOutput(outputId = "haitiPlotMonthP", width = "100%"),
    ),
    column(width = 1,
           h4("Download"),
           downloadButton("downloadMonthlyPercent", "CSV"),
           downloadButton("downloadMonthlyPPlot", "Plot"),
           textInput("widthMonthP", "Width:", value = 6, width = "50px"),
           textInput("heightMonthP", "Height:", value = 6, width = "50px"),
           checkboxInput("monthPFlip", "Flip", value = FALSE) 
    ),
    column(width = 12,
           tags$hr(style = "border-top: 1px solid #ccc;")
    )
  )
)

# Define server logic required to draw a histogram ----
server <- function(input, output, session) {
  
  irradiance_o <- reactive({
    
    file_index <- match(input$config, sam_options)
    if (!is.na(file_index)) {
      irradiance <- read.csv(paste0(sam_folder, "/", sam_shading_files[file_index]), stringsAsFactors = F)
      return(irradiance)
    } else {
      # Handle the case where input$config doesn't match any options
      return(NULL)  # Or provide a default dataset, or display an error message
    }
    
  })
  
  irradiance_d <- reactive({
    
    # Put the csv from the selection input into a dataframe
    irradiance_out <- irradiance_o()
    
    # Create percent shade dataframe (% GHI for each column)
    irradiance_out_percent <- irradiance_out / irradiance_out[,12]
    irradiance_out_percent <- round(irradiance_out_percent * 100, 2)[,2:11]
    irradiance_out_percent$Timestep <- irradiance_out$Timestep
    
    # Create an array to hold the data in a (row, column, slice) format as (day, position, year)
    irradiance_arr <- array(NA, dim = c(24, 10, 365))
    percent_arr <- array(NA, dim = c(24, 10, 365))
    
    # For each day of the year, fill the array with 24 hour x 10 bins of irradiance values
    for (i in 1:365){
      
      irradiance_arr[1:24, 1:10, i] <- as.matrix(irradiance_out[((24 * (i - 1)) + 1):(24 * i), 2:11])
      percent_arr[1:24, 1:10, i] <- as.matrix(irradiance_out_percent[((24 * (i - 1)) + 1):(24 * i), 1:10])
      
    }
    
    # Turn 0 values to NA to avoid 
    irradiance_arr[irradiance_arr == 0] <- NA
    percent_arr[percent_arr == 0] <- NA
    
    # Take the mean values for each day and position within the array and convert to a dataframe
    irradiance_day <- as.data.frame(apply(irradiance_arr, MARGIN = c(3, 2), mean, na.rm = T))
    percent_day <- as.data.frame(apply(percent_arr, MARGIN = c(3, 2), mean, na.rm = T))
    
    # Add a day of year column
    irradiance_day$Day <- 1:365
    percent_day$Day <- 1:365
    
    # Take the position in the dataframe measurements in meters and round down and convert to numbers (from strings)
    colnames(irradiance_day) <- c(round(as.numeric(substr(colnames(irradiance_out)[2:11], 2, 10)), 3), "Day")
    colnames(percent_day) <- c(round(as.numeric(substr(colnames(irradiance_out)[2:11], 2, 10)), 3), "Day")
    
    # Return
    list(irradiance_day, percent_day)
    
  })
  
  irradiance_d_l <- reactive({
    
    # Put the csv from the selection input into a dataframe
    irradiance_day <- irradiance_d()[[1]]
    
    # Repeat the pattern through the next row of panels
    irradiance_day_copy <- irradiance_day[, 1:10]
    colnames(irradiance_day_copy) <- as.numeric(colnames(irradiance_day_copy)) + as.numeric(colnames(irradiance_day)[10])
    irradiance_day_longer <- cbind(irradiance_day, irradiance_day_copy)
    
    # Transform the dataframe into a long format (column format) for plotting
    irradiance_long <- pivot_longer(irradiance_day_longer, -"Day", names_to = "Position", values_to = "Irradiance")
    
    # Make positions numeric
    irradiance_long$Position <- as.numeric(irradiance_long$Position)
    
    # Return
    irradiance_long
    
  })
  
  irradiance_m <- reactive({
    
    # Grab daily mean irradiance data
    irradiance_day <- irradiance_d()[[1]]
    
    # Assign months
    irradiance_day <- irradiance_day %>%
      mutate(Month = case_when(
        Day >= 1 & Day <= 31 ~ 1,
        Day >= 32 & Day <= 59 ~ 2,
        Day >= 60 & Day <= 90 ~ 3,
        Day >= 91 & Day <= 120 ~ 4,
        Day >= 121 & Day <= 151 ~ 5,
        Day >= 152 & Day <= 181 ~ 6,
        Day >= 182 & Day <= 212 ~ 7,
        Day >= 213 & Day <= 243 ~ 8,
        Day >= 244 & Day <= 273 ~ 9,
        Day >= 274 & Day <= 304 ~ 10,
        Day >= 305 & Day <= 334 ~ 11,
        Day >= 335 & Day <= 365 ~ 12
      ))
    
    # Take monthly means
    irradiance_month <- irradiance_day %>%
      group_by(Month) %>%
      summarize(across(where(is.numeric), ~ mean(.x, na.rm = TRUE)))
    
    # Grab daily mean irradiance data
    irradiance_day_percent <- irradiance_d()[[2]]
    
    # Assign months
    irradiance_day_percent <- irradiance_day_percent %>%
      mutate(Month = case_when(
        Day >= 1 & Day <= 31 ~ 1,
        Day >= 32 & Day <= 59 ~ 2,
        Day >= 60 & Day <= 90 ~ 3,
        Day >= 91 & Day <= 120 ~ 4,
        Day >= 121 & Day <= 151 ~ 5,
        Day >= 152 & Day <= 181 ~ 6,
        Day >= 182 & Day <= 212 ~ 7,
        Day >= 213 & Day <= 243 ~ 8,
        Day >= 244 & Day <= 273 ~ 9,
        Day >= 274 & Day <= 304 ~ 10,
        Day >= 305 & Day <= 334 ~ 11,
        Day >= 335 & Day <= 365 ~ 12
      ))
    
    # Take monthly means
    irradiance_month_percent <- irradiance_day_percent %>%
      group_by(Month) %>%
      summarize(across(where(is.numeric), mean, na.rm = TRUE))
    
    # Round to whole percentages
    irradiance_month_percent[,2:11] <- round(irradiance_month_percent[,2:11])
    
    # Repeat the pattern through the next row of panels
    irradiance_month_copy <- irradiance_month[, 2:11]
    colnames(irradiance_month_copy) <- as.numeric(colnames(irradiance_month_copy)) + as.numeric(colnames(irradiance_month)[11])
    irradiance_month_longer <- cbind(irradiance_month[, 1:11], irradiance_month_copy)
    
    # Repeat the pattern through the next row of panels
    irradiance_month_percent_copy <- irradiance_month_percent[, 2:11]
    colnames(irradiance_month_percent_copy) <- as.numeric(colnames(irradiance_month_percent_copy)) + as.numeric(colnames(irradiance_month_percent)[11])
    irradiance_month_percent_longer <- cbind(irradiance_month_percent[, 1:11], irradiance_month_percent_copy)
    
    # Transform the dataframe into a long format (column format) for plotting
    irradiance_month_long <- pivot_longer(irradiance_month_longer, 2:21, names_to = "Position", values_to = "Irradiance")
    
    # Transform the dataframe into a long format (column format) for plotting
    irradiance_month_percent_long <- pivot_longer(irradiance_month_percent_longer, 2:21, names_to = "Position", values_to = "Percent")
    
    # Make positions numeric
    irradiance_month_long$Position <- as.numeric(irradiance_month_long$Position)
    irradiance_month_percent_long$Position <- as.numeric(irradiance_month_percent_long$Position)
    
    # Return
    list(irradiance_month_long, irradiance_month_percent_long)
    
  })
  
  haitiPlotDay_s <- reactive({
    
    # Grab the freshly processed data
    irradiance_processed <- irradiance_d_l()
    
    # Calculate GCR intercepts for panels
    gcr_y <- 1.71 # Lenght of panel (1.797m) * cos(18 degree tilt)
    
    if(max(irradiance_processed$Position) > 6){
      second_row <- 5.4
    }else {second_row <- 2.92}
    
    # Plot a heatmap
    heat <- ggplot(irradiance_processed, aes(as.factor(Day), Position, fill = Irradiance)) + 
      geom_tile() + 
      geom_rect(xmin = as.factor(0),
                xmax = as.factor(366),
                ymin = 0, ymax = gcr_y, fill = "black", alpha = 0.5) +
      #geom_hline(yintercept = gcr_y) +
      geom_rect(xmin = as.factor(0),
                xmax = as.factor(366),
                ymin = second_row, ymax = second_row + gcr_y, fill = "black", alpha = 0.5) +
      #geom_hline(yintercept = second_row) + geom_hline(yintercept = second_row + gcr_y) +
      ylim(0, 10.5) +
      scale_fill_gradientn(limits = c(0, 700), colors = c("#FFF2CC", "#E8C872", "#F1BC31", "#E25822", "#B22222", "#7C0A02", "#461111")) +
      scale_x_discrete(breaks = seq(0, 365, 5)) +
      theme_bw() + 
      theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
      labs(title = "Mean Daily Irradiance (W/m^2)", x = "Day of Year", y = "Position within Array (m)")
    
    # Return
    heat
    
  })
  
  haitiPlotMonth_s <- reactive({
    
    # Grab the freshly processed data
    irradiance_monthly <- irradiance_m()[[1]]
    
    if (input$monthFlip) {
      
      # Plot a heatmap
      heat <- ggplot(irradiance_monthly, aes(y = as.factor(Month), x = as.factor(Position), fill = Irradiance)) + 
        geom_tile() + 
        scale_fill_gradientn(limits = c(0, 700), colors = c("#FFF2CC", "#E8C872", "#F1BC31", "#E25822", "#B22222", "#7C0A02", "#461111")) +
        theme_bw() + 
        theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
        labs(title = "", x = "", y = "")
    
    } else {
        
      # Calculate GCR intercepts for panels
      gcr_y <- 1.71 # Lenght of panel (1.797m) * cos(18 degree tilt)
      
      if(max(irradiance_monthly$Position) > 6){
        second_row <- 5.4
      }else {second_row <- 2.92}
      
      # Plot a heatmap
      heat <- ggplot(irradiance_monthly, aes(as.factor(Month), Position, fill = Irradiance)) + 
        geom_tile() + 
        geom_rect(xmin = as.factor(0),
                  xmax = as.factor(13),
                  ymin = 0, ymax = gcr_y, fill = "black", alpha = 0.5) +
        #geom_hline(yintercept = gcr_y) +
        geom_rect(xmin = as.factor(0),
                  xmax = as.factor(366),
                  ymin = second_row, ymax = second_row + gcr_y, fill = "black", alpha = 0.5) +
        #geom_hline(yintercept = second_row) + geom_hline(yintercept = second_row + gcr_y) +
        ylim(0, 10.5) +
        scale_fill_gradientn(limits = c(0, 700), colors = c("#FFF2CC", "#E8C872", "#F1BC31", "#E25822", "#B22222", "#7C0A02", "#461111")) +
        scale_x_discrete(labels = month.name) +
        theme_bw() + 
        theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
        labs(title = "Mean Monthly Irradiance (W/m^2)", x = "Month", y = "Position within Array (m)")
      
    }
    
    # Return
    heat
    
  })
  
  haitiPlotMonthP_s <- reactive({
    
    # Grab the freshly processed data
    irradiance_monthly_p <- irradiance_m()[[2]]
    
    if (input$monthPFlip) {
      
      # Plot a heatmap
      heat <- ggplot(irradiance_monthly_p, aes(y = as.factor(Month), x = as.factor(Position), fill = Percent)) + 
        geom_tile() + 
        scale_fill_gradientn(limits = c(0, 100), colors = c("#FFF2CC", "#E8C872", "#F1BC31", "#E25822", "#B22222", "#7C0A02", "#461111")) +
        theme_bw() + 
        theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
        labs(title = "", x = "", y = "")
      
    } else {
      
      # Calculate GCR intercepts for panels
      gcr_y <- 1.71 # Lenght of panel (1.797m) * cos(18 degree tilt)
      
      if(max(irradiance_monthly_p$Position) > 6){
        second_row <- 5.4
      }else {second_row <- 2.92}
      
      # Plot a heatmap
      heat <- ggplot(irradiance_monthly_p, aes(as.factor(Month), Position, fill = Percent)) + 
        geom_tile() + 
        geom_rect(xmin = as.factor(0),
                  xmax = as.factor(13),
                  ymin = 0, ymax = gcr_y, fill = "black", alpha = 0.5) +
        #geom_hline(yintercept = gcr_y) +
        geom_rect(xmin = as.factor(0),
                  xmax = as.factor(366),
                  ymin = second_row, ymax = second_row + gcr_y, fill = "black", alpha = 0.5) +
        #geom_hline(yintercept = second_row) + geom_hline(yintercept = second_row + gcr_y) +
        ylim(0, 10.5) +
        scale_fill_gradientn(limits = c(0, 100), colors = c("#FFF2CC", "#E8C872", "#F1BC31", "#E25822", "#B22222", "#7C0A02", "#461111")) +
        scale_x_discrete(labels = month.name) +
        theme_bw() + 
        theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
        labs(title = "Percent GHI (%)", x = "Month", y = "Position within Array (m)")
      
    }
    
    # Return
    heat
    
  })
  
  output$haitiPlotDay <- renderPlotly({
    
    heat_in <- haitiPlotDay_s()
    
    ggplotly(heat_in, tooltip="Irradiance")
    
  })
  
  output$haitiPlotMonth <- renderPlotly({
    
    heat_in <- haitiPlotMonth_s()
    
    ggplotly(heat_in, tooltip="Irradiance")
    
  })
  
  output$haitiPlotMonthP <- renderPlotly({
    
    heat_in <- haitiPlotMonthP_s()
   
    ggplotly(heat_in, tooltip="Percent")
    
  })
  
  output$downloadDaily <- downloadHandler(
    filename = function() {
      paste(sam_shading_files[match(input$config, sam_options)], "_Daily.csv", sep = "")
    },
    content = function(file) {
      day_long_out <- irradiance_d_l()
      day_long_wide <- day_long_out %>%
        pivot_wider(names_from = Position, values_from = Irradiance)
      write.csv(day_long_wide, file, row.names = FALSE)
    }
  )
  
  output$downloadMonthly <- downloadHandler(
    filename = function() {
      paste(sam_shading_files[match(input$config, sam_options)], "_Monthly.csv", sep = "")
    },
    content = function(file) {
      month_out <- irradiance_m()[[1]]
      month_wide <- month_out %>%
        pivot_wider(names_from = Position, values_from = Irradiance)
      write.csv(month_wide, file, row.names = FALSE)
    }
  )
  
  output$downloadMonthlyPercent <- downloadHandler(
    filename = function() {
      paste(sam_shading_files[match(input$config, sam_options)], "_Monthly_Percent.csv", sep = "")
    },
    content = function(file) {
      month_out_percent <- irradiance_m()[[2]]
      month_wide_percent <- month_out_percent %>%
        pivot_wider(names_from = Position, values_from = Percent)
      write.csv(month_wide_percent, file, row.names = FALSE)
    }
  )
  
  output$downloadDailyPlot <- downloadHandler(
    filename = function() {
      paste0(sam_shading_files[match(input$config, sam_options)], "_Daily.png")
    },
    content = function(file) {
      heat_in <- haitiPlotDay_s()
      png(file, width = as.numeric(input$widthDay),
          height = as.numeric(input$heightDay),
          units = "in",
          res = 300,)
      print(heat_in)
      dev.off()
    }
  )
  
  output$downloadMonthlyPlot <- downloadHandler(
    filename = function() {
      paste0(sam_shading_files[match(input$config, sam_options)], "_Monthly.png")
    },
    content = function(file) {
      heat_in <- haitiPlotMonth_s()
      png(file, width = as.numeric(input$widthMonth),
          height = as.numeric(input$heightMonth),
          units = "in",
          res = 300)
      print(heat_in)
      dev.off()
    }
  )
  
  output$downloadMonthlyPPlot <- downloadHandler(
    filename = function() {
      paste0(sam_shading_files[match(input$config, sam_options)], "_Monthly_Percent.png")
    },
    content = function(file) {
      heat_in <- haitiPlotMonthP_s()
      png(file, width = as.numeric(input$widthMonthP),
          height = as.numeric(input$heightMonthP),
          units = "in",
          res = 300,)
      print(heat_in)
      dev.off()
    }
  )
  
}

# Create and launch the Shiny app:
shinyApp(ui = ui, server = server)