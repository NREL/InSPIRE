library(shiny)
library(tidyr)
library(stringdist)
library(dplyr)
library(plotly)
library(viridis)

ui <- fluidPage(
  titlePanel("InSPIRE Data Portal Processing",
    windowTitle = "InSPIRE Data Portal Data Processing" # Set browser window title
  ),
  fluidRow(
    column(width = 4,
           h2("Description"),
           h5("Here you will find the most up-to-date data pulled directly from the InSPIRE Data Portal. This R Shiny App pulls the CSV straight from the data portal and processes. It creates counts for the total number of papers for each of many categories including the InSPIRE topics and sub-topics, jurisdictions including country and U.S. state, type of publication, and low-impact development strategy. In addition, it automatically summarizes and produces the text to input into SankeyMATIC to create a Sankey chart showing the distribution of publications by InSPIRE topic and sub-topic.")
    ),
    column(width = 4,
           h2("SankeyMatic Text"),
           h5("Copy and paste the text below into the SankeyMatic input form to produce an up-to-date Sankey plot."),
           uiOutput("longTextOutput")
    ),
    column(width = 4,
           h2("Download Processed Data"),
           downloadButton("downloadData", "Processed Data Download"),
           p("Download InSPIRE Data Portal statistics CSV.")
    ),
    column(width = 12,
           tags$hr(style = "border-top: 1px solid #ccc;")
    ),
    column(width = 12,
           h2("Publications through Time"),
           plotlyOutput(outputId = "publicationByYear", width = "100%")
    ),
    column(width = 12,
           tags$hr(style = "border-top: 1px solid #ccc;")
    ),
    column(width = 6,
           h2("Document Types"),
           plotlyOutput(outputId = "publicationByType", width = "100%")
    ),
    column(width = 6,
           h2("Low-Impact Development Strategies"),
           plotlyOutput(outputId = "publicationByStrategy", width = "100%")
    )
  )
)

server <- function(input, output) {
  
  data_portal_out <- reactive({
    
    withProgress(message = 'Downloading data...', value = 0, {
      
      # Read in csv
      data_portal <- read.csv('https://openei.org/w/index.php?title=Special:Ask&x=-5B-5BCategory%3AInSPIRE-20References-5D-5D%2F-3FDocument-20type%2F-3FInSPIRE-2Dstrategy%2F-3FInSPIRE-2Djurisdiction%2F-3FInSPIRE-2Dcountry%2F-3FInSPIRE-2Dstate%2F-3FModel%2F-3FNovelData%2F-3FInSPIRE-2Dfield-2Ddata%2F-3FPublicationDate%2F-3FInSPIRE-2Dtopic%2F-3FInSPIRE-2Dentomology-2Dtopic%2F-3FInSPIRE-2Dlivestock-2Dtopic%2F-3FInSPIRE-2Dwildlife-2Dtopic%2F-3FInSPIRE-2Dplant-2Dscience-2Dtopic%2F-3FInSPIRE-2Dhuman-2Dhealth-2Dtopic%2F-3FInSPIRE-2Dmicroclimatology-2Dtopic%2F-3FInSPIRE-2Dsoil-2Dtopic%2F-3FInSPIRE-2Dhydrology-2Dtopic%2F-3FInSPIRE-2Dsocial-2Dperspectives-2Dtopic%2F-3FInSPIRE-2Dpolicy-2Dtopic%2F-3FInSPIRE-2Dmarket-2Dtopic%2F-3FInSPIRE-2Deconomics-2Dtopic%2F-3FInSPIRE-2Dpv-2Dtopic%2F-3FInSPIRE-2Dconfiguration-2Dtopic%2F-3FInSPIRE-2Dsiting-2Dtopic%2F-3FInSPIRE-2Dimpact-2Dassessments-2Dtopic&mainlabel=&limit=2000&format=csv', header = T)
      
      # Increment the progress bar, and update the detail text.
      incProgress(0.5, detail = 'Processing data!')
      
      # Add sub-topics for the topics without sub-topics
      data_portal$Review <- data_portal$Tools <- data_portal$Standardization <- data_portal$Methods <- NA
      
      # Replace no text with NA
      data_portal[data_portal == ""] <- NA
      
      # Rename those single topics for matching later
      colnames(data_portal) <- c("Title",
                                 "Document Type",
                                 "Low Impact Development Strategy",
                                 "Jurisdiction",
                                 "Country",
                                 "State",
                                 "Model",
                                 "Novel",
                                 "Field",
                                 "Publication Date",
                                 "Topics",
                                 "Entomology",
                                 "Livestock",
                                 "Wildlife",
                                 "Plant Science",
                                 "Human Health",
                                 "Microclimatology",
                                 "Soil",
                                 "Hydrology",
                                 "Social Perspectives",
                                 "Policy and Regulatory Issues",
                                 "Market Assessments",
                                 "Economics",
                                 "PV Technologies",
                                 "System Configuration",
                                 "Siting",
                                 "Impact Assessments",
                                 "Methodological Comparisons",
                                 "Standardization and Best Practices",
                                 "Tools",
                                 "Reviews/Informational")
      
      # Separate topics into rows
      data_portal_c <- data_portal %>%
        separate_rows(Topics, sep = ",")
      
      # Add sub-topics to single topic topics for tallying later
      data_portal_c$'Methodological Comparisons'[data_portal_c$Topics == "Methodological Comparisons"] <- "Methodological Comparisons"
      data_portal_c$Tools[data_portal_c$Topics == "Tools"] <- "Tools"
      data_portal_c$'Standardization and Best Practices'[data_portal_c$Topics == "Standardization and Best Practices"] <- "Standardization and Best Practices"
      data_portal_c$'Reviews/Informational'[data_portal_c$Topics == "Reviews/Informational"] <- "Reviews/Informational"
      
      # Compress to longer format dataframe
      sub_topic_columns <- c("Entomology",
                             "Livestock",
                             "Wildlife",
                             "Plant Science",
                             "Human Health",
                             "Microclimatology",
                             "Soil",
                             "Hydrology",
                             "Social Perspectives",
                             "Policy and Regulatory Issues",
                             "Market Assessments",
                             "Economics",
                             "PV Technologies",
                             "System Configuration",
                             "Siting",
                             "Impact Assessments",
                             "Methodological Comparisons",
                             "Standardization and Best Practices",
                             "Tools",
                             "Reviews/Informational")
      
      # Compress topic columns into long format
      data_portal_l <- pivot_longer(data_portal_c, cols = sub_topic_columns, names_to = "Topic", values_to = "SubTopic")
      
      # Separate sub-topics into rows
      data_portal_el <- data_portal_l %>%
        separate_rows(SubTopic, sep = ",")
      
      # Remove any rows where the sub-topic is NA
      data_portal_eln <- data_portal_el[!is.na(data_portal_el$SubTopic),]
      
      # Remove all rows where the topic doesn't match the sub-topics
      data_portal_elnf <- data_portal_eln[data_portal_eln$Topics == data_portal_eln$Topic, ]
      data_portal_elnf <- data_portal_elnf[, -which(names(data_portal_elnf) == "Topics")]
      
      # Separate jurisdiction, country, state into rows
      data_portal_jurisdiction <- data_portal[, c('Jurisdiction', 'Country', 'State')] %>%
        separate_rows(Jurisdiction, sep = ",") %>%
        separate_rows(Country, sep = ",") %>%
        separate_rows(State, sep = ",")
      
      # Set county to United States where State is filled
      data_portal_jurisdiction$Country[!is.na(data_portal_jurisdiction$State)] <- "United States"
      
      # Separate low-impact development strategy
      data_portal_strategy <- data_portal[, c('Title', 'Low Impact Development Strategy')] %>%
        separate_rows(`Low Impact Development Strategy`, sep = ",")
      
      # Extract publication year
      data_portal_year <- data_portal[, c('Title', 'Publication Date')]
      data_portal_year$Year <- as.numeric(gsub(".*?([0-9]+)$", "\\1", data_portal_year$`Publication Date`))
      
      # Count topics and sub-topics
      data_portal_document_count <- table(data_portal[, 'Document Type'])
      data_portal_strategy_count <- table(data_portal_strategy$`Low Impact Development Strategy`)
      data_portal_jurisdiction_count <- table(data_portal_jurisdiction$Jurisdiction)
      data_portal_country_count <- table(data_portal_jurisdiction$Country)
      data_portal_state_count <- table(data_portal_jurisdiction$State)
      data_portal_year_count <- table(data_portal_year$Year)
      data_portal_topic_count <- table(data_portal_c[, 'Topics'])
      data_portal_subtopic_count <- table(data_portal_elnf$SubTopic)
      
      # Define category names
      categories <- c("Document Type", "Low Impact Development Strategy", "Jurisdiction", "Country", "State", "Year", "Topic", "SubTopic")
      
      # Combine tables into a single list
      counts <- list(data_portal_document_count, data_portal_strategy_count, data_portal_jurisdiction_count,
                     data_portal_country_count, data_portal_state_count, data_portal_year_count,
                     data_portal_topic_count, data_portal_subtopic_count)
      
      # Create a data frame from the list
      data_portal_counts <- data.frame(
        Category = rep(categories, sapply(counts, length)),
        Subject = unlist(lapply(counts, names)),
        Count = unlist(counts)
      )
      
      # Sort the dataframe by category
      data_portal_counts <- data_portal_counts %>% arrange(Category, Subject)
      
    })
    
    # Return
    data_portal_counts
    
  })
  
  lines_out <- reactive({
    
    # Ensure that CSV is downloaded and processed
    req(data_portal_out())
    
    # Grab topc and subtopic counts
    data_portal_in <- data_portal_out()
    data_portal_topics <- data_portal_in[data_portal_in$Category == "Topic", 2:3]
    data_portal_subs <- data_portal_in[data_portal_in$Category == "SubTopic", 2:3]
    
    # Define functions to find index of matching topics and subtopics
    topic_match <- function(findTopic){
      t_m <- which(stringsimmatrix(data_portal_topics[, 1], findTopic) > 0.75)
      if(length(t_m) == 1){
        return(data_portal_topics[t_m, 2])
      }else{
        return(0)
        print(paste0("Topic: ", findTopic))
      }
    }
    
    sub_match <- function(findSub){
      s_m <- which(stringsimmatrix(data_portal_subs[, 1], findSub) > 0.75)
      if(length(s_m) == 1){
        return(data_portal_subs[s_m, 2])
      }else{
        return(0)
        print(paste0("Sub-Topic: ", findSub))
      }
    }
    
    # Calculate totals for major topics
    biosci <- sum(topic_match("Entomology"),
                  topic_match("Wildlife"),
                  topic_match("Livestock"),
                  topic_match("Human Health"),
                  topic_match("Plant Science"), na.rm = T)
    
    physsci <- sum(topic_match("Hydrology"),
                   topic_match("Soil"),
                   topic_match("Microclimatology"), na.rm = T)
    
    socsci <- sum(topic_match("Social Perspectives"),
                  topic_match("Market Assessments"),
                  topic_match("Policy and Regulatory Issues"),
                  topic_match("Economics"), na.rm = T)
    
    tech <- sum(topic_match("PV Technologies"),
                topic_match("System Configurations"),
                topic_match("Siting"), na.rm = T)
    
    cross <- sum(topic_match("Standardization and Best Practices"),
                 topic_match("Impact Assessments"),
                 topic_match("Tools"),
                 topic_match("Methodological Comparisons"),
                 topic_match("Reviews / Informational"), na.rm = T)
    
    # Define your text lines here
    lines <- c(paste0("Data Portal Entries [", biosci,"] Biological Sciences"),
               paste0("Data Portal Entries [", physsci,"] Physical Sciences"),
               paste0("Data Portal Entries [", socsci,"] Social Sciences"),
               paste0("Data Portal Entries [", tech,"] Technology"),
               paste0("Data Portal Entries [", cross,"] Crosscutting"),
               "",
               paste0("Physical Sciences [", topic_match("Hydrology"), "] Hydrology"),
               paste0("Physical Sciences [", topic_match("Soil"), "] Soil"),
               paste0("Physical Sciences [", topic_match("Microclimatology"), "] Microclimatology"),
               "",
               paste0("Biological Sciences [", topic_match("Entomology"), "] Entomology"),
               paste0("Biological Sciences [", topic_match("Wildlife"), "] Wildlife"),
               paste0("Biological Sciences [", topic_match("Livestock"), "] Livestock"),
               paste0("Biological Sciences [", topic_match("Human Health"), "] Human Health"),
               paste0("Biological Sciences [", topic_match("Plant Sciences"), "] Plant Sciences"),
               "",
               paste0("Technology [", topic_match("PV Technologies"), "] PV Technologies"),
               paste0("Technology [", topic_match("System Configuration"), "] System Configuration"),
               paste0("Technology [", topic_match("Siting"), "] Siting"),
               "",
               paste0("Social Sciences [", topic_match("Social Perspectives"), "] Social Perspectives"),
               paste0("Social Sciences [", topic_match("Market Assessments"), "] Market Assessments"),
               paste0("Social Sciences [", topic_match("Policy and Regulatory Issues"), "] Policy & Regulatory"),
               paste0("Social Sciences [", topic_match("Economics"), "] Economics"),
               "",
               paste0("Crosscutting [", topic_match("Standardization and Best Practices"), "] Standards & Best Practices"),
               paste0("Crosscutting [", topic_match("Impact Assessments"), "] Impact Assessments"),
               paste0("Crosscutting [", topic_match("Tools"), "] Tools"),
               paste0("Crosscutting [", topic_match("Methodological Comparisons"), "] Methodological Comparisons"),
               paste0("Crosscutting [", topic_match("Reviews / Informational"), "] Reviews / Informational"),
               "",
               paste0("Plant Sciences [", sub_match("Plant Productivity and Yields"), "] Plant Productivity and Yields"),
               paste0("Plant Sciences [", sub_match("Groundcover abundance/richness/diversity"), "] Groundcover"),
               paste0("Plant Sciences [", sub_match("Plant Phenology"), "] Plant Phenology"),
               paste0("Plant Sciences [", sub_match("Plant Physiology"), "] Plant Physiology"),
               paste0("Plant Sciences [", sub_match("Nutrition"), "] Nutrition"),
               paste0("Plant Sciences [", sub_match("Pests and Diseases"), "] Pests and Diseases"),
               paste0("Plant Sciences [", sub_match("Fire Risks"), "] Fire Risks"),
               paste0("Plant Sciences [", sub_match("Irrigation Efficiency"), "] Irrigation Efficiency"),
               "",
               paste0("Livestock [", sub_match("Stocking Rates & Approaches"), "] Stocking Rates & Approaches"),
               paste0("Livestock [", sub_match("Animal welfare/temp/water intake"), "] Animal Welfare / Temperature / Water Consumption"),
               paste0("Livestock [", sub_match("Animal Behavior"), "] Animal Behavior"),
               paste0("Livestock [", sub_match("Weight/milk/fiber/meat product"), "] Weight / Milk / Fiber / Meat Production"),
               "",
               paste0("Entomology [", sub_match("Abundance Richness and Diversity"), "] Abundance Richness and Diversity"),
               paste0("Entomology [", sub_match("Pollinators/Predators"), "] Pollinators/Predators"),
               paste0("Entomology [", sub_match("Entomology"), "] Entomology"),
               paste0("Entomology [", sub_match("Insect Impacts on Ag Yields"), "] Insect Impacts on Ag Yields"),
               "",
               paste0("Wildlife [", sub_match("Habitat suitability"), "] Habitat suitability"),
               paste0("Wildlife [", sub_match("Impact on wildlife/habitats"), "] Impact on wildlife/habitats"),
               paste0("Wildlife [", sub_match("wildlife"), "] wildlife"),
               "",
               paste0("Human Health [", sub_match("Temperature"), "] Temperature"),
               paste0("Human Health [", sub_match("Sun Exposure"), "] Sun Exposure"),
               paste0("Human Health [", sub_match("Other health impacts"), "] Other"),
               "",
               paste0("Microclimate [", sub_match("Air Temperature"), "] Air Temperature"),
               paste0("Microclimate [", sub_match("Light and Shading"), "] Light and Shading"),
               paste0("Microclimate [", sub_match("Relative Humidity"), "] Relative Humidity"),
               paste0("Microclimate [", sub_match("Wind and airflow"), "] Wind"),
               paste0("Microclimate [", sub_match("PAR/PPFD"), "] PAR/PPFD"),
               "",
               paste0("Soil [", sub_match("Bulk Density/Compaction"), "] Bulk Density/Compaction"),
               paste0("Soil [", sub_match("Soil Temperature"), "] Soil Temperature"),
               paste0("Soil [", sub_match("Soil Management"), "] Soil Management"),
               paste0("Soil [", sub_match("Nutrients"), "] Nutrients"),
               paste0("Soil [", sub_match("Soil Carbon"), "] Soil Carbon"),
               paste0("Soil [", sub_match("Erosion"), "] Erosion"),
               "",
               paste0("Hydrology [", sub_match("Evapotranspiration"), "] Evapotranspiration"),
               paste0("Hydrology [", sub_match("Landscape level hydrology"), "] Landscape level hydrology"),
               paste0("Hydrology [", sub_match("soil water content"), "] soil water content"),
               paste0("Hydrology [", sub_match("Stormwater runoff"), "] Stormwater runoff"),
               "",
               paste0("Social Perspectives [", sub_match("Community Perspectives"), "] Community Perspectives"),
               paste0("Social Perspectives [", sub_match("Farmer/Landowner Perspectives"), "] Farmer/Landowner Perspectives"),
               paste0("Social Perspectives [", sub_match("Solar Industry Perspectives"), "] Solar Industry Perspectives"),
               paste0("Social Perspectives [", sub_match("Implementation Barriers"), "] Implementation Barriers"),
               paste0("Social Perspectives [", sub_match("Broader DEI and Social impacts"), "] Broader DEI and Social impacts"),
               "",
               paste0("Policy [", sub_match("Ag policies and regs"), "] Ag policies and regs"),
               paste0("Policy [", sub_match("Energy policies and regs"), "] Energy policies and regs"),
               paste0("Policy [", sub_match("Incentive structures"), "] Incentive structures"),
               paste0("Policy [", sub_match("Federal/state/county policies"), "] Federal/state/county policies"),
               "",
               paste0("Market [", sub_match("Technical potential"), "] Technical potential"),
               paste0("Market [", sub_match("Market potential"), "] Market potential"),
               paste0("Market [", sub_match("Agricultural supply chains"), "] Ag supply chain"),
               paste0("Market [", sub_match("Value propositions"), "] Value propositions"),
               "",
               paste0("Economics [", sub_match("Configuration/Climate/Crop Analysis"), "] Configuration/Climate/Crop Analysis"),
               paste0("Economics [", sub_match("Cost Benchmarks for O&M/CAPEX"), "] Cost Benchmarks for O&M/CAPEX"),
               paste0("Economics [", sub_match("Techno-economic analyses"), "] Techno-economic analyses"),
               paste0("Economics [", sub_match("Rural development impacts"), "] Rural development co-benefits"),
               "",
               paste0("PV Technology [", sub_match("Impact on energy generation"), "] Impact on energy generation"),
               paste0("PV Technology [", sub_match("Novel PV Materials"), "] Novel PV Materials"),
               paste0("PV Technology [", sub_match("Soiling"), "] Soiling"),
               paste0("PV Technology [", sub_match("Panel temperatures"), "] Panel Temperature"),
               "",
               paste0("Novel Configurations [", sub_match("Heights/spacing/layouts"), "] Heights/spacing/layouts"),
               paste0("Novel Configurations [", sub_match("Alternative racking designs"), "] Alternative Racking"),
               paste0("Novel Configurations [", sub_match("Compatibility with Farming"), "] Compatibility with Farming"),
               paste0("Novel Configurations [", sub_match("Tracking algorithms"), "] Tracking algorithms"),
               "",
               paste0("Siting [", sub_match("Site Suitability"), "] Site Suitability"),
               paste0("Siting [", sub_match("Siting Guidelines"), "] Siting Guidelines"),
               "",
               paste0("Impact Assessment [", sub_match("Environmental/Climate LCA"), "] Environmental/Climate LCA"),
               paste0("Impact Assessment [", sub_match("Food-Energy-Water Nexus"), "] Food-Energy-Water Nexus"),
               paste0("Impact Assessment [", sub_match("GHG Emissions/Reductions"), "] GHG Emissions/Reductions"),
               paste0("Impact Assessment [", sub_match("Land Impact/LER"), "] Land Impact/LER"),
               "",
               paste0("Tools [", sub_match("Tools"), "] Tools"),
               paste0("Standards and Best Practices [", sub_match("Standardization and Best Practices"), "] Standards and Best Practices"),
               paste0("Methodological Comparisons [", sub_match("Methodological Comparisons"), "] Methodological Comparisons"),
               paste0("Reviews/Informational [", sub_match("Reviews/Informational"), "] Reviews/Informational"))
    
    # Return
    lines
    
  })
  
  output$longTextOutput <- renderUI({
    
    # Ensure that CSV is downloaded and processed
    req(lines_out())
    
    # Extract lines for SankeyMATIC text
    line_input <- lines_out()
    
    # Create a full-page-width textbox
    tags$textarea(id = "longText", rows = 10, cols = 80, style = "width:100%;",
                  paste(line_input, collapse = "\n"))
  })
  
  output$downloadData <- downloadHandler(
    filename = function() {
      paste("InSPIRE_Data_Portal_counts", ".csv", sep = "")
    },
    content = function(file) {
      data_portal_csv <- data_portal_out()
      write.csv(data_portal_csv, file, row.names = FALSE)
    }
  )
  
  output$publicationByYear <- renderPlotly({
    
    # Ensure that CSV is downloaded and processed
    req(data_portal_out())
    
    # Grab topc and subtopic counts
    data_portal_in <- data_portal_out()
    
    # Plot a heatmap
    pubDate <- ggplot(data_portal_in[data_portal_in$Category == "Year",], aes(x = as.factor(Subject), y = Count, text = paste0("No.: ", Count))) + 
      geom_bar(stat="identity", width=0.7, fill="#436850") +
      theme_minimal() +
      labs(title = "", x = "Year", y = "No. of Publications")
    ggplotly(pubDate, tooltip = "text")
    
  })
  
  output$publicationByType <- renderPlotly({
    
    # Ensure that CSV is downloaded and processed
    req(data_portal_out())
    
    # Grab topc and subtopic counts
    data_portal_in <- data_portal_out()
    
    # Create figure in Plotly
    fig <- plot_ly(data_portal_in[data_portal_in$Category == "Document Type", 2:3], labels = ~Subject, values = ~Count, type = 'pie',
                   textposition = 'outside',
                   textinfo = 'label+percent',
                   insidetextfont = list(color = '#FFFFFF'),
                   hoverinfo = 'label+text',
                   text = ~paste(Count, 'Publications'),
                   marker = list(colors = viridis(length(data_portal_in[data_portal_in$Category == "Document Type", 2])),
                                 line = list(color = '#FFFFFF', width = 1)),
                   #The 'pull' attribute can also be used to create space between the sectors
                   showlegend = FALSE)
    
    fig <- fig %>% layout(title = '',
                          xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
                          yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))
    
    fig
    
  })
  
  output$publicationByStrategy <- renderPlotly({
    
    # Ensure that CSV is downloaded and processed
    req(data_portal_out())
    
    # Grab topc and subtopic counts
    data_portal_in <- data_portal_out()
    
    # Create figure in Plotly
    fig <- plot_ly(data_portal_in[data_portal_in$Category == "Low Impact Development Strategy", 2:3], labels = ~Subject, values = ~Count, type = 'pie',
                   textposition = 'outside',
                   textinfo = 'label+percent',
                   insidetextfont = list(color = '#FFFFFF'),
                   hoverinfo = 'label+text',
                   text = ~paste(Count, 'Publications'),
                   marker = list(colors = viridis(length(data_portal_in[data_portal_in$Category == "Document Type", 2])),
                                 line = list(color = '#FFFFFF', width = 1)),
                   #The 'pull' attribute can also be used to create space between the sectors
                   showlegend = FALSE)
    
    fig <- fig %>% layout(title = '',
                          xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
                          yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))
    
    fig
    
  })
  
}

shinyApp(ui, server)