# ========================================================================
# FINAL SDM CODE: Willow Warbler 2000 → 2020 Projection
# Author: Amy Cockburn-Pittner
# Description: Train model using 1998–2000 WorldClim data; project to 2005,2010,2015 CHELSA climate
# ========================================================================

# ========================
# STEP 0: SETUP
# ========================

# Clear workspace
rm(list = ls())
gc()

# Load required packages
library(terra)
library(dismo)
library(rJava)
library(sf)
library(ggplot2)
library(dplyr)
library(rnaturalearth)
library(rnaturalearthdata)
library(raster)

# Set Java memory
options(java.parameters = "-Xmx4g")

# Set working directory
setwd("/Users/amycockburn-pittner/Desktop/SDM_Project")

# Confirm Java and MaxEnt setup
Sys.setenv(JAVA_HOME = "/Library/Java/JavaVirtualMachines/jdk-21.jdk/Contents/Home")
if (!file.exists(Sys.getenv("JAVA_HOME"))) stop("JAVA_HOME path is incorrect.")
maxent()  # Should display MaxEnt usage info



# ========================================================================
# STEP 1: Load and Process Climate Data- WorldClim (1998–2000)
# ========================================================================

# Define file path for WorldClim monthly data
bioclim_path <- "/Users/amycockburn-pittner/Desktop/SDM_Project/data/"

# Load monthly .tif files (should have 36 layers: 12 months x 3 years)
pr_files_wc     <- sort(list.files(file.path(bioclim_path, "pr"), pattern = "\\.tif$", full.names = TRUE))
tasmax_files_wc <- sort(list.files(file.path(bioclim_path, "tasmax"), pattern = "\\.tif$", full.names = TRUE))
tasmin_files_wc <- sort(list.files(file.path(bioclim_path, "tasmin"), pattern = "\\.tif$", full.names = TRUE))

# Stack rasters
pr_stack     <- rast(pr_files_wc)
tasmax_stack <- rast(tasmax_files_wc)
tasmin_stack <- rast(tasmin_files_wc)

# Define UK extent and crop
uk_extent <- ext(-10, 2.5, 49, 61)

pr_crop     <- crop(pr_stack, uk_extent)
tasmax_crop <- crop(tasmax_stack, uk_extent)
tasmin_crop <- crop(tasmin_stack, uk_extent)

# Convert to raster::stack for biovars()
pr_r     <- raster::stack(pr_crop)
tasmax_r <- raster::stack(tasmax_crop)
tasmin_r <- raster::stack(tasmin_crop)

# Calculate bioclimatic variables per year 
bio_1998 <- biovars(pr_r[[1:12]],  tasmin_r[[1:12]],  tasmax_r[[1:12]])
bio_1999 <- biovars(pr_r[[13:24]], tasmin_r[[13:24]], tasmax_r[[13:24]])
bio_2000 <- biovars(pr_r[[25:36]], tasmin_r[[25:36]], tasmax_r[[25:36]])

# Average across 1998–2000 to match training period
mean_biovars <- (rast(bio_1998) + rast(bio_1999) + rast(bio_2000)) / 3
names(mean_biovars) <- paste0("bio", 1:19)

# Save raster stack to use for model training
writeRaster(mean_biovars, "bioclim_2000_mean_final.tif", overwrite = TRUE)


# ========================================================================
# STEP 2: Load and Clean Presence Data
# ========================================================================

# Load BTO presence points and their coordinates
presence_raw <- read.csv("data/presence_points.csv")
coords_raw   <- read.csv("data/coordinates.csv")

# Merge presence and coordinates on grid reference
presence_merged <- merge(presence_raw, coords_raw, by = "GRIDREF")

# Filter to: Willow Warbler, years 1998–2000, and valid UK extent
presence_data <- presence_merged %>%
  filter(
    YEAR %in% 1998:2000,
    ENGLISH_NAME == "Willow Warbler",
    between(LATITUDE, 49, 61),
    between(LONGITUDE, -10, 2)
  ) %>%
  select(LONGITUDE, LATITUDE)


# Sample 5000 Background Points from Valid Climate Cells
set.seed(42)  # for reproducibility
valid_cells <- which(!is.na(values(mean_biovars[[1]])))
bg_cells <- sample(valid_cells, 5000)
bg_points <- xyFromCell(mean_biovars[[1]], bg_cells)

presence_env <- terra::extract(mean_biovars, presence_vect)
bg_env       <- terra::extract(mean_biovars, bg_vect)


# Remove any rows with NA values
presence_clean <- presence_data[complete.cases(presence_env), ]
bg_clean       <- bg_points[complete.cases(bg_env), ]

# Optional checks
cat("Cleaned presence points:", nrow(presence_clean), "\n") #2166
cat("Cleaned background points:", nrow(bg_clean), "\n") #5000


# ========================================================================
# STEP 3: Train MaxEnt Model on 2000 Mean Climate (WorldClim)
# ========================================================================

# Make sure your bioclim raster is loaded (if not, load it again)
mean_biovars <- rast("bioclim_2000_mean_final.tif")
names(mean_biovars) <- paste0("bio", 1:19)  # Ensure naming is consistent


# Output directory for model results
output_dir_2000_allbio <- "Maxent_Model_2000_ALLBIO"
if (!dir.exists(output_dir_2000_allbio)) dir.create(output_dir_2000_allbio)

# Fit MaxEnt model using all 19 BIOCLIM variables
model_2000_allbio <- maxent(
  x = stack(mean_biovars),       # Environmental variables
  p = presence_clean,            # Presence points
  a = bg_clean,                  # Background points
  path = output_dir_2000_allbio, # Output folder
  args = c("jackknife=true")     # Variable importance plots
)

# Save prediction raster
prediction_2000_allbio <- predict(
  stack(mean_biovars),
  model_2000_allbio,
  args = c("outputformat=logistic")
)
writeRaster(prediction_2000_allbio, "willow_warbler_2000_allBIO.tif", overwrite = TRUE)

# Check variable contributions
results_allbio <- read.csv(file.path(output_dir_2000_allbio, "maxentResults.csv"))
contributions <- results_allbio[, grep("contribution", colnames(results_allbio))]
print(round(contributions, 2))


#Results
#bio1.contribution bio10.contribution bio11.contribution bio12.contribution bio13.contribution bio14.contribution bio15.contribution
#1              9.31               0.03               0.31               0.95               0.03               1.77               2.35
#bio16.contribution bio17.contribution bio18.contribution bio19.contribution bio2.contribution bio3.contribution bio4.contribution
#1               5.23               0.01               0.78               0.75              1.14             12.49             47.66
#bio5.contribution bio6.contribution bio7.contribution bio8.contribution bio9.contribution
#1             13.22              1.09              0.23              1.62              1.03



# ========================================================================
# STEP 4: Train Reduced MaxEnt Model (Top 5 BIOs from WorldClim) (dont change as this is training 2000 baseline)
# ========================================================================

# Define top 5 contributing variables (from full model contribution output)
top_bios_2000 <- c("bio4", "bio5", "bio3", "bio1", "bio16")

# Subset BIOCLIM stack to just those variables
selected_bios_2000 <- mean_biovars[[top_bios_2000]]

# Create output directory
output_dir_2000_top5 <- "Maxent_Model_2000_Top5BIOFINAL"
if (!dir.exists(output_dir_2000_top5)) dir.create(output_dir_2000_top5)

# Train MaxEnt model using top 5 variables
model_2000_top5 <- maxent(
  x = stack(selected_bios_2000),
  p = presence_clean,
  a = bg_clean,
  path = output_dir_2000_top5,
  args = c("jackknife=true")
)

# Predict habitat suitability
prediction_2000_top5 <- predict(
  stack(selected_bios_2000),
  model_2000_top5,
  args = c("outputformat=logistic")
)

# Save prediction raster
writeRaster(prediction_2000_top5, "willow_warbler_2000_top5BIOFINAL.tif", overwrite = TRUE)




# ========================================================================
# STEP 5: Load and Process CHELSA 2004–2005 Climate Data (for Projection)
# ========================================================================

# Paths to CHELSA data folders
chelsa_2004_path <- "/Users/amycockburn-pittner/Desktop/CHELSA_2004"
chelsa_2005_path <- "/Users/amycockburn-pittner/Desktop/2005_chelsa"

# -----------------------
# Load 2004 Monthly Files
# -----------------------
tasmax_2004_files <- sort(list.files(chelsa_2004_path, pattern = "tasmax.*2004.*\\.tif$", full.names = TRUE))
tasmin_2004_files <- sort(list.files(chelsa_2004_path, pattern = "tasmin.*2004.*\\.tif$", full.names = TRUE))
pr_2004_files     <- sort(list.files(chelsa_2004_path, pattern = "pr.*2004.*\\.tif$",     full.names = TRUE))

tasmax_2004 <- rast(tasmax_2004_files)
tasmin_2004 <- rast(tasmin_2004_files)
pr_2004     <- rast(pr_2004_files)

# -----------------------
# Load 2005 Monthly Files
# -----------------------
tasmax_2005_files <- sort(list.files(chelsa_2005_path, pattern = "tasmax.*2005.*\\.tif$", full.names = TRUE))
tasmin_2005_files <- sort(list.files(chelsa_2005_path, pattern = "tasmin.*2005.*\\.tif$", full.names = TRUE))
pr_2005_files     <- sort(list.files(chelsa_2005_path, pattern = "pr.*2005.*\\.tif$",     full.names = TRUE))

tasmax_2005 <- rast(tasmax_2005_files)
tasmin_2005 <- rast(tasmin_2005_files)
pr_2005     <- rast(pr_2005_files)

# -----------------------
# Crop all rasters to UK extent
# -----------------------
uk_extent <- ext(-10, 2.5, 49, 61)

crop_and_resample <- function(rast_obj) {
  cropped <- crop(rast_obj, uk_extent)
  resample(cropped, mean_biovars[[1]], method = "bilinear")
}

tasmax_2004_res <- crop_and_resample(tasmax_2004)
tasmin_2004_res <- crop_and_resample(tasmin_2004)
pr_2004_res     <- crop_and_resample(pr_2004)

tasmax_2005_res <- crop_and_resample(tasmax_2005)
tasmin_2005_res <- crop_and_resample(tasmin_2005)
pr_2005_res     <- crop_and_resample(pr_2005)

# -----------------------
# Unit Conversion (CHELSA)
# -----------------------
tasmax_2004_res <- (tasmax_2004_res / 10) - 273.15
tasmin_2004_res <- (tasmin_2004_res / 10) - 273.15
pr_2004_res     <- pr_2004_res / 10

tasmax_2005_res <- (tasmax_2005_res / 10) - 273.15
tasmin_2005_res <- (tasmin_2005_res / 10) - 273.15
pr_2005_res     <- pr_2005_res / 10

# -----------------------
# Calculate BIOCLIM Variables
# -----------------------
bio_2004 <- rast(biovars(raster::stack(pr_2004_res), raster::stack(tasmin_2004_res), raster::stack(tasmax_2004_res)))
bio_2005 <- rast(biovars(raster::stack(pr_2005_res), raster::stack(tasmin_2005_res), raster::stack(tasmax_2005_res)))

# -----------------------
# Average BIOCLIM: 2004–2005
# -----------------------
bio_2004_2005 <- (bio_2004 + bio_2005) / 2
names(bio_2004_2005) <- paste0("bio", 1:19)

# Save for projection step
writeRaster(bio_2004_2005, "bioclim_2004_2005_projection.tif", overwrite = TRUE)


# Load the raster stack as dont want to re-run above
bio_2004_2005 <- rast("bioclim_2004_2005_projection.tif")

# Optional: Check the names to confirm it loaded properly
names(bio_2004_2005)


# ========================================================================
# STEP 6: Project Reduced 2000 MaxEnt Model → CHELSA 2004–2005 Climate
# ========================================================================

# Load processed 2004–2005 CHELSA BIOCLIM variables
bio_2004_2005 <- rast("bioclim_2004_2005_projection.tif")
names(bio_2004_2005) <- paste0("bio", 1:19)  # Ensure consistent naming

# Subset to the same top 5 variables used in 2000 model
selected_2004_2005_top5 <- bio_2004_2005[[top_bios_2000]]

# Project the 2000 model to 2004–2005 CHELSA climate
projection_2004_2005_top5 <- predict(
  stack(selected_2004_2005_top5),
  model_2000_top5,
  args = c("outputformat=logistic")
)

# Save projection raster
writeRaster(projection_2004_2005_top5, "willow_warbler_projection_2004_2005_top5BIO.tif", overwrite = TRUE)

# Convert raster to dataframe for plotting
proj_df_2004_2005 <- as.data.frame(projection_2004_2005_top5, xy = TRUE, na.rm = TRUE)
colnames(proj_df_2004_2005)[3] <- "suitability"

uk_outline <- ne_countries(scale = "medium", returnclass = "sf") %>%
  dplyr::filter(sovereignt == "United Kingdom")



# Plot habitat suitability: 2004–2005
ggplot() +
  geom_raster(data = proj_df_2004_2005, aes(x = x, y = y, fill = suitability)) +
  geom_sf(data = uk_outline, fill = NA, color = "white", linewidth = 0.3) +
  scale_fill_viridis_c(
    name = "Habitat Suitability",
    limits = c(0, 0.4),  # max = 0.326
    option = "D",
    oob = scales::squish
  ) +
  coord_sf(xlim = c(-10, 2.5), ylim = c(49, 61), expand = FALSE) +
  labs(
    title = "Willow Warbler Habitat Suitability (2004–2005)",
    subtitle = "Projected using Reduced 2000 Model (Top 5 BIOs)",
    x = "Longitude", y = "Latitude"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5),
    legend.position = "right"
  )


# ========================================================================
# STEP 7: Evaluate 2004–2005 Projection with Willow Warbler Presence Data
# ========================================================================

# Load and filter presence data for 2005
presence_2005 <- presence_merged %>%
  filter(YEAR == 2005, ENGLISH_NAME == "Willow Warbler",
         between(LATITUDE, 49, 61), between(LONGITUDE, -10, 2)) %>%
  select(LONGITUDE, LATITUDE)

# Convert to SpatVector using same CRS as projection
crs_proj_2005 <- as.character(crs(projection_2004_2005_top5))
presence_2005_vect <- vect(presence_2005, geom = c("LONGITUDE", "LATITUDE"), crs = crs_proj_2005)
bg_vect_2005 <- vect(bg_clean, type = "points", crs = crs_proj_2005)

# Convert RasterLayer to SpatRaster
projection_2004_2005_top5 <- rast(projection_2004_2005_top5)


# Extract suitability values for presence (2005 data)
presence_vals_2005 <- terra::extract(projection_2004_2005_top5, presence_2005_vect)[, 2]

# Extract suitability values for background points
bg_vals_2005 <- terra::extract(projection_2004_2005_top5, bg_vect_2005)[, 2]

# Check the extracted values
head(presence_vals_2005)
head(bg_vals_2005)


# Run evaluation
eval_2004_2005 <- evaluate(p = na.omit(presence_vals_2005), a = na.omit(bg_vals_2005))

# Calculate threshold and TSS
threshold_2004_2005 <- threshold(eval_2004_2005, "spec_sens")
tss_2004_2005 <- max(eval_2004_2005@TPR + eval_2004_2005@TNR - 1, na.rm = TRUE)

# Output results
cat("Evaluation (2000 → 2004–2005 Projection):\n")
cat("AUC:", round(eval_2004_2005@auc, 3), "\n")
cat("TSS:", round(tss_2004_2005, 3), "\n")
cat("Threshold (spec_sens):", round(threshold_2004_2005, 3), "\n")

#Results
#cat("Evaluation (2000 → 2004–2005 Projection):\n")
#Evaluation (2000 → 2004–2005 Projection):
 # > cat("AUC:", round(eval_2004_2005@auc, 3), "\n")
#AUC: 0.655 
#> cat("TSS:", round(tss_2004_2005, 3), "\n")
#TSS: 0.221 
#> cat("Threshold (spec_sens):", round(threshold_2004_2005, 3), "\n")
#Threshold (spec_sens): 0.036 


# ========================================================================
# STEP 8: Load and Process CHELSA 2009–2010 Climate Data (for Projection)
# ========================================================================

# Paths to CHELSA data folders
chelsa_2009_path <- "/Users/amycockburn-pittner/Desktop/chelsaV2/GLOBAL/monthly"
chelsa_2010_path <- "/Users/amycockburn-pittner/Desktop/chelsaV2/GLOBAL/monthly"

# -----------------------
# Load 2009 Monthly Files
# -----------------------
tasmax_2009_files <- sort(list.files(file.path(chelsa_2009_path, "tasmax"), pattern = "2009.*\\.tif$", full.names = TRUE))
tasmin_2009_files <- sort(list.files(file.path(chelsa_2009_path, "tasmin"), pattern = "2009.*\\.tif$", full.names = TRUE))
pr_2009_files     <- sort(list.files(file.path(chelsa_2009_path, "pr"),     pattern = "2009.*\\.tif$", full.names = TRUE))

tasmax_2009 <- rast(tasmax_2009_files)
tasmin_2009 <- rast(tasmin_2009_files)
pr_2009     <- rast(pr_2009_files)

# -----------------------
# Load 2010 Monthly Files
# -----------------------
tasmax_2010_files <- sort(list.files(file.path(chelsa_2010_path, "tasmax"), pattern = "2010.*\\.tif$", full.names = TRUE))
tasmin_2010_files <- sort(list.files(file.path(chelsa_2010_path, "tasmin"), pattern = "2010.*\\.tif$", full.names = TRUE))
pr_2010_files     <- sort(list.files(file.path(chelsa_2010_path, "pr"),     pattern = "2010.*\\.tif$", full.names = TRUE))

tasmax_2010 <- rast(tasmax_2010_files)
tasmin_2010 <- rast(tasmin_2010_files)
pr_2010     <- rast(pr_2010_files)

# -----------------------
# Crop all rasters to UK extent
# -----------------------
uk_extent <- ext(-10, 2.5, 49, 61)

crop_and_resample <- function(rast_obj) {
  cropped <- crop(rast_obj, uk_extent)
  resample(cropped, mean_biovars[[1]], method = "bilinear")
}

tasmax_2009_res <- crop_and_resample(tasmax_2009)
tasmin_2009_res <- crop_and_resample(tasmin_2009)
pr_2009_res     <- crop_and_resample(pr_2009)

tasmax_2010_res <- crop_and_resample(tasmax_2010)
tasmin_2010_res <- crop_and_resample(tasmin_2010)
pr_2010_res     <- crop_and_resample(pr_2010)

# -----------------------
# Unit Conversion (CHELSA)
# -----------------------
tasmax_2009_res <- (tasmax_2009_res / 10) - 273.15
tasmin_2009_res <- (tasmin_2009_res / 10) - 273.15
pr_2009_res     <- pr_2009_res / 10

tasmax_2010_res <- (tasmax_2010_res / 10) - 273.15
tasmin_2010_res <- (tasmin_2010_res / 10) - 273.15
pr_2010_res     <- pr_2010_res / 10

# -----------------------
# Calculate BIOCLIM Variables
# -----------------------
bio_2009 <- rast(biovars(raster::stack(pr_2009_res), raster::stack(tasmin_2009_res), raster::stack(tasmax_2009_res)))
bio_2010 <- rast(biovars(raster::stack(pr_2010_res), raster::stack(tasmin_2010_res), raster::stack(tasmax_2010_res)))

# -----------------------
# Average BIOCLIM: 2009–2010
# -----------------------
bio_2009_2010 <- (bio_2009 + bio_2010) / 2
names(bio_2009_2010) <- paste0("bio", 1:19)

# Save for projection step
writeRaster(bio_2009_2010, "bioclim_2009_2010_projection.tif", overwrite = TRUE)


# Load the raster stack if already ran code and saved
bio_2009_2010 <- rast("bioclim_2009_2010_projection.tif")



# ========================================================================
# STEP 9: Project Reduced 2000 MaxEnt Model → CHELSA 2009–2010 Climate
# ========================================================================


# Load 2009–2010 BIOCLIM raster
bio_2009_2010 <- rast("bioclim_2009_2010_projection.tif")
names(bio_2009_2010) <- paste0("bio", 1:19)

# Subset to the same top 5 BIO variables
selected_2009_2010_top5 <- bio_2009_2010[[top_bios_2000]]


# Project using trained 2000 model
projection_2009_2010_top5 <- predict(
  stack(selected_2009_2010_top5),
  model_2000_top5,
  args = c("outputformat=logistic")
)

# Save projection
writeRaster(projection_2009_2010_top5, "willow_warbler_projection_2009_2010_top5BIO.tif", overwrite = TRUE)


# Check range of suitability values for projection_2009_2010
summary(projection_2009_2010_top5)


# Convert raster to dataframe for plotting
proj_df_2009_2010 <- as.data.frame(projection_2009_2010_top5, xy = TRUE, na.rm = TRUE)
colnames(proj_df_2009_2010)[3] <- "suitability"

# Adjust plot color scale based on the actual data range
ggplot() +
  geom_raster(data = proj_df_2009_2010, aes(x = x, y = y, fill = suitability)) +
  geom_sf(data = uk_outline, fill = NA, color = "white", linewidth = 0.3) +
  scale_fill_viridis_c(
    name = "Habitat Suitability",
    limits = c(0, 0.3),  # Adjusted limits to match the range of the suitability values
    option = "D",
    oob = scales::squish
  ) +
  coord_sf(xlim = c(-10, 2.5), ylim = c(49, 61), expand = FALSE) +
  labs(
    title = "Willow Warbler Habitat Suitability (2009–2010)",
    subtitle = "Projected using Reduced 2000 Model (Top 5 BIOs)",
    x = "Longitude", y = "Latitude"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5),
    legend.position = "right"
  )


# Evaluate 2000 → 2009–2010 Projection (Top 5 BIOs)
# Filter presence data for 2010 (Willow Warbler in UK)
presence_2010 <- presence_merged %>%
  filter(YEAR == 2010, ENGLISH_NAME == "Willow Warbler",
         between(LATITUDE, 49, 61), between(LONGITUDE, -10, 2)) %>%
  select(LONGITUDE, LATITUDE)

# Load 2009–2010 projection raster and check CRS
projection_2009_2010 <- rast("willow_warbler_projection_2009_2010_top5BIO.tif")
crs_proj_2010 <- crs(projection_2009_2010)

# Convert presence and background data to SpatVector format (matching raster CRS)
presence_2010_vect <- vect(presence_2010, geom = c("LONGITUDE", "LATITUDE"), crs = crs_proj_2010)
bg_vect_proj <- vect(bg_clean, type = "points", crs = crs_proj_2010)

# Extract suitability values for presence (2010 data)
presence_vals_2010_raw <- terra::extract(projection_2009_2010, presence_2010_vect)

# Rename the second column to "suitability" for consistency
colnames(presence_vals_2010_raw)[2] <- "suitability"

# Extract the suitability values for presence
presence_vals_2010 <- presence_vals_2010_raw$suitability

# Check the extracted values
head(presence_vals_2010)

# Extract suitability values for background points (2010 data)
bg_vals_2010_raw <- terra::extract(projection_2009_2010, bg_vect_proj)

# Rename the column for background values
colnames(bg_vals_2010_raw)[2] <- "suitability"

# Extract the suitability values for background
bg_vals_2010 <- bg_vals_2010_raw$suitability

# Check the extracted background values
head(bg_vals_2010)

# Now you can proceed with evaluation
eval_proj_2010 <- dismo::evaluate(p = presence_vals_2010, a = bg_vals_2010)
threshold_proj_2010 <- threshold(eval_proj_2010, "spec_sens")
tss_proj_2010 <- max(eval_proj_2010@TPR + eval_proj_2010@TNR - 1, na.rm = TRUE)

# Output the evaluation results
cat("Evaluation (2000 → 2009–2010 Projection):\n")
cat("AUC:", round(eval_proj_2010@auc, 3), "\n")
cat("TSS:", round(tss_proj_2010, 3), "\n")
cat("Threshold (spec_sens):", round(threshold_proj_2010, 3), "\n")

#Results
# AUC 0.466
#TSS 0.106
# Threshold 0

# ========================================================================
# STEP 10: Load and Process CHELSA 2014–2015 Climate Data (for Projection)
# ========================================================================
# Paths to CHELSA data folders
chelsa_2014_path <- "/Users/amycockburn-pittner/Desktop/CHELSA_2014_2015/chelsav2/GLOBAL/monthly"
chelsa_2015_path <- "/Users/amycockburn-pittner/Desktop/CHELSA_2014_2015/chelsav2/GLOBAL/monthly"


# Load 2014 Monthly Files
tasmax_2014_files <- sort(list.files(file.path(chelsa_2014_path, "tasmax"), pattern = "2014.*\\.tif$", full.names = TRUE))
tasmin_2014_files <- sort(list.files(file.path(chelsa_2014_path, "tasmin"), pattern = "2014.*\\.tif$", full.names = TRUE))
pr_2014_files     <- sort(list.files(file.path(chelsa_2014_path, "pr"),     pattern = "2014.*\\.tif$", full.names = TRUE))

tasmax_2014 <- rast(tasmax_2014_files)
tasmin_2014 <- rast(tasmin_2014_files)
pr_2014     <- rast(pr_2014_files)

# Load 2015 Monthly Files
tasmax_2015_files <- sort(list.files(file.path(chelsa_2015_path, "tasmax"), pattern = "2015.*\\.tif$", full.names = TRUE))
tasmin_2015_files <- sort(list.files(file.path(chelsa_2015_path, "tasmin"), pattern = "2015.*\\.tif$", full.names = TRUE))
pr_2015_files     <- sort(list.files(file.path(chelsa_2015_path, "pr"),     pattern = "2015.*\\.tif$", full.names = TRUE))

tasmax_2015 <- rast(tasmax_2015_files)
tasmin_2015 <- rast(tasmin_2015_files)
pr_2015     <- rast(pr_2015_files)

# Crop and resample to match UK training extent
uk_extent <- ext(-10, 2.5, 49, 61)
crop_and_resample <- function(r) resample(crop(r, uk_extent), mean_biovars[[1]], method = "bilinear")

tasmax_2014_res <- crop_and_resample(tasmax_2014)
tasmin_2014_res <- crop_and_resample(tasmin_2014)
pr_2014_res     <- crop_and_resample(pr_2014)

tasmax_2015_res <- crop_and_resample(tasmax_2015)
tasmin_2015_res <- crop_and_resample(tasmin_2015)
pr_2015_res     <- crop_and_resample(pr_2015)

# Unit conversion
tasmax_2014_res <- (tasmax_2014_res / 10) - 273.15
tasmin_2014_res <- (tasmin_2014_res / 10) - 273.15
pr_2014_res     <- pr_2014_res / 10

tasmax_2015_res <- (tasmax_2015_res / 10) - 273.15
tasmin_2015_res <- (tasmin_2015_res / 10) - 273.15
pr_2015_res     <- pr_2015_res / 10

# Calculate BIOCLIM
bio_2014 <- rast(biovars(stack(pr_2014_res), stack(tasmin_2014_res), stack(tasmax_2014_res)))
bio_2015 <- rast(biovars(stack(pr_2015_res), stack(tasmin_2015_res), stack(tasmax_2015_res)))

# Average BIOCLIM
bio_2014_2015 <- (bio_2014 + bio_2015) / 2
names(bio_2014_2015) <- paste0("bio", 1:19)

# Save
writeRaster(bio_2014_2015, "bioclim_2014_2015_projection.tif", overwrite = TRUE)

# Visual check
plot(bio_2014_2015[[1]], main = "BIO1: Mean Annual Temp (2014–2015 Avg)")

# If already ran above just re-load to save time
bio_2014_2015 <- rast("bioclim_2014_2015_projection.tif")

# ========================================================================
# STEP 11: Project Reduced 2000 MaxEnt Model → CHELSA 2014–2015 Climate
# ========================================================================

# Load BIOCLIM projection
bio_2014_2015 <- rast("bioclim_2014_2015_projection.tif")
names(bio_2014_2015) <- paste0("bio", 1:19)

# Subset top 5 variables
selected_2014_2015_top5 <- bio_2014_2015[[top_bios_2000]]

# Predict
projection_2014_2015_top5 <- predict(
  stack(selected_2014_2015_top5),
  model_2000_top5,
  args = c("outputformat=logistic")
)

# Save projection
writeRaster(projection_2014_2015_top5, "willow_warbler_projection_2014_2015_top5BIO.tif", overwrite = TRUE)

# Convert the raster to a dataframe for plotting
proj_df_2014_2015 <- as.data.frame(projection_2014_2015_top5, xy = TRUE, na.rm = TRUE)
colnames(proj_df_2014_2015)[3] <- "suitability"

# Plot habitat suitability: 2014–2015
ggplot() +
  geom_raster(data = proj_df_2014_2015, aes(x = x, y = y, fill = suitability)) +
  geom_sf(data = uk_outline, fill = NA, color = "white", linewidth = 0.3) +
  scale_fill_viridis_c(
    name = "Habitat Suitability",
    limits = c(0, 0.4),  # max = 0.326
    option = "D",
    oob = scales::squish
  ) +
  coord_sf(xlim = c(-10, 2.5), ylim = c(49, 61), expand = FALSE) +
  labs(
    title = "Willow Warbler Habitat Suitability (2014–2015)",
    subtitle = "Projected using Reduced 2000 Model (Top 5 BIOs)",
    x = "Longitude", y = "Latitude"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5),
    legend.position = "right"
  )


# ========================================================================
# STEP 12: Evaluate 2000 → 2014–2015 Projection (Top 5 BIOs)
# ========================================================================

# Filter Willow Warbler presence data for 2015
presence_2015 <- presence_merged %>%
  filter(YEAR == 2015, ENGLISH_NAME == "Willow Warbler",
         between(LATITUDE, 49, 61), between(LONGITUDE, -10, 2)) %>%
  select(LONGITUDE, LATITUDE)

# Load the 2014–2015 projection raster
projection_2014_2015 <- rast("willow_warbler_projection_2014_2015_top5BIO.tif")

# Match CRS and convert to SpatVector
crs_proj_2015 <- as.character(crs(projection_2014_2015))
presence_2015_vect <- vect(presence_2015, geom = c("LONGITUDE", "LATITUDE"), crs = crs_proj_2015)
bg_vect_2015 <- vect(bg_clean, type = "points", crs = crs_proj_2015)

# Extract suitability values for presence points
presence_vals_proj_2015 <- terra::extract(projection_2014_2015, presence_2015_vect)[, 2]

# Extract suitability values for background points
bg_vals_proj_2015 <- terra::extract(projection_2014_2015, bg_vect_2015)[, 2]

# Check if the extraction was successful by inspecting the extracted values
head(presence_vals_proj_2015)
head(bg_vals_proj_2015)

# Evaluate performance
eval_proj_2015 <- dismo::evaluate(p = presence_vals_proj_2015, a = bg_vals_proj_2015)
threshold_proj_2015 <- threshold(eval_proj_2015, "spec_sens")
tss_proj_2015 <- max(eval_proj_2015@TPR + eval_proj_2015@TNR - 1, na.rm = TRUE)

# Output evaluation results
cat("Evaluation (2000 → 2014–2015 Projection):\n")
cat("AUC:", round(eval_proj_2015@auc, 3), "\n")
cat("TSS:", round(tss_proj_2015, 3), "\n")
cat("Threshold (spec_sens):", round(threshold_proj_2015, 3), "\n")

#Results
#AUC: 0.664 
#TSS: 0.269
#Threshold (spec_sens): 0.13


# ========================================================================
# STEP 12: Project Reduced 2000 MaxEnt Model → CHELSA 2020 Climate
# ========================================================================

# Load previously saved 2020 BIOCLIM raster from CHELSA
bio_2020 <- rast("bioclim_2020_projection.tif")
names(bio_2020) <- paste0("bio", 1:19)  # Ensure naming consistency
summary(bio_2020)

# Subset to the same top 5 BIO variables used in 2000 model
selected_2020_top5 <- bio_2020[[top_bios_2000]]
summary(selected_2020_top5)


# Project the 2000 model to 2020 conditions
projection_2020_top5 <- predict(
  stack(selected_2020_top5),
  model_2000_top5,
  args = c("outputformat=logistic")
)

# Save the 2020 projection raster
writeRaster(projection_2020_top5, "willow_warbler_projection_2020_top5BIO.tif", overwrite = TRUE)

# Optional: Plot with ggplot2
proj_df_2020 <- as.data.frame(projection_2020_top5, xy = TRUE, na.rm = TRUE)
colnames(proj_df_2020)[3] <- "suitability"


# Plot habitat suitability: 2020
ggplot() +
  geom_raster(data = proj_df_2020, aes(x = x, y = y, fill = suitability)) +
  geom_sf(data = uk_outline, fill = NA, color = "white", linewidth = 0.3) +
  scale_fill_viridis_c(
    name = "Habitat Suitability",
    limits = c(0, 0.4),  # max = 0.239
    option = "D",
    oob = scales::squish
  ) +
  coord_sf(xlim = c(-10, 2.5), ylim = c(49, 61), expand = FALSE) +
  labs(
    title = "Willow Warbler Habitat Suitability (2020)",
    subtitle = "Projected using Reduced 2000 Model (Top 5 BIOs)",
    x = "Longitude", y = "Latitude"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5),
    legend.position = "right"
  )

# ========================================================================
# STEP 13: Evaluate 2000 → 2020 Projection (Top 5 BIOs)
# ========================================================================
# Filter Willow Warbler presence data for 2019 (to match CHELSA 2020 range)
presence_2020 <- presence_merged %>%
  filter(YEAR == 2019, ENGLISH_NAME == "Willow Warbler",
         between(LATITUDE, 49, 61), between(LONGITUDE, -10, 2)) %>%
  select(LONGITUDE, LATITUDE)

# Load 2020 projection raster
projection_2020 <- rast("willow_warbler_projection_2020_top5BIO.tif")

# Convert presence and background points to SpatVector with correct CRS
crs_proj_2020 <- as.character(crs(projection_2020))
presence_2020_vect <- vect(presence_2020, geom = c("LONGITUDE", "LATITUDE"), crs = crs_proj_2020)
bg_vect_proj <- vect(bg_clean, type = "points", crs = crs_proj_2020)

# Extract suitability values for presence points in 2020
presence_vals_proj2020 <- terra::extract(projection_2020, presence_2020_vect)[, 2]

# Extract suitability values for background points in 2020
bg_vals_proj2020 <- terra::extract(projection_2020, bg_vect_proj)[, 2]

# Check if the extraction was successful by inspecting the extracted values
head(presence_vals_proj2020)
head(bg_vals_proj2020)

# Run evaluation
eval_proj2020 <- dismo::evaluate(p = presence_vals_proj2020, a = bg_vals_proj2020)
threshold_proj2020 <- threshold(eval_proj2020, "spec_sens")
tss_proj2020 <- max(eval_proj2020@TPR + eval_proj2020@TNR - 1, na.rm = TRUE)

# Output results
cat("Evaluation (2000 → 2020 Projection):\n")
cat("AUC:", round(eval_proj2020@auc, 3), "\n")
cat("TSS:", round(tss_proj2020, 3), "\n")
cat("Threshold (spec_sens):", round(threshold_proj2020, 3), "\n")

#Results
# AUC 0.641
# TSS 0.222
# threshold 0.093

# ========================================================================
# STEP 14 : All plots together (Facet Plot)
# ========================================================================

# Maximum suitability value across years ~0.326
# So we set scale limits = c(0, 0.4) for consistent legend across years

# Load UK outline for map
uk_outline <- ne_countries(scale = "medium", returnclass = "sf") %>%
  filter(sovereignt == "United Kingdom")

# Helper function to convert raster files to dataframes for plotting
raster_to_df <- function(file, label) {
  r <- rast(file)
  df <- as.data.frame(r, xy = TRUE, na.rm = TRUE)
  colnames(df)[3] <- "suitability"
  df$Period <- label
  return(df)
}

# Load and label all projection rasters
df_2005 <- raster_to_df("willow_warbler_projection_2004_2005_top5BIO.tif", "2004–2005")
df_2010 <- raster_to_df("willow_warbler_projection_2009_2010_top5BIO.tif", "2009–2010")
df_2015 <- raster_to_df("willow_warbler_projection_2014_2015_top5BIO.tif", "2014–2015")
df_2020 <- raster_to_df("willow_warbler_projection_2020_top5BIO.tif", "2020")

#Combine all years into one dataframe
suitability_all <- bind_rows(df_2005, df_2010, df_2015, df_2020)

#  Plot all together using facet_wrap
ggplot() +
  geom_raster(data = suitability_all, aes(x = x, y = y, fill = suitability)) +
  geom_sf(data = uk_outline, fill = NA, color = "white", linewidth = 0.3) +
  scale_fill_viridis_c(
    name = "Habitat Suitability",
    limits = c(0, 0.4),  # 🔧 Adjusted based on max suitability (0.326)
    option = "D",
    oob = scales::squish
  ) +
  facet_wrap(~Period) +
  coord_sf(xlim = c(-10, 2.5), ylim = c(49, 61), expand = FALSE) +
  labs(
    title = "Willow Warbler Habitat Suitability Projections",
    subtitle = "Projected using Reduced 2000 MaxEnt Model (Top 5 BIOs)",
    x = "Longitude", y = "Latitude"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5),
    legend.position = "right"
  )
