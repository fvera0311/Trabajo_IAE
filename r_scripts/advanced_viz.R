# Script R para visualizaciones avanzadas con ggplot2
# advanced_viz.R - Visualizaciones interactivas con plotly
library(tidyverse)
library(gridExtra)

args <- commandArgs(trailingOnly = TRUE)
csv_path <- args[1]
output_dir <- args[2]

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
df <- read_csv(csv_path, show_col_types = FALSE) %>%
  mutate(Estado = factor(DEATH_EVENT, levels = c(0, 1), labels = c("Sobrevivió", "Falleció")))

p1 <- ggplot(df, aes(x = age, fill = Estado)) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("#2ecc71", "#e74c3c")) +
  labs(title = "Distribución de Edad", x = "Edad", y = "Densidad") + theme_minimal()

p2 <- ggplot(df, aes(x = Estado, y = ejection_fraction, fill = Estado)) +
  geom_violin(alpha = 0.5) +
  scale_fill_manual(values = c("#2ecc71", "#e74c3c")) +
  labs(title = "Fracción de Eyección", x = "", y = "Fracción (%)") + theme_minimal()

g <- arrangeGrob(p1, p2, ncol = 2)
ggsave(file.path(output_dir, "r_distributions.png"), g, width = 12, height = 6, dpi = 300)