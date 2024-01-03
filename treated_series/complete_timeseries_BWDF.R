#install.packages('imputeTS')
#install.packages('readxl')
library(imputeTS)
library(readxl)
library(ggplot2)

defined_gapsize_limit=1200


df<-read_excel("InflowData_1.xlsx")
summary(df)


ts.plot(df$`DMA A (L/s)`)


df$`DMA A (L/s)`<-as.numeric(df$`DMA A (L/s)`)
df$`DMA B (L/s)`<-as.numeric(df$`DMA B (L/s)`)
df$`DMA C (L/s)`<-as.numeric(df$`DMA C (L/s)`)
df$`DMA D (L/s)`<-as.numeric(df$`DMA D (L/s)`)
df$`DMA E (L/s)`<-as.numeric(df$`DMA E (L/s)`)
df$`DMA F (L/s)`<-as.numeric(df$`DMA F (L/s)`)
df$`DMA G (L/s)`<-as.numeric(df$`DMA G (L/s)`)
df$`DMA H (L/s)`<-as.numeric(df$`DMA H (L/s)`)
df$`DMA I (L/s)`<-as.numeric(df$`DMA I (L/s)`)
df$`DMA J (L/s)`<-as.numeric(df$`DMA J (L/s)`)

# some stats on the NA for each variable
statsNA(df$`DMA A (L/s)`, print_only=T)
statsNA(df$`DMA B (L/s)`, print_only=T)
statsNA(df$`DMA C (L/s)`, print_only=T)
statsNA(df$`DMA D (L/s)`, print_only=T)
statsNA(df$`DMA E (L/s)`, print_only=T)
statsNA(df$`DMA F (L/s)`, print_only=T)
statsNA(df$`DMA G (L/s)`, print_only=T)
statsNA(df$`DMA H (L/s)`, print_only=T)
statsNA(df$`DMA I (L/s)`, print_only=T)
statsNA(df$`DMA J (L/s)`, print_only=T)
# To visualize the missing values in these time series 

ggplot_na_distribution(df$`DMA A (L/s)`)
ggplot_na_distribution(df$`DMA B (L/s)`)
ggplot_na_distribution(df$`DMA C (L/s)`)
ggplot_na_distribution(df$`DMA D (L/s)`)
ggplot_na_distribution(df$`DMA E (L/s)`)
ggplot_na_distribution(df$`DMA F (L/s)`)
ggplot_na_distribution(df$`DMA G (L/s)`)
ggplot_na_distribution(df$`DMA H (L/s)`)
ggplot_na_distribution(df$`DMA I (L/s)`)
ggplot_na_distribution(df$`DMA J (L/s)`)


df_dma_a_complete<- na_seadec(df$`DMA A (L/s)`, algorithm="ma", find_frequency = TRUE, maxgap=defined_gapsize_limit)
ggplot_na_imputations(df$`DMA A (L/s)`, df_dma_a_complete, subtitle="DMA A")

df_dma_b_complete<- na_seadec(df$`DMA B (L/s)`, algorithm="ma", find_frequency = TRUE, maxgap=defined_gapsize_limit)
ggplot_na_imputations(df$`DMA B (L/s)`, df_dma_b_complete, subtitle="DMA B")


df_dma_c_complete<- na_seadec(df$`DMA C (L/s)`, algorithm="ma", find_frequency = TRUE, maxgap=defined_gapsize_limit)
ggplot_na_imputations(df$`DMA C (L/s)`, df_dma_c_complete, subtitle="DMA C")

df_dma_d_complete<- na_seadec(df$`DMA D (L/s)`, algorithm="ma", find_frequency = TRUE, maxgap=defined_gapsize_limit)
ggplot_na_imputations(df$`DMA D (L/s)`, df_dma_d_complete, subtitle="DMA D")

df_dma_e_complete<- na_seadec(df$`DMA E (L/s)`, algorithm="ma", find_frequency = TRUE, maxgap=defined_gapsize_limit)
ggplot_na_imputations(df$`DMA E (L/s)`, df_dma_e_complete, subtitle="DMA E")

#this one doesn't look good
df_dma_f_complete<- na_seadec(df$`DMA F (L/s)`, algorithm="ma", find_frequency = TRUE, maxgap=defined_gapsize_limit)
#df_dma_f_complete<- na_seadec(df$`DMA F (L/s)`, algorithm="interpolation", find_frequency = TRUE, maxgap=defined_gapsize_limit)
ggplot_na_imputations(df$`DMA F (L/s)`, df_dma_f_complete, subtitle="DMA F")


#this one doesn't look good
df_dma_g_complete<- na_seadec(df$`DMA G (L/s)`, algorithm="ma", find_frequency = TRUE, maxgap=defined_gapsize_limit)
ggplot_na_imputations(df$`DMA G (L/s)`, df_dma_g_complete, subtitle="DMA G")

df_dma_h_complete<- na_seadec(df$`DMA H (L/s)`, algorithm="ma", find_frequency = TRUE, maxgap=defined_gapsize_limit)
ggplot_na_imputations(df$`DMA H (L/s)`, df_dma_h_complete, subtitle="DMA H")

df_dma_i_complete<- na_seadec(df$`DMA I (L/s)`, algorithm="ma", find_frequency = TRUE, maxgap=defined_gapsize_limit)
#kalman takes too long and results are not better than ma
#df_dma_i_complete<- na_seadec(df$`DMA I (L/s)`, algorithm="kalman", find_frequency = TRUE, maxgap=defined_gapsize_limit)
ggplot_na_imputations(df$`DMA I (L/s)`, df_dma_i_complete, subtitle="DMA I")

df_dma_j_complete<- na_seadec(df$`DMA J (L/s)`, algorithm="ma", find_frequency = TRUE, maxgap=defined_gapsize_limit)
ggplot_na_imputations(df$`DMA J (L/s)`, df_dma_j_complete, subtitle="DMA J")
?na_seadec

df_complete<-data.frame(df$`Date-time CET-CEST (DD/MM/YYYY HH:mm)`,df_dma_a_complete,df_dma_b_complete,df_dma_c_complete,df_dma_d_complete,df_dma_e_complete,df_dma_f_complete, df_dma_g_complete, df_dma_h_complete, df_dma_i_complete, df_dma_j_complete)
colnames(df_complete)<-c("datetime", "dma_A","dma_B", "dma_C", "dma_D","dma_E", "dma_F","dma_G","dma_H","dma_I", "dma_J")

write.csv(df_complete,"inflow_completed_in_R.csv")
