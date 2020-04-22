library(ggplot2)
library(cowplot)
library(stringr)

l <- vector("list", 6)
df = read.csv('hiv1.csv')
sources = c("KEGG", "REAC", "WP")
text_size = 18
text_width = 30
text_angle = 0
for (i in 1:3){
  source = sources[i]
  subdf = df[which(df$source == source),]
  bp = ggplot(subdf[1:5,], aes(x=reorder(str_wrap(term_name,width = text_width), -adjusted_p_value), y=intersection_size)) +
    geom_bar(stat = "identity", aes(fill = adjusted_p_value)) +
    coord_flip() +labs(title="",x ="", y = "Counts") +
    theme(axis.text.y = element_text(size = text_size, angle = text_angle))+
    scale_fill_gradient(high="blue", low = "red", guide = guide_colourbar(title="adjusted pvalue", reverse = TRUE))
  l[[i]] = bp
}

df = read.csv('hepac.csv')
for (i in 1:3){
  source = sources[i]
  subdf = df[which(df$source == source),]
  bp = ggplot(subdf[1:5,], aes(x=reorder(str_wrap(term_name,width = text_width), -adjusted_p_value), y=intersection_size)) +
    geom_bar(stat = "identity", aes(fill = adjusted_p_value)) +
    coord_flip() +labs(title="",x ="", y = "Counts") + 
    theme(axis.text.y = element_text(size = text_size, angle = text_angle))+
    scale_fill_gradient(high="blue", low = "red", guide = guide_colourbar(title="adjusted pvalue", reverse = TRUE))
  l[[i+3]] = bp
}

labels = c("   (a)  KEGG (HIV 1)", "  (b) Reactome (HIV 1)", " (c)  WikiPathways (HIV 1)", "(d)  KEGG (Hepatitis C)", 
           "(e)  Reactome (Hepatitis C)", "(f)  WikiPathways (Hepatitis C)")
plot_grid(l[[1]], l[[2]], l[[3]], l[[4]], l[[5]], l[[6]], labels = labels, ncol = 3, nrow = 2, align = 'hv')


df = read.table('familywise.txt', header = TRUE)
ggplot(df, aes(x = reorder(Family, -ROCAUC), y=ROCAUC)) + geom_boxplot(aes(fill = Num_positives)) + 
  scale_fill_gradient(high="black", low = "grey", guide = guide_colourbar(title="Positives", reverse = TRUE)) + 
  theme_classic() + theme(axis.text.x = element_text(angle = 20, vjust = 0.6)) + labs(title="",x ="Family")
