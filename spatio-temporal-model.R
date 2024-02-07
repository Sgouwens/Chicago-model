# Unfinished

library(tidyverse)
library(mgcv)
library(lme4)
library(sp)
library(gstat)
library(INLA)
library(broom)

weekend_day = c('Friday', "Saturday", "Sunday")

### Loading Chicago crime data (preprocessed in Python)
data = read.csv("./data/chicago-crime-preprocessed.csv",
                colClasses=c(Date="POSIXct")) %>% 
  filter(Primary.Type == 'THEFT') %>% 
  select(Date, Latitude_mid, Longitude_mid, Count)

### Completing data, obtaining useful derivatives of data 2.4M
data = data %>% 
  complete(Date, Latitude_mid, Longitude_mid,
           fill = list(Count=0)) %>% 
  mutate(Count = replace_na(Count, 0),
         year = year(Date),
         day = weekdays(Date),
         day_of_year = yday(Date),
         Weekend = factor(ifelse(day %in% weekend_day, "yes", "no"))) %>% 
  filter(between(year, 2005, 2018),
         between(Latitude_mid, 41.765, 41.945),
         between(Longitude_mid, -87.795, -87.615))

### Fitting the model
# As the number of criminal activities is a count variable, a Poisson model may be suitable.
# Different area's in the city have different criminality rates. Low rates are seen in the
# skirts of the city, higher rates at the coastal area in the center. We model the rate
# by splines, flexible curves that are able to capture local behaviour. We also add spatial
# correlation. This means that the parameter fits capture the fact that the criminality rate
# at a point is correlated to the rate nearly. The further away, the less correlated the 
# rates are.

model = gam(Count ~ year + s(Latitude_mid, Longitude_mid, bs = c("tp", "tp")) + s(day_of_year),
            family=poisson(link = 'log'),
            correlation = corExp(form = ~Latitude_mid+Longitude_mid, nugget=TRUE), 
            method = "REML",
            data=data)

data_pred = data %>% 
  mutate(pred = predict(model, type='response'),
         month = month(Date) + 12*(year -2010)) %>% 
  filter(between(Latitude_mid,41.825, 41.92),
         between(Longitude_mid,-87.675, -87.645)) %>% 
  group_by(month, Latitude_mid, Longitude_mid) %>% 
  summarise(Count = sum(Count),
            Count_pred = sum(pred))

data_pred %>% 
  ggplot(aes(x=month)) +
  geom_point(aes(y=Count), color='blue') +
  geom_point(aes(y=Count_pred), color='red') +
  facet_wrap(vars(Latitude_mid, Longitude_mid))


