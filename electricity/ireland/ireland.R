library(tidyverse)
library(lubridate)

data <- tibble()
set.seed(5)
for(i in 1:6){
  temp <- read_delim(paste0('electricity/ireland/File', i, '.txt'), delim = ' ', col_names = FALSE)
  
  # Only grab smart meters running over the desired period
  temp %>%
    group_by(X1) %>%
    summarise(start = min(X2),
              end = max(X2),
              n = n()) %>%
    filter(start == 19501 & end == 73048 & n == 25730) %>%
    .$X1 -> tempID
  
  # The data is too big for 8GB RAM, subsample 50 smart meters per group
  subset <- sample(tempID, 50)
  data <- rbind(data, temp[temp$X1 %in% subset, ])
}

rm(temp)
colnames(data) <- c('id', 'datetime', 'cons')

# set it so start + 1 is January 1, 2009, the first date in the datetime system
start <- ymd('2008-12-31')

data <- mutate(data,
               date = as.numeric(substr(datetime, 1, 3)),
               date = start + date,
               time = as.integer(substr(datetime, 4, 5)))
data$datetime <- NULL

# This then gets exported to irelandWeatherScraping.R

data %>%
  group_by(id, date) %>%
  summarise(daily = sum(cons)) %>% 
  ggplot() + geom_line(aes(date, daily, group = id)) + theme_bw()

data %>%
  group_by(id, date) %>%
  summarise(obs = n()) %>% 
  ggplot() + geom_line(aes(date, obs, group = id)) + theme_bw()


weather <- read.csv('electricity/ireland/weather.csv') %>%
  mutate(date = ymd(date))

data <- left_join(data, weather)

publicHols <- ymd(c('2009-01-01', '2009-03-17', '2009-04-10', '2009-04-12', '2009-04-13', '2009-05-04', '2009-06-01', '2009-08-03', '2009-10-26', '2009-12-24',
                    '2009-12-25', '2009-12-26', '2009-12-28', '2009-12-31', '2010-01-01', '2010-03-17', '2010-04-02', '2010-04-04', '2010-04-05', '2010-05-03',
                    '2010-06-07', '2010-08-02', '2010-10-25', '2010-12-24', '2010-12-25', '2010-12-26', '2010-12-27', '2010-12-31'))

data  <- mutate(data, holiday = date %in% publicHols,
                day = wday(date, label = TRUE))

saveRDS(data, 'electricity/ireland/data.RDS')
data <- readRDS('electricity/ireland/data.RDS')

# Look for patterns and such


data %>%
  filter(id == min(data$id)) %>%
  group_by(date) %>%
  summarise(max = max(Temperature),
            min = min(Temperature)) %>%
  gather(type, temp, -date) %>%
  ggplot() + geom_line(aes(date, temp, colour = type))

idVec <- unique(data$id)
idSample <- sample(idVec, 20)

dates <- unique(data$date)
weekStart <- sample(dates[1:(length(dates)-7)], 1)

data %>%
  filter(id %in% idSample &
          date >= weekStart & date < weekStart+ 7) %>%
  group_by(id) %>%
  mutate(time = seq_along(id)) %>%
  ggplot() + geom_line(aes(time, cons)) + 
  theme_bw() + 
  facet_wrap(~id, scales = 'free') + 
  scale_x_continuous(breaks = 3 * 48, labels = weekStart)
  
data %>% 
  filter(id == sample(idVec, 1)) %>%
  mutate(month = month(date),
         year = year(date),
         time = factor(paste(year, month),
                       levels = c(paste(2009, 7:12), paste(2010, 1:12)))) %>%
  group_by(time) %>%
  mutate(t = seq_along(id)) %>%
  ggplot() + geom_line(aes(t, log(cons + 0.01))) + facet_wrap(~ time, scales = 'free', ncol = 3) + 
  theme_bw()

acfs <- tibble()
maxLag <- 48 * 7 * 4
for(i in 1:length(idSample)){
  
  data %>% 
    filter(id == idSample[i]) %>%
    .$cons %>%
    acf(lag.max = maxLag, plot = FALSE)-> acfVec
  
  acfs <- rbind(acfs,
                tibble(id = idSample[i], 
                       acf = acfVec$acf[2:(maxLag+1)],
                       lag = 1:maxLag,
                       type = 'acf'))
  
  data %>% 
    filter(id == idSample[i]) %>%
    .$cons %>%
    pacf(lag.max = maxLag, plot = FALSE)-> pacfVec
  
  acfs <- rbind(acfs,
                tibble(id = idSample[i], 
                       acf = pacfVec$acf[2:maxLag],
                       lag = 2:maxLag,
                       type = 'pacf'))
  
}

signif <- qnorm(1 - 0.025 / maxLag) / sqrt(25730 - maxLag)

ggplot(acfs) + geom_line(aes(lag, acf)) + 
  geom_hline(aes(yintercept = signif), linetype = 'dashed') + 
  geom_hline(aes(yintercept = - signif), linetype = 'dashed') + 
  facet_wrap(id ~ type, ncol = 4, scales = 'free') + 
  theme(strip.background = element_blank(),
    strip.text.x = element_blank()) + 
  scale_x_continuous(labels = 1:14, breaks = seq(48, 48 * 14, 48))

sigPacf <- tibble()
for(i in 1:300){
  data %>% 
    filter(id == idVec[i]) %>%
    mutate(cons = log(cons + 0.01)) %>%
    .$cons %>%
    pacf(lag.max = maxLag, plot = FALSE) -> pacfVec

  sigPacf <- rbind(sigPacf,
                   tibble(id = idVec[i],
                          lags = 1:maxLag, 
                          pacf = pacfVec$acf[1:maxLag]))
  
}

sigPacf %>%
  filter(abs(pacf) > signif) %>%
  group_by(lags) %>%
  summarise(n = n() / 300) %>%
  ggplot() + geom_line(aes(lags, n)) +
  theme_bw() + 
  scale_x_continuous(labels = 1:14, breaks = seq(48, 48 * 14, 48))

sigPacf %>%
  filter(abs(pacf) > signif & lags <= 7 * 48) %>%
  mutate(hh = lags %% 48) %>%
  group_by(hh) %>%
  summarise(n = n() / (7 * 300)) %>%
  ggplot() + geom_line(aes(hh, n)) +
  theme_bw() 

sigPacf %>%
  filter(abs(pacf) > signif) %>%
  ggplot() + geom_point(aes(lags, pacf), size = 0.1) + 
  theme_bw() + 
  scale_x_continuous(labels = 1:14, breaks = seq(48, 48 * 14, 48))

sigPacf %>%
  filter(lags < max(lags)) %>%
  group_by(lags) %>%
  summarise(med = median(pacf),
            lower = quantile(pacf, 0.025),
            upper = quantile(pacf, 0.975)) %>%
  mutate(day = floor(lags / 48) + 1) %>%
  ggplot() + geom_point(aes(lags %% 48, med)) + 
  geom_errorbar(aes(lags %% 48, ymin = lower, ymax = upper)) + 
  facet_wrap(~day, scales = 'free', ncol = 7) +
  theme_bw()

# Create X/Y Matrix for modelling

data %>% 
  filter(id == min(data$id)) %>%
  select(Temperature, Humidity, holiday, day) %>%
  mutate(day = factor(day, ordered = FALSE)) -> x

X <- model.matrix(~ Temperature + Humidity + day + holiday, data = x)

data %>%
  select(id, cons, date, time) %>%
  spread(id, cons) %>%
  select(-date, -time) %>%
  as.matrix() -> Y

saveRDS(X, 'electricity/ireland/X.RDS')
saveRDS(Y, 'electricity/ireland/Y.RDS')



