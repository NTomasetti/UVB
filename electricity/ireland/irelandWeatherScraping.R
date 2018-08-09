library(rvest)
library(lubridate)

dateSeq <- seq(ymd('2009-07-14'), ymd('2010-12-31'), by = 1)
weatherFull <- tibble()

for(i in 1:536){
  print(i)
  url <- paste0('https://www.weatherbase.com/weather/weatherhourly.php3?s=96930&cityname=Dublin-Ireland&date=', dateSeq[i], '&units=metric')
  webpage <- read_html(url)
  
  data_html <- html_nodes(webpage,'td')
  weather_data <- html_text(data_html)
  headings_html <- html_nodes(webpage,'th')
  headings <- html_text(headings_html)
  
  first <- min(which(weather_data == '12:00 AM'))
  
  weather <- matrix(weather_data[first:(length(weather_data)-19)], ncol = 12, byrow = TRUE)
  colnames(weather) <- headings[9:20]
  weather <- as.data.frame(weather)
  weather$date <- dateSeq[i]
  
  weather %>%
    filter(!duplicated(`Local Time`)) %>%
    mutate(Temperature = as.numeric(substr(as.character(Temperature), 1, nchar(as.character(Temperature))-3)),
           Dewpoint = as.numeric(substr(as.character(Dewpoint), 1, nchar(as.character(Dewpoint))-3)),
           Humidity = as.numeric(substr(as.character(Humidity), 1, nchar(as.character(Humidity))-2)),
           Barometer = as.numeric(substr(as.character(Barometer), 1, nchar(as.character(Barometer))-4)),
           Visibility = as.numeric(substr(as.character(Visibility), 1, nchar(as.character(Visibility))-3)),
           `Wind Speed` = as.numeric(substr(as.character(`Wind Speed`), 1, nchar(as.character(`Wind Speed`))-5))) -> weather
  
  weatherFull <- rbind(weatherFull, weather)
  
}

sanity <- rep(0, length(dateSeq))
for(i in seq_along(dateSeq)){
  weatherFull %>%
    filter(date == dateSeq[i]) %>%
    .$Barometer %>%
    min() -> sanity[i]
}
qplot(sanity)

weatherFull$`Wind Direction` <- NULL
weatherFull$`Gust Speed` <- NULL
weatherFull$Precipitation <- NULL
weatherFull$Events <- NULL
weatherFull$Barometer <- NULL
weatherFull$Visibility <- NULL
weatherFull$`Wind Speed` <- NULL
weatherFull$Conditions <- NULL


times <- unique(weatherFull$`Local Time`)

weatherFull %>%
  filter(`Local Time` %in% times[1:48]) -> weatherFull

weatherFull <- mutate(weatherFull,
                      time = as.numeric(factor(`Local Time`, levels = as.character(times[1:48]), labels = 1:48)))
weatherFull$`Local Time` <- NULL

# data from ireland.R, select one ID worth of dates and time and merge with weather to see if there are any date/time combos missing from weatherFull

data %>%
  filter(id == min(data$id)) %>%
  select(date, time) %>%
  left_join(weatherFull) -> dataSub

# there are some NA values, either as an NA in weatherFull or a date time combo that was missing in weatherFull but appeared in data
# C++ interpolation function

library(Rcpp)
library(RcppArmadillo)

cppFunction(depends = "RcppArmadillo",
            'arma::mat interpolate(arma::mat data) {
        
        int n = data.n_rows;
        arma::mat results(n, 3);
        results.fill(-100);

        results.row(0) = data.row(0);
        for(int i = 1; i < n-1; ++i){
          if(data(i, 0) == -100){
            bool na = true;
            int j = 1;
            while(na){
              if(data(i + j, 0) == -100){
                j += 1;
              } else {
                results.row(i) = j * 1.0 / (j + 1.0) * results.row(i-1) + 1.0 / (j + 1.0) * data.row(i + j);
                na = false;
              }
            }
          } else {
            results.row(i) = data.row(i);
          }
        }
        results.row(n-1) = data.row(n-1);
        return results;
    }')

tempMat <- dataSub %>% 
  select(Temperature, Dewpoint, Humidity) %>%
  as.matrix()
tempMat[is.na(tempMat)] <- -100

inter <- interpolate(tempMat)
dataSub$Temperature <- inter[,1]
dataSub$Dewpoint <- inter[,2]
dataSub$Humidity <- inter[,3]

write.csv(dataSub, 'electricity/ireland/weather.csv', row.names = FALSE)

