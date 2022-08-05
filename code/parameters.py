import numpy as np
import pandas as pd
from decimal import Decimal, getcontext
import sys

"""
Paste time series data into the raw_data vector to calculate the Hurst exponent of a time series.
The Hurst exponent indicates the intensity of long-range dependence in the time series and overall captures volatility persistence within the stochastic process, which can be represented through variable H ⋲ (0, 1). 

The value of the Hurst exponent can characterize price movement in three scenarios:
- If H ⋲ (0, ½), each increment is positively correlated, and the movement is characterized as exhibiting short-term memory dependency
- If H = ½, each increment is considered independent from one another, and the movement is characterized as classic Brownian motion
- If H ⋲ (½, 1), each increment is negatively correlated, and the movement is characterized as exhibiting long-term memory dependency

There exist numerous methods to estimate the Hurst exponent, namely the rescaled range analysis method (simple, no underlying assumptions needed), the Higuchi method, the periodogram method, and the variance method. In this script, I used the rescaled range method to calculate the Hurst exponent, though this method has its drawbacks with its high sensitivity to short-range dependency.

Details on using the rescaled range method can be viewed at: https://en.wikipedia.org/wiki/Rescaled_range.
"""

#Raw time series data should be pasted here from least recent to most recent data - sample data here is taken from S&P 500's historical returns from July 2021 to July 2022.
raw_data = [4343.54,4358.13,4320.82,4369.55,4384.63,4369.21,4374.3,4360.03,4327.16,4258.63,4323.21,4358.69,4367.48,4411.8,4422.23,4401.46,4400.65,4419.15,4395.26,
            4387.11,4423.15,4402.68,4429.1,4436.52,4432.35,4436.75,4447.7,4460.84,4468,4479.66,4448.08,4400.27,4405.8,4441.67,4479.54,4486.23,4496.19,4469.91,4509.37,
            4528.79,4522.68,4524.09,4536.95,4535.43,4520.03,4514.07,4493.28,4458.58,4468.73,4443.05,4480.7,4473.76,4432.99,4357.73,4354.18,4395.64,4448.98,4455.48,
            4443.11,4352.63,4359.46,4307.54,4357.05,4300.46,4345.72,4363.55,4399.76,4391.36,4361.19,4350.64,4363.8,4438.23,4471.37,4486.48,4519.63,4536.19,4549.78,
            4544.9,4566.48,4574.79,4551.68,4596.42,4605.38,4613.67,4630.65,4660.57,4680.06,4697.53,4701.7,4685.25,4646.71,4649.27,4682.85,4682.81,4700.9,4688.67,
            4706.64,4697.96,4682.95,4690.7,4701.46,4594.62,4655.27,4567,4513.04,4577.1,4538.43,4591.67,4686.75,4701.21,4667.45,4712.02,4669.15,4634.09,4709.84,4668.67,
            4620.64,4568.02,4649.23,4696.56,4725.78,4791.19,4786.36,4793.06,4778.73,4766.18,4796.56,4793.54,4700.58,4696.05,4677.02,4670.29,4713.07,4726.35,4659.02,
            4662.85,4577.34,4532.76,4482.73,4397.93,4410.13,4356.45,4349.93,4326.5,4431.85,4515.55,4546.54,4589.32,4477.44,4500.54,4483.87,4521.54,4587.18,4504.06,
            4418.64,4401.67,4471.07,4475.01,4380.26,4348.87,4304.74,4225.5,4288.7,4384.62,4373.79,4306.26,4386.54,4363.49,4328.87,4201.09,4170.62,4277.88,4259.52,
            4204.31,4173.11,4262.45,4357.95,4411.67,4463.09,4461.18,4511.61,4456.23,4520.16,4543.04,4575.52,4631.6,4602.45,4530.41,4545.86,4582.64,4525.12,4481.15,
            4500.21,4488.28,4412.53,4397.45,4446.59,4392.59,4391.69,4462.21,4459.45,4393.66,4271.78,4296.12,4175.2,4183.96,4287.5,4131.93,4155.38,4175.48,4300.17,
            4152.38,4123.34,3991.24,4001.05,3935.18,3930.08,4023.89,4008.01,4088.85,3923.68,3900.79,3901.36,3973.75,3941.48,3978.73,4057.84,4158.24,4132.15,4101.23,
            4176.82,4108.54,4121.43,4160.68,4115.77,4017.82,3900.86,3749.63,3735.48,3789.99,3666.77,3674.84,3764.79,3759.89,3795.73,3911.74,3900.11,3821.55,3818.83,
            3785.38,3825.33,3831.39,3845.08,3902.62,3899.38,3854.43,3818.8,3801.78,3790.38,3863.16,3830.85,3936.69,3959.9,3998.95,3961.63,3966.84,3921.05,4023.61,
            4072.43,4130.29,4118.63,4091.19]

delta_t = 1 # delta t equals 1 since we are using daily returns, however we can adjust this value depending on the timeframe

getcontext().prec = 5 #Calculated values in the program should be rounded to 5 decimal places

returns_data = [] # Daily logarithmic (ln used here) returns data based on raw data
for l in range(1,len(raw_data)):
    returns_data.append(np.log(raw_data[l]/raw_data[l-1]))



#################################################### Calculating the Hurst Exponent ####################################################

class Hurst_chunks:
    def __init__(self, list):
        self.list = list
        self.chunked = []
        self.chunknum = 0

    def chunk_num(self): # Ideally, there should be at least 100 data points in the set, however this program aims to demonstrate the step-by-step process in calculating the Hurst exponent
        if (len(self.list) >= 10) and (len(self.list) < 16):
            self.chunknum = 2
        elif (len(self.list) >= 16) and (len(self.list) < 30):
            self.chunknum = 3
        elif (len(self.list) >= 30) and (len(self.list) < 60):
            self.chunknum = 4
        elif len(self.list) >= 60:
            self.chunknum = 5
        else:
            print("Price dataset is too small.")
            return False
        return True

    def chunks(self): # Divide the data into respective chunks
        for k in range(0, self.chunknum + 1):
            count, inc = divmod(len(self.list), 2**k)
            for i in range(2**k):
                self.chunked.append(self.list[(i*count) + min(i, inc):((i+1)*count) + (min(i+1, inc))])
        return self.chunked
 
init_hurst = Hurst_chunks(returns_data)
chunk_number = init_hurst.chunk_num()
if chunk_number == True:
    chunks = init_hurst.chunks()
else:
    sys.exit() #Stop program if the dataset is too small.
 
class Rescaled_range:
    def __init__(self, prices):
        self.prices = prices
        self.means = []
        self.stds = []
        self.mean_centered = []
        self.total_ranges = []
        self.rescaled_ranges = []
 
    def total_means(self): # Find the means of each respective chunk
        for i in range(len(self.prices)):
            self.means.append(Decimal(np.mean(self.prices[i])))
        return self.means
 
    def total_stds(self): # Find the standard deviation values of each respective chunk
        for i in range(len(self.prices)):
            self.stds.append(Decimal(np.std(self.prices[i], ddof=1))) # stdev of sample, not entire population
        return self.stds
 
    def mean_centered_sum(self, mean, ind): # Find the mean centered sum in each chunk based on each chunk's calculated mean
        cumsum = 0
        for i in range(len(self.prices[ind])):
            cumsum += Decimal(self.prices[ind][i]) - mean
            self.mean_centered.append(cumsum)
        r = (max(self.mean_centered) - min(self.mean_centered))
        self.mean_centered.clear()
        return r
 
    def ranges(self): # find the range of each chunk
        for j in range(len(self.means)):
            self.total_ranges.append(self.mean_centered_sum(rr_means[j], j))
        return self.total_ranges
 
    def rescaled(self): # Rescaled range values of each chunk
        for i in range(len(self.total_ranges)):
            self.rescaled_ranges.append(self.total_ranges[i] / self.stds[i])
        return self.rescaled_ranges
 
rr_init = Rescaled_range(chunks)
rr_means = rr_init.total_means()
rr_stds = rr_init.total_stds()
rr_ranges = rr_init.ranges()
rr_rescaled = rr_init.rescaled()
 
class Hurst:
    def __init__(self, rescaled_values, chunk_ranges, total_len):
        self.rescaled_values = rescaled_values
        self.lengths = [1]
        self.chunk_ranges = chunk_ranges
        self.rescaled_averages = [rescaled_values[0]]
        self.average_chunk_lengths = []
        self.total_len = total_len
        self.log_x = []
        self.log_y = []

    def rescaled_ranges_averages(self): # Find the average rescaled range value for each region
        for i in range(1, self.chunk_ranges + 1):
            self.lengths.append(2**i)
            self.rescaled_averages.append(sum(rr_rescaled[((2**i)-1):((2**(i+1))-1)])/(((2**(i+1))-1)-((2**i)-1)))
        return self.rescaled_averages
    
    def avg_num_points(self): # Find the average number of datapoints in each region
        for i in range(len(self.lengths)):
            if ((self.total_len / self.lengths[i]) % 1) <= 0.25:
                self.average_chunk_lengths.append(int(np.floor(self.total_len/self.lengths[i])))
            else:
                self.average_chunk_lengths.append(int(np.ceil(self.total_len/self.lengths[i])))
        return self.average_chunk_lengths
    
    def dataframe(self, datapoints, rs):
        data = {"Number of ranges per region": self.lengths, "Average number of data points per region": datapoints,
            "Rescaled Ranges": rs
        }
        df = pd.DataFrame(data)
        df.set_index("Number of ranges per region", inplace=True)
        return df
    
    def log_x_values(self):
        for i in range(len(self.average_chunk_lengths)):
            self.log_x.append(Decimal(self.average_chunk_lengths[i]).ln()) #Use ln for logarithm
        return self.log_x

    def log_y_values(self):
        for i in range(len(self.rescaled_averages)):
            self.log_y.append(Decimal(self.rescaled_averages[i]).ln())
        return self.log_y
    
    def log_table(self):
        log_data = {"Number of ranges per region": self.lengths, "Logarithm of data range sizes (log x)": self.log_x, "Logarithm of R/S (log y)": self.log_y}
        df_log = pd.DataFrame(log_data)
        df_log.set_index("Number of ranges per region", inplace=True)
        return df_log
    
    def hurst_exponent(self): #Hurst exponent is calculated by taking the slope of the logy values against the logx values
        x_average = Decimal(np.mean(self.log_x))
        y_average = Decimal(np.mean(self.log_y))
        denominator = 0
        numerator = []

        for i in range(len(self.log_x)):
            numerator.append((self.log_x[i] - x_average)*(self.log_y[i] - y_average))
            denominator += (self.log_x[i] - x_average) ** 2
        return Decimal(sum(numerator)/denominator)

table = Hurst(rr_rescaled, init_hurst.chunknum, len(chunks[0]))
rs_values = table.rescaled_ranges_averages()
datapoints_per_region = table.avg_num_points()

print("The rescaled range values for each range is summarized in the table below:")
print(table.dataframe(datapoints_per_region, rs_values))

table.log_x_values()
table.log_y_values()
print("The logarithm values can be viewed in the table below:")
print(table.log_table())

hurst = table.hurst_exponent()
print(f"The Hurst exponent of the given data series is {hurst}.")


"""
After we have calculated the Hurst exponent, we can estimate the other two required parameters: volatility and drift.
"""

#################################################### Calculating Volatility and Drift ####################################################

volatility = (np.var(returns_data, ddof=1)/((np.abs(delta_t)**(float(hurst)*2))**0.5))
drift = (np.average(returns_data)/delta_t) + ((volatility**2)/2)
