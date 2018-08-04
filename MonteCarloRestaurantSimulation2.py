import sys
import random
import numpy as np
import pandas as pd
from typing import Dict, Any

# dictionaries that hold model objects (dishes and salary) information
dishes_data = {}
salary_data = {}

# simulation output matrix
# - colums - hour_slots : hours (restaurant can be open from 24h a day)
# - rows - simulation_iterations : a given simulation iteration output
hour_slots = 24
simulation_iterations = 1000
simulation_output = np.zeros((simulation_iterations, hour_slots))
max_people_capacity = 50

# simulation analysis is captured in a Panda Frame
# For each hour of the day, it holds the profit or loss percentage
monte_carlo_results = 0
hourly_analysis_frame = pd.DataFrame(monte_carlo_results,
                                     columns=['max profit', 'min profit', '+100% profit', '100-90% profit',
                                              '90-80% profit', '80-70% profit', '70-60% profit', '60-50% profit',
                                              '50-40% profit', '40-30% profit', '30-20% profit', '20-10% profit',
                                              '10-0% profit',
                                              '0-10% loss', '10-20% loss', '20-30% loss', '30-40% loss',
                                              '40-50% loss', '50-60% loss', '60-70% loss', '70-80% loss', '80-90% loss',
                                              '90-100% loss', '+100% loss'],
                                     index = np.arange(23), dtype=float)


def data_init():
    """
    This function initializes the model objects parameters. For a given simulation, these parameters will yield a
    certain profitability/loss. If the profitability/loss is not desirable, then, the parameters can be
    tweaked before running a new simulation
    :return:
    """
    # base cost, base time, hoursly rates, menu price
    dish_data_init("duck confit", 8, 1 / 4, 5, 25)
    dish_data_init("boeuf bourgignon", 6, 1 / 4, 5, 30)
    dish_data_init("gratin dauphinois", 3, 1 / 6, 5, 15)
    dish_data_init("nicoise salad", 4, 1 / 6, 5, 20)
    salary_data_init("cook", 15, 2)
    salary_data_init("waiter", 12, 4)
    salary_data_init("manager", 20, 1)

def dish_data_init(dish_name, base_cost, time_to_cook, time_to_serve, menu_price):
    """
    This function initializes the dish parameters. These parameters can be adjusted across simulations
    to measure the impact on the overall restaurant profitability
    :param dish_name: dish name
    :param base_price: dish base price
    :param time_to_cook: hourly pro rated time to cook the dish
    :param time_to_serve: hourly pro rated time to serve the dish
    :param menu_price: price of the dish on the menu
    :return:
    >>> dish_data_init("duck confit", 8, 1 / 4, 5, 25)
    duck confit dish cost: {'ingredients cost': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], 'time to cook': 0.25, 'time to serve': 5, 'menu price': 25}
    """
    # Base cost could be adjusted per hour
    cost_data = [0 for x in range(24)]
    cost_data[11:24] = [base_cost for x in range(24-11)]
    dish_data = {'ingredients cost': cost_data,
                'time to cook': time_to_cook,
                'time to serve': time_to_serve,
                'menu price' : menu_price}
    dishes_data[dish_name] = dish_data
    print("{0} dish cost: {1}".format(dish_name, dishes_data[dish_name]))


def salary_data_init(position_name, hourly_rate_base, peak_hours_factor):
    """
    This function initializes an employe salary parameters.
    The employee salary is a parameter in our model that can be adjusted across simulations to see how it impacts the
    profitability of the restaurant
    :param position_name: job name
    :param hourly_rate_base: hourly salary rate
    :param peak_hours_factor: number of employees needed at peak hour
    >>> pos_name "cook"
    >>>
    """
    hourly_rate = [0 for x in range(24)]
    hourly_rate[11] = hourly_rate_base
    hourly_rate[12:15] = [(peak_hours_factor * hourly_rate_base) for x in range(15 - 12)]
    hourly_rate[15:17] = [hourly_rate_base for x in range(17 - 15)]
    hourly_rate[17:23] = [(peak_hours_factor * hourly_rate_base) for x in range(23 - 17)]
    hourly_rate[23] = hourly_rate_base
    salary_data[position_name] = hourly_rate
    print("{0} salary data: {1}".format(position_name, salary_data[position_name]))


def data_print(data_dict):
    for key,value  in data_dict.items():
        print("key = {0} - \t value = {1}".format(key, value))
    print("\n")

def monte_carlo_simulation(iterations):
    """
    This function is the engine of the monte carlo simulation for the restaurant model.
    It runs the restaurant_model() simulation for a certain number of iterations. The iterations
    parameter allows to experiment with the Monte Carlo and see, for a guven set of parameters,
    how the number of simulations affects the output.
    :return: The results are stored in the simulation_output nparray.
    """
    for i in range(iterations):
        restaurant_model(i)

def restaurant_model(iteration):
    """
    This function implements the restaurant model. The model has the following random variables
    - number of people walk-in
    - number of dishes chosen by a given customer
    - variation of a dish cost (eg. based on seasons, food market, etc.)
    For every iteration of a given simulation, the random variables are going to change
    :param iteration:
    """
    for time in range(11, hour_slots):
        # We assume the restaurant opens at 11am
        simulation_output[iteration][time]= customer_walkins(time)
    #print(simulation_output)

def customer_walkins(time):
    """
    This function randomizes the number of customer that walk into the restaurant
    :param time: hour of day
    :return: total profit percentage for a given hour.
    """
    # the random number of customers depends on the time
    if ((time < 12) or ((time > 14) and (time <17)) or (time > 22)):
        random_walkins = random.randrange(5, max_people_capacity)
    elif (((time >=12) and (time < 14)) or ((time >=17) and (time <= 22))):
        random_walkins = random.randrange(max_people_capacity - 10, max_people_capacity)
    else:
        random_walkins = random.randrange(1, max_people_capacity)

    #print("{0}:00h - {1} customers walk in".format(time, random_walkins))
    total_hourly_profit = 0
    total_hourly_cost = 0
    for i in range(1,random_walkins+1):
        #print("{0}:00h - Customer {1} started ordering".format(time, i))
        order_cost, order_profit = customer_order(time)
        total_hourly_cost += order_cost
        total_hourly_profit += order_profit
        #print("{0}:00h - Customer {1} completed ordering".format(time, i))
        #print("{0}:00h - Total cost: ${1:4.2f} - Total profit: ${2:4.2f}".format(time, total_hourly_cost, total_hourly_profit))
    #print("{0}:00h - Total profit: {1:4.2%}".format(time, total_hourly_profit/total_hourly_cost))
    return (total_hourly_profit/total_hourly_cost)

def customer_order(time):
    """
    This function randomizes the number of dishes ordered by a customer
    :param time: The number of dishes ordered at a given hour
    :return:cost sum and profit sum
    """

    # The random number of items chosen depends on the time. During peak hours, customer order more dishes
    if ((time < 12) or ((time > 14) and (time <17)) or (time > 22)):
        num_items = random.randrange(1, 2)
    elif (((time >=12) and (time < 14)) or ((time >=17) and (time <= 22))):
        num_items = random.randrange(1, 4)
    else:
        num_items = 1

    #print("{0}:00h - \t Customer picked {1} items".format(time, num_items))
    cost_sum = 0
    profit_sum = 0
    for i in range(num_items):
        choice = random.choice(list(dishes_data))
        #print("{0}:00h - \t\trandom pick is: {1}".format(time, choice))
        cost = dish_cost(choice, time)
        cost_sum += cost
        profit_sum += dish_profit(choice, cost, time)
    #print("{0}:00h - \t Customer cost: ${1:4.2f} - profit ${2:4.2f}".format(time,  cost_sum, profit_sum))
    return cost_sum, profit_sum

def dish_profit(dish_name, cost, time):
    """
    This function calculates a dish profit
    :param dish_name: random selection of a dish
    :param cost: the base cost of a dish is predetermined
    :param time: a given hour
    :return: profit or loss of a dish
    >>> cost = 20.44
    >>> dish = "duck confit"
    >>> time = 11
    >>> dish_profit(dish, cost, time)
    $4.56
    """
    print("{0}:00h - \t\t\tCalculate dish {1} profit ".format(time,dish_name))
    dish_menu_price = dishes_data[dish_name]["menu price"]
    print("{0}:00h - \t\t\t dish menu price: ${1} - dish cost: ${2:4.2f}".format(time, dish_menu_price, cost))
    profit = (dish_menu_price - cost)
    print("{0}:00h - \t\t\t profit: ${1:4.2f}".format(time, profit))
    return profit


def random_dish_cost_base(dish_name):
    """
    This function returns the random factor to apply to dish base cost.
    :param dish_name:
    :return: dish cost base random factor
    """
    cost = 0
    if dish_name == "chiken":
        cost = random.random()
    elif dish_name == "beef":
        cost = random.random()
    else:
        cost = random.random()
    print("\t\t\t\t\t\trandom {0} base cost factor: ${1:4.2f}".format(dish_name, cost))
    return cost

def dish_cost(dish_name, time):
    """
    This function calculates a dish total cost, adding base cost, labor cost and the randomized dish cost factor
    :param dish_name: dish name
    :param time: time of the day
    :return: dish cost
    """
    print("{0}:00h - \t\t\t\tCalculating dish {1} cost".format(time, dish_name))
    base_cost = dishes_data[dish_name]["ingredients cost"][time]
    labor_cost = dishes_data[dish_name]["time to cook"]*(salary_data["cook"][time] + salary_data["waiter"][time] + salary_data["manager"][time])
    print("{0}:00h - \t\t\t\tbase cost: ${1:4.2f} - labor cost: ${2:4.2f}".format(time, base_cost, labor_cost))
    randomized_base_cost = (random_dish_cost_base(dish_name) + 1) * base_cost
    print("{0}:00h - \t\t\t\t{1} dish randomized cost: {2:4.2f}".format(time, dish_name, randomized_base_cost))
    cost_output = randomized_base_cost + labor_cost
    print("{0}:00h - \t\t\t\trandom cost output: ${1:4.2f}".format(time, cost_output))
    return cost_output

def monte_carlo_sample_hourly_analysis(sample, time):
    """
    This function caclulates the profit distribution for a given hour in 10% range increments.
    The results are stored in the global Pand frame hourly_analysis_frame.
    :param sample: sample np array for all simulations
    :param time: hour of the day in the model

    """
    # Monte Carlo Analysis
    sample_size = sample.size

    # print("\t\t\ttime : ", time)
    # print("\t\t\tmax profit {0:2.2%}".format(sample.max()))
    # print("\t\t\tmin profit {0:2.2%}".format(sample.min()))
    # print("\t\t\tprofit > 100% - {0:2.2f}".format((sample > 1).sum()/sample.size))
    # print("\t\t\t90% < profit < 100% - {0:2.2f}".format(np.logical_and(sample >= 0.9, sample <1).sum()/sample_size))
    # print("\t\t\t80% < profit < 90% - {0:2.2f}".format(np.logical_and(sample >= 0.8, sample < 0.9).sum()/sample_size))
    # print("\t\t\t70% < profit < 80% - {0:2.2f}".format(np.logical_and(sample >= 0.7, sample < 0.8).sum()/sample_size))
    # print("\t\t\t60% < profit < 70% - {0:2.2f}".format(np.logical_and(sample >= 0.6, sample < 0.7).sum()/sample_size))
    # print("\t\t\t50% < profit < 60% - {0:2.2f}".format(np.logical_and(sample >= 0.5, sample < 0.6).sum()/sample_size))
    # print("\t\t\t40% < profit < 50% - {0:2.2f}".format(np.logical_and(sample >= 0.4, sample < 0.5).sum()/sample_size))
    # print("\t\t\t30% < profit < 40% - {0:2.2f}".format(np.logical_and(sample >= 0.3, sample < 0.4).sum()/sample_size))
    # print("\t\t\t20% < profit < 30% - {0:2.2f}".format(np.logical_and(sample >= 0.2, sample < 0.3).sum()/sample_size))
    # print("\t\t\t10% < profit < 20% - {0:2.2f}".format(np.logical_and(sample >= 0.1, sample < 0.2).sum()/sample_size))
    # print("\t\t\t0% < profit < 10% - {0:2.2f}".format(np.logical_and(sample >= 0, sample < 0.1).sum() / sample_size))
    # print("\t\t\t-10% < loss < 0% - {0:2.2f}".format(np.logical_and(sample >= -0.1, sample < 0).sum() / sample_size))
    # print("\t\t\t-20% < loss < -10% - {0:2.2f}".format(np.logical_and(sample >= -0.2, sample < -0.1).sum() / sample_size))
    # print("\t\t\t-30% < loss < -20% - {0:2.2f}".format(np.logical_and(sample >= -0.3, sample < -0.2).sum() / sample_size))
    # print("\t\t\t-40% < loss < -30% - {0:2.2f}".format(np.logical_and(sample >= -0.4, sample < -0.3).sum() / sample_size))
    # print("\t\t\t-50% < loss < -40% - {0:2.2f}".format(np.logical_and(sample >= -0.5, sample < -0.4).sum() / sample_size))
    # print("\t\t\t-60% < loss < -50% - {0:2.2f}".format(np.logical_and(sample >= -0.6, sample < -0.5).sum() / sample_size))
    # print("\t\t\t-70% < loss < -60% - {0:2.2f}".format(np.logical_and(sample >= -0.7, sample < -0.6).sum() / sample_size))
    # print("\t\t\t-80% < loss < -70% - {0:2.2f}".format(np.logical_and(sample >= -0.8, sample < -0.7).sum() / sample_size))
    # print("\t\t\t-90% < loss < -80% - {0:2.2f}".format(np.logical_and(sample >= -0.9, sample < -0.8).sum() / sample_size))
    # print("\t\t\t-100% < loss < -90% - {0:2.2f}".format(np.logical_and(sample >= -1.0, sample < -0.9).sum() / sample_size))


                                             #index=['11am', '12pm', '1pm', '2pm', '3pm', '4pm', '5pm', '6pm', '7pm', '8pm', '9pm', '10pm'])
    hourly_analysis_frame['max profit'][time] = sample.max()
    hourly_analysis_frame['min profit'][time] = sample.min()
    hourly_analysis_frame['+100% profit'][time] = (sample > 1).sum()/sample.size
    hourly_analysis_frame['100-90% profit'][time] = np.logical_and(sample >= 0.9, sample <1).sum()/sample_size
    hourly_analysis_frame['90-80% profit'][time] = np.logical_and(sample >= 0.8, sample < 0.9).sum() / sample_size
    hourly_analysis_frame['80-70% profit'][time] = np.logical_and(sample >= 0.7, sample < 0.8).sum() / sample_size
    hourly_analysis_frame['70-60% profit'][time] = np.logical_and(sample >= 0.6, sample < 0.7).sum() / sample_size
    hourly_analysis_frame['60-50% profit'][time] = np.logical_and(sample >= 0.5, sample < 0.6).sum() / sample_size
    hourly_analysis_frame['50-40% profit'][time] = np.logical_and(sample >= 0.4, sample < 0.5).sum() / sample_size
    hourly_analysis_frame['40-30% profit'][time] = np.logical_and(sample >= 0.3, sample < 0.4).sum() / sample_size
    hourly_analysis_frame['30-20% profit'][time] = np.logical_and(sample >= 0.2, sample < 0.3).sum() / sample_size
    hourly_analysis_frame['20-10% profit'][time] = np.logical_and(sample >= 0.1, sample < 0.2).sum() / sample_size
    hourly_analysis_frame['10-0% profit'][time] = np.logical_and(sample >= 0.0, sample < 0.1).sum() / sample_size
    hourly_analysis_frame['0-10% loss'][time] = np.logical_and(sample >= -0.1, sample < -0.0).sum() / sample_size
    hourly_analysis_frame['10-20% loss'][time] = np.logical_and(sample >= -0.2, sample < -0.1).sum() / sample_size
    hourly_analysis_frame['20-30% loss'][time] = np.logical_and(sample >= -0.3, sample < -0.2).sum() / sample_size
    hourly_analysis_frame['30-40% loss'][time] = np.logical_and(sample >= -0.4, sample < -0.3).sum() / sample_size
    hourly_analysis_frame['40-50% loss'][time] = np.logical_and(sample >= -0.5, sample < -0.4).sum() / sample_size
    hourly_analysis_frame['50-60% loss'][time] = np.logical_and(sample >= -0.6, sample < -0.5).sum() / sample_size
    hourly_analysis_frame['60-70% loss'][time] = np.logical_and(sample >= -0.7, sample < -0.6).sum() / sample_size
    hourly_analysis_frame['70-80% loss'][time] = np.logical_and(sample >= -0.8, sample < -0.7).sum() / sample_size
    hourly_analysis_frame['80-90% loss'][time] = np.logical_and(sample >= -0.9, sample < -0.8).sum() / sample_size
    hourly_analysis_frame['90-100% loss'][time] = np.logical_and(sample >= -1.0, sample < -0.9).sum() / sample_size

    #print(hourly_analysis_frame['max profit'][time])


def monte_carlo_analysis():
    """
    This function performs the overall analysis of the Monte Carlo simulations.

    """
    #
    # For each hour slot, it performs the profitability analysis and stores the results in the
    # global Panda frame hourly_analysis_frame[]
    #
    for hour in range(11,hour_slots):
        monte_carlo_sample_hourly_analysis(simulation_output[:, hour], hour)

def debug_func():
    """
    This function is used to unit test helper functions

    """
    data_init();
    data_print(dishes_data);
    data_print(salary_data);
    for hour in [11,12,14,15,20,23]:
        dish = "duck confit"
        dish_profit(dish, dish_cost(dish, hour) , hour)
        #dish = "gratin dauphinois"
        #dish_profit(dish, dish_cost(dish, hour), hour)
        #dish = "boeuf bourgignon"
        #dish_profit(dish, dish_cost(dish, hour), hour)
        #dish = "nicoise salad"
        #dish_profit(dish, dish_cost(dish, hour), hour)

    # customer_order(11)
    # customer_walkins(11)
    # restaurant_model(0)

if __name__ == '__main__':
    #debug_func()
    # 1. Initialize the model for a given simulation
    # 2. Run the simulation for a pre-determined number of iterations
    # 3. Do the analysis of the results and capture the analysis results in a CSV file
    data_init()
    monte_carlo_simulation(simulation_iterations)
    monte_carlo_analysis()
    hourly_analysis_frame.to_csv('simulation.csv', sep=' ')
