import sys
import random
import numpy as np

from typing import Dict, Any

# dictionary to hold objects information
dishes_data = {}
salary_data = {}

# simulation output matric
# - colums : hours
# - rows : simulation output
simulation_output = np.zeros((100, 24))


def dish_data_init(dish_name, base_price, time_to_cook, time_to_serve, menu_price):
    # TBD - tweak model
    cost_data = [0 for x in range(24)]
    cost_data[11] = base_price
    cost_data[12] = base_price + 2   # overhead of $2 for every dish ??
    cost_data[13:17] = [base_price for x in range(17-13)]
    cost_data[17:22] = [base_price + 2 for x in range(22-17)]  # overhead of $2 for every dish ??
    cost_data[22:24] = [base_price for x in range(24-22)]
    dish_data = {'ingredients price': cost_data,
                'time to cook': time_to_cook,
                'time to serve': time_to_serve,
                'menu price' : menu_price}
    dishes_data[dish_name] = dish_data


def salary_data_init(position_name, hourly_rate_base, peak_time_overhead):
    # TBD - tweak model
    hourly_rate = [0 for x in range(24)]
    hourly_rate[11] = hourly_rate_base
    hourly_rate[12] = hourly_rate_base + peak_time_overhead
    hourly_rate[13:17] = [hourly_rate_base for x in range(17 - 13)]
    hourly_rate[17:22] = [hourly_rate_base + peak_time_overhead for x in range(22 - 17)]
    hourly_rate[22:24] = [hourly_rate_base for x in range(24 - 22)]
    salary_data[position_name] = hourly_rate


def data_init():
    # TBD - tweak prices, base time, hoursly rates, menu price
    dish_data_init("chicken", 3, 1/6, 5, 30)
    dish_data_init("beef", 5, 1/4, 5, 40)
    dish_data_init("chow mein", 2, 1/6, 5, 15)
    dish_data_init("spring rools", 2, 1 / 4, 5, 10)
    salary_data_init("cook", 20, 5)
    salary_data_init("waiter", 15, 6)
    salary_data_init("manager", 30, 10)

def data_print(data_dict):
    for key,value  in data_dict.items():
        print("key = {0} - \t value = {1}".format(key, value))
    print("\n")

def monte_carlo_simulation():
    for i in range(100):
        restaurant_model(i)

def restaurant_model(iteration):
    for time in range(11,24):
        simulation_output[iteration][time]= customer_walkins(time)
    print(simulation_output)

def customer_walkins(time):
    # TBD - adjust random pick in function of hours
    random_walkins = random.randrange(1, 50)
    print("{0}:00h - {1} customers walk in".format(time, random_walkins))
    total_hourly_profit = 0
    total_hourly_cost = 0
    for i in range(1,random_walkins+1):
        print("{0}:00h - Customer {1} started ordering".format(time, i))
        order_cost, order_profit = customer_order(time)
        total_hourly_cost += order_cost
        total_hourly_profit += order_profit
        print("{0}:00h - Customer {1} completed ordering".format(time, i))
        print("{0}:00h - Total cost: ${1:4.2f} - Total profit: ${2:4.2f}".format(time, total_hourly_cost,
                                                                               total_hourly_profit))
    print("{0}:00h - Total profit: {1:4.2%}".format(time, total_hourly_profit/total_hourly_cost))
    return (total_hourly_profit/total_hourly_cost)

def customer_order(time):
    # order dishes randomly on menu
    num_items = random.randrange(1, len(dishes_data))
    print("{0}:00h - \t Customer picked {1} items".format(time, num_items))
    cost_sum = 0
    profit_sum = 0
    for i in range(num_items):
        choice = random.choice(list(dishes_data))
        print("{0}:00h - \t\trandom pick is: {1}".format(time, choice))
        cost = dish_cost(choice, time)
        cost_sum += cost
        profit_sum += dish_profit(choice, cost, time)
    print("{0}:00h - \t Customer cost: ${1:4.2f} - profit ${2:4.2f}".format(time,  cost_sum, profit_sum))
    return cost_sum, profit_sum

def dish_profit(dish_name, cost, time):
    print("{0}:00h - \t\t\tCalculate dish {1} profit ".format(time,dish_name))
    dish_menu_price = dishes_data[dish_name]["menu price"]
    print("{0}:00h - \t\t\t dish menu price: ${1} - dish cost: ${2:4.2f}".format(time, dish_menu_price, cost))
    profit = (dish_menu_price - cost)
    print("{0}:00h - \t\t\t profit: ${1:4.2f}".format(time, profit))
    return profit


def random_dish_cost_base(dish_name):
    cost = 0
    if dish_name == "chiken":
        cost = random.random()*2
    elif dish_name == "beef":
        cost = random.random()*4
    else:
        cost = random.random()
    print("\t\t\t\t\t\trandom {0} base cost: ${1:4.2f}".format(dish_name, cost))
    return cost

def dish_cost(dish_name, time):
    print("{0}:00h - \t\t\t\tCalculating dish {1} cost".format(time, dish_name))
    base_cost = dishes_data[dish_name]["ingredients price"][time]
    labor_cost = dishes_data[dish_name]["time to cook"]*(salary_data["cook"][time] + salary_data["waiter"][time] + salary_data["manager"][time])
    print("{0}:00h - \t\t\t\tbase cost: ${1:4.2f} - labor cost: ${2:4.2f}".format(time, base_cost, labor_cost))
    cost_output = random_dish_cost_base(dish_name) + base_cost + labor_cost
    print("{0}:00h - \t\t\t\trandom cost output: ${1:4.2f}".format(time, cost_output))
    return cost_output

def monte_carlo_sample_hourly_analysis(sample, time):
    # Monte Carlo Analysis
    sample_size = sample.size
    # TBD - store the results in a Panda or another nparray for further analysis across all times
    print("\t\t\ttime : ", time)
    print("\t\t\tmax profit {0:2.2%}".format(sample.max()))
    print("\t\t\tmin profit {0:2.2%}".format(sample.min()))
    print("\t\t\tprofit > 100% - {0:2.2f}".format((sample > 1).sum()/sample.size))
    print("\t\t\t90% < profit < 100% - {0:2.2f}".format(np.logical_and(sample >= 0.9, sample <1).sum()/sample_size))
    print("\t\t\t80% < profit < 90% - {0:2.2f}".format(np.logical_and(sample >= 0.8, sample < 0.9).sum()/sample_size))
    print("\t\t\t70% < profit < 80% - {0:2.2f}".format(np.logical_and(sample >= 0.7, sample < 0.8).sum()/sample_size))
    print("\t\t\t60% < profit < 70% - {0:2.2f}".format(np.logical_and(sample >= 0.6, sample < 0.7).sum()/sample_size))
    print("\t\t\t50% < profit < 60% - {0:2.2f}".format(np.logical_and(sample >= 0.5, sample < 0.6).sum()/sample_size))
    print("\t\t\t40% < profit < 50% - {0:2.2f}".format(np.logical_and(sample >= 0.4, sample < 0.5).sum()/sample_size))
    print("\t\t\t30% < profit < 40% - {0:2.2f}".format(np.logical_and(sample >= 0.3, sample < 0.4).sum()/sample_size))
    print("\t\t\t20% < profit < 30% - {0:2.2f}".format(np.logical_and(sample >= 0.2, sample < 0.3).sum()/sample_size))
    print("\t\t\t10% < profit < 20% - {0:2.2f}".format(np.logical_and(sample >= 0.1, sample < 0.2).sum()/sample_size))
    print("\t\t\t0% < profit < 10% - {0:2.2f}".format(np.logical_and(sample >= 0, sample < 0.1).sum() / sample_size))
    print("\t\t\t-10% < loss < 0% - {0:2.2f}".format(np.logical_and(sample >= -0.1, sample < 0).sum() / sample_size))
    print("\t\t\t-20% < loss < 10% - {0:2.2f}".format(np.logical_and(sample >= -0.2, sample < -0.1).sum() / sample_size))
    print(
        "\t\t\t-30% < loss < 20% - {0:2.2f}".format(np.logical_and(sample >= -0.3, sample < -0.2).sum() / sample_size))

def monte_carlo_analysis():
    for hour in range(11,23):
        monte_carlo_sample_hourly_analysis(simulation_output[:, hour], hour)

if __name__ == '__main__':
    data_init();
    #data_print(dishes_data);
    #data_print(salary_data);
    #dish_cost("chicken", 11);
    #dish_profit("chicken", 12, 11)
    #customer_order(11)
    #customer_walkins(11)
    #restaurant_model(0)
    monte_carlo_simulation()
    #monte_carlo_sample_hourly_analysis(simulation_output[:,11], 11)
    monte_carlo_analysis()
    #sample_func()
