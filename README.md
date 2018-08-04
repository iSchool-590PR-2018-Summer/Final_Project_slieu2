

TEMPLATE for your report to fill out:

# Title: Profit margins for 5 dishes served in a restaurant

## Team Member(s): Susan Lieu


# Monte Carlo Simulation Scenario & Purpose:
To measure the profit margin on four dishes throughout the day (11am - 10pm)

## Simulation's variables of uncertainty
net cost of dish = ingredients (ci) + time to prepare and cook dish (tc): Both the ingredients and time to prepare can vary. The ingredients of each dish can vary based on the season and prices set by wholesalers. The time to prepare a dish can vary depending on which cook is making the dish.
number of dishes ordered per hour = number of people who comes to the restaurant where the maximum number of patrons is 30 people. The number of people who come into the restaurant doesn’t really matter because they can order a drink and not a meal or conversely, one person can order two or more dishes.
labor costs or work rate (wr) =  dollar amount per hour per server/cook X number of hours worked. The labor cost fluctuates but it’s not uncertain. During lunch and dinner peak hours, there are more cooks and servers than off-peak hours. This fluctuation affects the overall profit margins so this variable needs to be factored in. 
The objects of the model and their parameters are the dish and the employee costs. 
The random variables are: the number of employees at a given time, the number of dishes ordered, the cost factor applied to the base cost of the ingredients of the dish. The random variable range are adjusted based on the time of the day. For instance, the number of customer walk-ins is between five and maximum number during non-peak hours. And it’s close to the maximum capacity during peak hours.
For time from 11am to 10pm, build statistical sample of total cost —> Tc(time): - pick work rate that corresponds to time - generate random variables for Xcust, Xpick, Xbase_cost,  - dish cost Tc = dish_base cost * (1 + Xbase_cost)



Example: 
Net cost of a dish is between $3-4 and it takes 12 minutes to make where the cook makes $15/hour. List price of dish is $12. If one person orders this dish per hour, then the profit margin is low since it barely covers the server’s hourly wage of $15/hr, not to mention the cook’s hourly wage.

The resulting analysis: for each hours, determine profit/loss distribution in increments of 10% ranges.

Min cost : $3 + 10min($15/h) = $5.5
Max cost: $4 + 15min($15/h) = $7.25

For each hour: - See distribution of profit: (menu price - Tc(time)) - Determine % of orders that have profit less than 10%-20%, 20%-30% and >30%

## Hypothesis or hypotheses before running the simulation:
The model is based on the total cost of dish such as ingredients, time to cook and serve the dish, and the menu price. The employee cost is based on the number of people working at a given time which affects the profit throughout the day.


## Analytical Summary of your findings: (e.g. Did you adjust the scenario based on previous simulation outcomes?  What are the management decisions one could make from your simulation's output, etc.)
Before running the simulation, I expected the peak hours to be the most profitable, but it turns out that’s not the case:
If you increase the number of customers, it doesn’t affect the profit distribution as much. 
If you increase/decrease the price of ingredients, it affects the profit a lot. 
The profit during peak lunch hour is quite low compared to peak dinner hour. During peak dinner hour, the profit margins are between 20 to 30 percent for about 65 percent of the time. At lunch hours 12-2pm, I’m losing money by 10 to 20 percent at 35 to 40 percent of the time and 20 to 30 percent of loss at 65 percent of time! And it looks like the dish that’s losing money is the duck confit.
Based on the outcomes, you can determine which variable has the most impact on profitability and then you can adjust it accordingly. The random variables are things that cannot be controlled, but it provides a rough estimates of some of the factors.

## Instructions on how to use the program: 
To see the program run with less iterations, the global variable simulation_iterations = 1000 can be changes to a smaller number.

To see the fluctuations profit, the menu price can be changed in the def data_init() function.
