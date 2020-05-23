Consider a  warehouse belonging  to an online retail company that sells products to a variety of customers. Inside this
warehouse, the products are stored in 12 different locations, labeled by the following letters from A to L:



As the orders are placed by the customers online, an Autonomous Warehouse Robot is moving around the
warehouse to collect the products for future deliveries.


The 12 locations are all connected to a computer system, which is ranking in real time the priorities of
product collection for these 12 locations. F


The mission is to build an AI that will always take the shortest route to the top
priority location, whatever the location it starts from, and having the option to go by an intermediary
location which is in the top 3 priorities.

Parameters 


• the average atmospheric temperature over a month

• the optimal range of temperatures of the server, which will be [18◦C, 24◦C]

• the minimum temperature of the server below which it fails to operate, which will be −20◦C

• the maximum temperature of the server above which it fails to operate, which will be 80◦C

• the minimum number of users in the server, which will be 10

• the maximum number of users in the server, which will be 100

• the maximum number of users in the server that can go up or down per minute, which will be 5

• the minimum rate of data transmission in the server, which will be 20

• the maximum rate of data transmission in the server, which will be 300

• the maximum rate of data transmission that can go up or down per minute, which will be 10



Variables:


• the temperature of the server at any minute

• the number of users in the server at any minute

• the rate of data transmission at any minute

• the energy spent by the AI onto the server (to cool it down or heat it up) at any minute

• the energy spent by the server’s integrated cooling system that automatically brings the server’s temperature back to the optimal range whenever the server’s temperature goes outside this optimal range


Assumption 1: 

The temperature of the server can be approximated through Multiple Linear
Regression, by a linear function of the atmospheric temperature, the number of users and the
rate of data transmission:

