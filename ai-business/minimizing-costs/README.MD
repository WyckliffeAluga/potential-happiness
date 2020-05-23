Build an AI to minimize energy consumption of a single server, and thanks to the object oriented structure of the implementation, the AI model will be applicable to any server of a same data center, thus minimizing the costs of the whole data center.



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

Assumption 2: 

The energy spent by a system ( AI or the server’s integrated cooling system)
that changes the server’s temperature from Tt to Tt+1 within 1 unit of time (here 1 minute),
can be approximated ag

