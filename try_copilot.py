import numpy as np 
import matplotlib.pyplot as plt

x = np.ones(7)
# make another variable y with the same numbers of elements as x but with random numbers
y = np.random.rand(7)



# plot x and y
plt.plot(x,y)
# label the axis
plt.xlabel('x')
plt.ylabel('y')
# add a title 
plt.title('My first plot')
# matplotlib slider example
# q: can i use latex with copilot?
# a: yes, but you need to use the r prefix to the string
# q: mass of the electron
# a: 9.10938356e-31 kg

# show the plot
plt.show()



