# ---------------------------------------------- #
# Imports
import matplotlib.pyplot as plt
import matplotlib.colors as col
from IPython import display
from demo_game import blue,green,orange,yellow
# ---------------------------------------------- #
# Colours
purple = (138,0,255)
# ---------------------------------------------- #

# Plotting Script
# Begin by making the figure interactive so we can update it live
plt.ion()

# Plotting Function
def plot(scores, mean_scores,ax1,ax2,fig,right_wall_total,left_wall_total,floor_total,ceiling_total,snake_collison_total):
    # This function is used to keep track of our snake's analytics 

    # Begin by collecting the current figure so no extra figures are produced
    display.clear_output(wait=True)
    display.display(plt.gcf())

    # Axis 1: Score Line Chart
    # Clear the axis 
    ax1.clear()
    # Plotting the current score
    ax1.plot(scores,color='black',label = 'Current Score')
    # Plotting the mean score
    ax1.plot(mean_scores,color='red',label='Mean Score')
    # Applying a legend to the plot
    ax1.legend(loc='upper left')
    # Setting the x label of the plot
    ax1.set_xlabel('Number Of Games')
    # Ensuring the limit of the y axis remains at 0
    ax1.set_ylim(0)
    # Adding data labels to the plot (Current Score)
    ax1.text(len(scores)-1, scores[-1], str(scores[-1]))
    # Adding data labels to the plot (Mean Score)
    ax1.text(len(mean_scores)-1, mean_scores[-1], str(  round(mean_scores[-1],2)))
    # Setting aa title for the plot
    ax1.set_title('Game Scores & Mean Score')
    # Setting the facecolor of the plot
    ax1.set_facecolor('#f6f6f6')

    # Axis 2: Collision Horizontal Bar Chart
    # Clear the axis 
    ax2.clear()
    # Setting the facecolor of the plot
    ax2.set_facecolor('#f6f6f6')
    # Setting aa title for the plot
    ax2.set_title('Collision Count')
    # Setting the x labels of the plot
    x = ['Right Wall','Left Wall','Floor','Ceiling','Body Collision']
    # Adding the data points of the plot
    y = [right_wall_total,left_wall_total,floor_total,ceiling_total,snake_collison_total]
    # Producing the horizontal bar chart
    # We use a list comprehension to convert the RGB colour codes to values between 
    hbars = ax2.barh(x,y,color=[tuple(ti/255 for ti in blue),
                tuple(ti/255 for ti in orange),tuple(ti/255 for ti in yellow),tuple(ti/255 for ti in green),tuple(ti/255 for ti in purple)])
    # Adding data labels to the plot
    ax2.bar_label(hbars,y)
    # Calling to show the figure
    # block = False prevents any blocking between the two subplots
    plt.show(block=False)
    # Set a pause to allow the update to occur more smoothly
    plt.pause(.1)

