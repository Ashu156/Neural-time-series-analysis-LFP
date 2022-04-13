# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:15:04 2022

@author: Dell
"""
from scipy.io import loadmat       # Import function to read data.
from pylab import *                # Import numerical and plotting functions
# from IPython.lib.display import YouTubeVideo  # Enable YouTube videos
rcParams['figure.figsize']=(12,3)  # Change the default figure size
import matplotlib.pyplot as plt

# Loading the data and basic visualization
data = loadmat('C:/Users/Dell/Downloads/Case-Studies-Python-binder/matfiles/02_EEG-1.mat')         # Load the data,
EEGa = data['EEGa']                             # ... and get the EEG from one condition,
t = data['t'][0]                                # ... and a time axis,
ntrials = len(EEGa)                             # ... and compute the number of trials.

mn = EEGa.mean(0)                               # Compute the mean signal across trials (the ERP).
sd = EEGa.std(0)                                # Compute the std of the signal across trials.
sdmn = sd / sqrt(ntrials)                       # Compute the std of the mean.

plot(t, mn, 'k', lw=3)                          # Plot the ERP of condition A,
plot(t, mn + 2 * sdmn, 'k:', lw=1)              # ... and include the upper CI,
plot(t, mn - 2 * sdmn, 'k:', lw=1)              # ... and the lower CI.
xlabel('Time [s]')                              # Label the axes,
ylabel('Voltage [$\mu$ V]')
title('ERP of condition A')                     # ... provide a useful title,
show()                                          # ... and show the plot.

# 

data.keys()
EEGb = data['EEGb']
ntrials, nsamples = EEGa.shape

dt = t[1] - t[0]  # Determine the sampling interval

plot(t, EEGa[0])  # Plot condition A, trial 1 data vs t.
plot([0.25, 0.25], [-4,4], 'k', lw=2)    # Add a vertical line to indicate the stimulus time               
xlabel('Time [s]')                   # Label the x-axis as time.
ylabel('Voltage [$\mu$ V]')          # Label the y-axis as voltage.
title('EEG data from condition A, Trial 1')  # Add a title

# savefig('imgs/2-2b')
show()

figure(figsize=(12, 3))     # Resize the figure to make it easier to see
plot(t,EEGa[0])                 # Plot condition A, trial 1, data vs t,
plot(t,EEGb[0], 'r')            # ... and the data from condition B, trial 1,
xlabel('Time [s]')              # Label the x-axis as time.
ylabel('Voltage [\mu V]')       # Label the y-axis as voltage.
title('EEG data from conditions A (blue) and B (red), Trial 1') # And give it a title.
# savefig('imgs/2-3')
show()

imshow(EEGa,                                   # Image the data from condition A.
           cmap='BuPu',                            # ... set the colormap (optional)
           extent=[t[0], t[-1], 1, ntrials],       # ... set axis limits (t[-1] represents the last element of t)
           aspect='auto',                          # ... set aspect ratio 
           origin='lower')                         # ... put origin in lower left corner
xlabel('Time[s]')                              # Label the axes
ylabel('Trial #')
colorbar()                                     # Show voltage to color mapping
vlines(0.25, 1, 1000, 'k', lw=2)               # Indicate stimulus onset with line
# savefig('imgs/2-4')
show()

imshow(EEGb,                                   # Image the data from condition A.
            cmap='BuPu',                            # ... set the colormap (optional)
           extent=[t[0], t[-1], 1, ntrials],       # ... set axis limits (t[-1] represents the last element of t)
           aspect='auto',                          # ... set aspect ratio 
           origin='lower')                         # ... put origin in lower left corner
xlabel('Time[s]')                              # Label the axes
ylabel('Trial #')
colorbar()                                     # Show voltage to color mapping
vlines(0.25, 1, 1000, 'k', lw=2)               # Indicate stimulus onset with line
# savefig('imgs/2-4')
show()

# Plotting the ERP for condition A
plot(t, EEGa.mean(0))        # Plot the ERP of condition A
plot([0.25, 0.25], [-0.3, 0.3], 'k', lw=2)    # Add a vertical line to indicate the stimulus time               
xlabel('Time [s]')           # Label the axes
ylabel('Voltage [$\mu V$]')
title('ERP of condition A')  # ... provide a title
# savefig('imgs/2-5')
show()                       # ... and show the plot

# Plotting the ERP for condition B
plot(t, EEGb.mean(0))        # Plot the ERP of condition A
plot([0.25, 0.25], [-0.3, 0.3], 'k', lw=2)    # Add a vertical line to indicate the stimulus time               
xlabel('Time [s]')           # Label the axes
ylabel('Voltage [$\mu V$]')
title('ERP of condition B')  # ... provide a title
# savefig('imgs/2-5')
show()                       # ... and show the plot

# Confidence intervals for ERP (Method 1: Using CLT)

# For condition A
mn_a = EEGa.mean(0)  # Compute the mean across trials (the ERP)
sd_a = EEGa.std(0)  # Compute the std across trials
sdmn_a= sd_a / sqrt(ntrials)  # Compute the std of the mean

# Plot 
fig = plt.figure(figsize=(12, 3))   # Save the axes for use in later cells and resize the figure
plt.plot(t, mn, 'k', lw=1.5)              # Plot the ERP of condition A
# ax.plot(t, mn + 2 * sdmn, 'k:', lw=1)  # ... and include the upper CI
plt.fill_between(t, mn_a - sdmn_a, mn_a + sdmn_a, color='k', alpha=0.2)
# ax.plot(t, mn - 2 * sdmn, 'k:', lw=1)  # ... and the lower C  
plt.plot([0.25, 0.25], [-0.4, 0.4], 'r:', lw=3)    # Add a vertical line to indicate the stimulus time               
plt.hlines(0, t[0], t[-1])
xlabel('Time [s]')                     # Label the axes
ylabel('Voltage [$\mu$ V]')
title('ERP of condition A')            # ... provide a useful title
fig                                    # ... and show the plot
show()


# For condition B
mn_b = EEGb.mean(0)  # Compute the mean across trials (the ERP)
sd_b = EEGb.std(0)  # Compute the std across trials
sdmn_b = sd_b / sqrt(ntrials)  # Compute the std of the mean

# Plot 
fig = plt.figure(figsize=(12, 3))   # Save the axes for use in later cells and resize the figure
plt.plot(t, mn_b, 'k', lw=1.5)              # Plot the ERP of condition A
# ax.plot(t, mn + 2 * sdmn, 'k:', lw=1)  # ... and include the upper CI
plt.fill_between(t, mn_b - sdmn_b, mn_b + sdmn_b, color='k', alpha=0.2)
# ax.plot(t, mn - 2 * sdmn, 'k:', lw=1)  # ... and the lower C  
plt.plot([0.25, 0.25], [-0.4, 0.4], 'r:', lw=3)    # Add a vertical line to indicate the stimulus time               
plt.hlines(0, t[0], t[-1])
xlabel('Time [s]')                     # Label the axes
ylabel('Voltage [$\mu$ V]')
title('ERP of condition B')            # ... provide a useful title


# Comparing ERPs

fig = plt.figure(figsize=(12, 3))   # Save the axes for use in later cells and resize the figure
plt.plot(t, mn_a, 'k', lw=1.5)              # Plot the ERP of condition A
plt.fill_between(t, mn_a - sdmn_a, mn_a + sdmn_a, color='k', alpha = 0.2)
plt.plot(t, mn_b, 'r', lw=1.5)              # Plot the ERP of condition B
plt.fill_between(t, mn_b - sdmn_b, mn_b + sdmn_b, color='r', alpha = 0.2)
plt.plot([0.25, 0.25], [-0.4, 0.4], 'b:', lw=3)    # Add a vertical line to indicate the stimulus time               
plt.hlines(0, t[0], t[-1])
xlabel('Time [s]')                     # Label the axes
ylabel('Voltage [$\mu$ V]')
title('Comparing ERPs between conditions A and B')            # ... provide a useful title

# Plotting the difference between ERP_A and ERP_B

mn_d = mn_a - mn_b     # Differenced ERP
sdmn_d = sqrt(sdmn_a ** 2 + sdmn_b ** 2)  # ... and its standard dev

fig = plt.figure(figsize=(12, 3))   # Save the axes for use in later cells and resize the figure
plt.plot(t, mn_d, 'k', lw=1.5)              # Plot the differenced ERP
plt.fill_between(t, mn_d - sdmn_d, mn_d + sdmn_d, color = 'k', alpha = 0.2)
plt.plot([0.25, 0.25], [-0.4, 0.4], 'r:', lw=3)    # Add a vertical line to indicate the stimulus time               
plt.hlines(0, t[0], t[-1])

# Confidence intervals for ERPs (Method 2: Bootstrapping)

# Draw 1000 integers with replacement from [0, 1000)
i = randint(0, ntrials, size=ntrials)
EEG0 = EEGa[i]  # Create the resampled EEG.
ERP0 = EEG0.mean(0)  # Create the resampled ERP

def bootstrapERP(EEGdata, size=None):  # Steps 1-2
    """ Calculate bootstrap ERP from data (array type)"""
    ntrials = len(EEGdata)             # Get the number of trials
    if size == None:                   # Unless the size is specified,
        size = ntrials                 # ... choose ntrials
    i = randint(ntrials, size=size)    # ... draw random trials,
    EEG0 = EEGdata[i]                  # ... create resampled EEG,
    return EEG0.mean(0)                # ... return resampled ERP.
                                       # Step 3: Repeat 3000 times 
ERP0 = [bootstrapERP(EEGa) for _ in range(10000)]
ERP0 = array(ERP0)                     # ... and convert the result to an array

# Determining CIs for the ERP
ERP0.sort(axis=0)         # Sort each column of the resampled ERP
N = len(ERP0)             # Define the number of samples
ciL = ERP0[int(0.025*N)]  # Determine the lower CI
ciU = ERP0[int(0.975*N)]  # ... and the upper CI
mn_a = EEGa.mean(0)        # Determine the ERP for condition A

fig = plt.figure(figsize=(12, 3))   # Save the axes for use in later cells and resize the figure
plt.plot(t, mn_a, 'k', lw=2)   # ... and plot it
plt.fill_between(t, mn_a - ciL, mn_a + ciU, color = 'k', alpha = 0.2)
plt.plot([0.25, 0.25], [-0.6, 0.6], 'r:', lw=3)    # Add a vertical line to indicate the stimulus time               
plt.hlines(0, t[0], t[-1])

mbA = mean(EEGa,0)          # Determine ERP for condition A
mnB = mean(EEGb,0)          # Determine ERP for condition B
mnD = mn_a - mn_b             # Compute the differenced ERP
stat = max(abs(mnD))        # Compute the statistic
print('stat = {:.4f}'.format(stat))

# A bootstrap test to differentiate between two conditions

EEG = vstack((EEGa, EEGb))                # Step 1. Merge EEG data from all trials
seed(123)                                 # For reproducibility

def bootstrapStat(EEG):                   # Steps 2-4.
    mn_a = bootstrapERP(EEG, size=ntrials) # Create resampled ERPa. The function 'bootstrapERP' is defined above!
    mn_b = bootstrapERP(EEG, size=ntrials) # Create resampled ERPb
    mn_d = mn_a - mn_a                       # Compute differenced ERP
    return max(abs(mn_d))                  # Return the statistic
                                          # Resample 3,000 times
statD = [bootstrapStat(EEG) for _ in range(3000)]

plt.hist(statD, bins = 10)
plt.plot([stat, stat], [0, 800], 'r:', lw=3) 
plt.xlim((0.140, 0.330))
plt.ylim((0, 800))

sum(statD > stat) / len(statD)
