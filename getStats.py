import numpy as np
import matplotlib.pyplot as plt
import math

song_durations = [
    9.00, 3.17, 5.40, 8.10, 8.05, 5.43, 6.31, 15.40, 6.37, 9.52, 8.26, 5.56,
    10.30, 5.50, 5.13, 6.21, 10.59, 8.24, 4.28, 10.08
]

song_release_year = [
    2004, 2021, 2016, 2020, 2001, 2009, 2012, 2016, 2015, 2017, 2020, 2013,
    2005, 2014, 2022, 2005, 2023, 2017, 2008, 2023
]


def plot_durations(filename):
    # Round different song durations to nearest minute
    rounded_durations = []
    for song in song_durations:
        if song - int(song) < 0.30:
            rounded_durations.append(math.floor(song))
        else:
            rounded_durations.append(math.ceil(song))

    print("The rounded song durations: ", rounded_durations)

    # Start by setting up the histogram
    range_durations = max(rounded_durations) - min(rounded_durations)
    plt.hist(rounded_durations, edgecolor="dodgerblue", bins=range_durations)
    plt.xticks(range(min(rounded_durations), max(rounded_durations) + 1))

    plt.title("Distribution of song length")
    plt.xlabel("Duration (minutes)")
    plt.ylabel("Number of songs")

    plt.show()
    plt.savefig(filename)


def plot_release_year(filename):

    plt.hist(song_release_year,
             edgecolor="dodgerblue",
             bins=[2000, 2005, 2010, 2015, 2020, 2025])
    plt.xticks(range(2000, 2026, 5))

    plt.title("Distribution of song release years")
    plt.xlabel("Release year")
    plt.ylabel("Number of songs")

    plt.show()
    plt.savefig(filename)


# Check lengths of lists
if len(song_durations) != 20 or len(song_release_year) != 20:
    print("Something is wrong")

# Get song duration histogram
filename = "distribution.png"
plot_durations(filename)

# new window
plt.figure()

# Get release year histogram
filename2 = "distribution_year.png"
plot_release_year(filename2)