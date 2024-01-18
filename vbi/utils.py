import time


def timer(func):
    '''
    decorator to measure elapsed time

    Parameters
    -----------
    func: function
        function to be decorated
    '''

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        display_time(end-start, message="{:s}".format(func.__name__))
        return result
    return wrapper

def display_time(time, message=""):
    '''
    display elapsed time in hours, minutes, seconds

    Parameters
    -----------
    time: float
        elaspsed time in seconds
    '''

    hour = int(time/3600)
    minute = (int(time % 3600))//60
    second = time-(3600.*hour+60.*minute)
    print("{:s} Done in {:d} hours {:d} minutes {:09.6f} seconds".format(
        message, hour, minute, second))
