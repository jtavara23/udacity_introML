#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    #"""
    # this way is sightly better
    for x in range(len(predictions)):
        cleaned_data.append((ages[x][0], net_worths[x][0],predictions[x][0] - net_worths[x][0]))
    cleaned_data.sort(key = lambda x: x[2]) #third element
    #"""
    """
    #    Another way of doing the same could be:
    import numpy
    errors = net_worths - predictions
    threshold = numpy.percentile(numpy.absolute(errors), 90)

    #print numpy.max(numpy.absolute(errors))
    #print numpy.sort(numpy.absolute(errors), axis=0)

    cleaned_data = [(age, net_worth, error) for age, net_worth, error in zip(ages, net_worths, errors) if abs(error) <= threshold]
    """

    return cleaned_data[:-10]

