##	Changes

####	crop_microsgraphs.py

Adding codes:

>
    #Make the w valid when image size smaller than 448#
    if x<224:
        w=x
    elif y<224:
        w=y
    else:
        w=224


