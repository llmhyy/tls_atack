To run the program, use python pyrawcap.py Source.xlsx 3 output/
If you have both python 2 and python 3 installed, run with python 3 (usually py -3)

Paramters:
The excel file which contains a list of websites. It expects the 1st column to be the name of the website (any name is fine, the output will be named that) and the 3rd column to be the URL (include https)
Number of reptitions (times to run per website)
Output text to prepend to save file (i.e. output/) means save it under a folder named output.

It saves logging to application.log and any errors to errors.out. It might also create a file named failed websites with the ending date and time, those are the websites in which the query failed.