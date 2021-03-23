# Notebook - Train.py
The model is trained to estimate early amount spent by the customer based on Time spent by customer on the App, session durations and his/her membership status.
The purpose of the project is to implement linear regression from scratch in Python without using sklearn library.




# HELPER FUNCTIONS

To read a csv file and convert into numpy array, you can use genfromtxt of the numpy package.
For Example:
```
train_data = np.genfromtxt(train_X_file_path, dtype=np.float64, delimiter=',', skip_header=1)
```
You can use the python csv module for writing data to csv files.
Refer to https://docs.python.org/2/library/csv.html.
For Example:
```
with open('sample_data.csv', 'w') as csv_file:
	writer = csv.writer(csv_file)
    writer.writerows(data)
```
