from ctgan.logger import Logger
import os

import datetime

# test = datetime.datetime.now()
# print(test.strftime("%Y%m%d"))

logger = Logger()

print("Initial directory path:", logger.dirpath)
print(os.path.exists(logger.dirpath))

# print("CHANGE DIR PATH")
# logger.change_dirpath("/Users/twh/Desktop/test_folder")
# print("Current directory path:", logger.dirpath)

print(logger.datetimeformat)
print(logger.now.strftime(logger.datetimeformat))

print(logger.filename)

logger.write_to_file("Hello world")

logger.write_to_file("Hi there")
