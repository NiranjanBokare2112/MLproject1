import logging
import logger
#Logging is a means of tracking ev ments that happen when some software runs. 
# Logging is important for software sdeveloping, debugging, and running.
import os
#provides a way to interact with the operating system. 
# It allows Python programs to perform various operating system-dependent functionalities in a portable manner. 

from datetime import datetime

LOG_file=f"logs_{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path=os.path.join(os.getcwd(),"logs",LOG_file)
os.makedirs(logs_path, exist_ok=True)
#exist_ok=True, it won't raise an error if the directory already exists.
LOG_File_PATH=os.path.join(logs_path,LOG_file)
#os.path.join() is used to join one or more path components intelligently.
logging.basicConfig(
    filename=LOG_File_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)  
#logging.basicConfig() function is used to configure the logging settings in Python.
# Here, we are setting the filename, format, and logging level.