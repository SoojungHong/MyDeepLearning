import logging

# in the beginning of code, put following to use logger 
logger = logging.getLogger()
fhandler = logging.FileHandler(filename='train_log.log', mode='a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.INFO)  
logger.removeHandler(sys.stderr) # this remove should be done, otherwise log doesn't appear

# in code, call logging - depends on INFO, DEBUG, etc 
logging.info("epoch size : "+str(epoch_size)+" "+ str(datetime.now()) ) #SJ_TEST
        
        
