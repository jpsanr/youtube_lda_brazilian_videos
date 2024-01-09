from google.cloud import logging
import time

class Logger():
    
    def __init__(self, use_gcp=False, log_name=None):
        self.use_gcp = use_gcp

        #logging_client = logging()
        if (self.use_gcp):
            logger = logging.logger(log_name)
             
    def log(self, texto):
        if (self.use_gcp):
            self.logger.log_text(str(texto))
        else:
            print(time.strftime("%H:%M:%S")+" - "+str(texto)) 

if __name__ == "__main__":
    lg = Logger()
    lg.log("Teste")

