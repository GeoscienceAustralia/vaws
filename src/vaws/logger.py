"""
    Setup logging - avoid simple print statements that can't be turned off
"""
import logging as pylog

LOGGING_NONE = 0
LOGGING_CONSOLE = 1
LOGGING_REGRESS = 2


# we hijack critical for our custom "regress" level
class NoCriticalFilter(pylog.Filter):
    def filter(self, record):
        if record.levelno >= pylog.CRITICAL:
            return 0
        return 1


class ConsoleLogger(object):
    def __init__(self, modName, level, filename):
        self.logger = pylog.getLogger(modName)
        self.logger.setLevel(pylog.WARNING)
        self.level = level
        self.filter = None
        self.handler_regress = None
        self.handler_console = None
        self.filename = filename
        
        if self.level == LOGGING_NONE:
            self.handler_console = pylog.StreamHandler()
            formatter = pylog.Formatter("%(name)s - %(message)s")
            self.handler_console.setFormatter(formatter)
            self.handler_console.setLevel(pylog.WARNING)
            self.filter = NoCriticalFilter()
            self.logger.addFilter(self.filter)
            self.logger.addHandler(self.handler_console)
            
        elif self.level == LOGGING_CONSOLE:
            self.logger.setLevel(pylog.DEBUG)
            self.handler_console = pylog.StreamHandler()
            formatter = pylog.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            self.handler_console.setFormatter(formatter)
            self.handler_console.setLevel(pylog.DEBUG)
            self.filter = NoCriticalFilter()
            self.logger.addFilter(self.filter)
            self.logger.addHandler(self.handler_console)
        
        elif self.level == LOGGING_REGRESS:
            print 'Setting up logger for regression...'
            self.handler_regress = pylog.FileHandler(self.filename, 'w')
            formatter = pylog.Formatter("%(message)s")
            self.handler_regress.setFormatter(formatter)
            self.handler_regress.setLevel(pylog.CRITICAL)
            self.logger.addHandler(self.handler_regress)
            
            self.handler_console = pylog.StreamHandler()
            formatter = pylog.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            self.handler_console.setFormatter(formatter)
            self.handler_console.setLevel(pylog.CRITICAL)
            self.logger.addHandler(self.handler_console)
            
    def reset(self):
        if self.filter: 
            self.logger.removeFilter(self.filter)
        if self.handler_regress:
            self.logger.removeHandler(self.handler_regress)
        if self.handler_console:
            self.logger.removeHandler(self.handler_console)
        
    def info(self, msg):
        self.logger.info(msg)
        
    def regress(self, msg):
        self.logger.critical(msg)
        
    def error(self, msg):
        self.logger.error(msg)
        
    def warn(self, msg):
        self.logger.warn(msg)
        
    def is_regress(self):
        return self.level == LOGGING_REGRESS


verbose = False
logger = None


def log():
    return logger


def configure(level, filename=None):
    global logger
    logger = ConsoleLogger('windsim', level, filename)
    
# unit tests
if __name__ == '__main__':
    import unittest
    
    class MyTestCase(unittest.TestCase):
        def test_basic(self):
            configure(LOGGING_CONSOLE)
            log().info("info message: %d" % (42))
            log().warn("this is your final warning!")
            log().error("You are in error, my friend")
            log().regress("regression value, %f" % (42))
            self.assertEquals(id(log()), id(log()))
    
    suite = unittest.TestLoader().loadTestsFromTestCase(MyTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)

    
    
    


