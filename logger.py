
import logging

"""You can use this basic one instead of you don't wanna name your logger.
import logging
logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug('debug message')
"""

def setLogger(name: str = 'SimpleLogger', 
              set_level: str = logging.DEBUG
              ):
    """Get logger object. \n
    Args:
        name (str): name of the logger.
            Example: 'ExampleLogger'
        set_level (str): level of the logger.
            Example: logging.DEBUG

    Returns:
        Logger: logger object.
        
    Example:
        logger = setLogger(name='ExampleLogger', set_level=logging.DEBUG)\n
        logger.debug('debug message')
    """

    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(set_level)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(set_level)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s: \n\t%(levelname)s: %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)
    
    return logger

# import these if use 'from logger import *'
__all__ = ['setLogger']

if __name__ == '__main__':
    logger = setLogger(name='ExampleLogger')
    logger.debug('debug message')
    logger.info('info message')
    logger.warning('warn message')
    logger.error('error message')
    logger.critical('critical message')
