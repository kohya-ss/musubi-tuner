import os
import sys
import logging

logger = logging.getLogger(__name__)

def is_save_requested():
    """
    Checks if the 's' key has been pressed in the console.
    This function is non-blocking and should be called periodically.
    """
    if os.name == 'nt':
        try:
            import msvcrt
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key.lower() == b's':
                    logger.info("Save requested by user (keyboard interrupt 's')")
                    return True
                # consume other keys
        except Exception:
            pass
    else:
        try:
            import fcntl
            import termios
            import tty

            fd = sys.stdin.fileno()
            if not os.isatty(fd):
                return False

            old_settings = termios.tcgetattr(fd)
            old_flags = fcntl.fcntl(fd, fcntl.F_GETFL)

            try:
                fcntl.fcntl(fd, fcntl.F_SETFL, old_flags | os.O_NONBLOCK)
                tty.setcbreak(fd)
                try:
                    key = sys.stdin.read(1)
                    if key and key.lower() == 's':
                        logger.info("Save requested by user (keyboard interrupt 's')")
                        try:
                            while sys.stdin.read(1):
                                pass
                        except IOError:
                            pass
                            
                        return True
                except IOError:
                    pass

            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                fcntl.fcntl(fd, fcntl.F_SETFL, old_flags)

        except Exception:
            pass

    return False
