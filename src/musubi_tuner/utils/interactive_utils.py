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
        # Linux/macOS support
        try:
            import select
            import termios
            import tty

            if not sys.stdin.isatty():
                return False

            old_settings = termios.tcgetattr(sys.stdin)
            try:
                tty.setcbreak(sys.stdin.fileno())
                # Check if data is available to read
                if select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1)
                    if key.lower() == 's':
                        logger.info("Save requested by user (keyboard interrupt 's')")
                        return True
            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        except Exception:
            pass

    return False
