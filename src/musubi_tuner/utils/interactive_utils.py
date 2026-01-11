import sys
import os
import threading
import atexit
import logging

logger = logging.getLogger(__name__)

_save_requested = False
_input_thread_started = False

def _restore_settings(fd, old_settings):
    try:
        import termios
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    except Exception:
        pass

def _input_listener():
    global _save_requested
    
    try:
        import tty
        import termios
        
        fd = sys.stdin.fileno()

        if not os.isatty(fd):
            return

        old_settings = termios.tcgetattr(fd)
        atexit.register(_restore_settings, fd, old_settings) # restores original terminal settings on exit
        tty.setcbreak(fd)
        
        while True:
            char = sys.stdin.read(1)
            
            if char.lower() == 's':
                logger.info("\nSave requested via 's' key")
                _save_requested = True

            if char == '\x03': # restores ctrl+c behaviour
                _restore_settings(fd, old_settings)
                import signal
                os.kill(os.getpid(), signal.SIGINT)
                break
                
    except Exception:
        pass

def start_listener():
    global _input_thread_started
    if _input_thread_started:
        return

    try:
        if sys.stdin.isatty():
            t = threading.Thread(target=_input_listener, daemon=True)
            t.start()
            _input_thread_started = True
    except Exception:
        pass

def is_save_requested():
    global _save_requested
    if os.name == 'nt':
        try:
            import msvcrt
            while msvcrt.kbhit():
                if msvcrt.getch().lower() == b's':
                    logger.info("Save requested by user")
                    return True
        except:
            pass
        return False

    else:
        if not _input_thread_started:
            start_listener()
        if _save_requested:
            _save_requested = False
            return True
    
    return False
